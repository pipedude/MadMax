import asyncio
import json
import logging
import time
from datetime import datetime, timezone
import queue
from vosk import Model, KaldiRecognizer

from core.audio_io import AudioIO
from core.gemini_client import GeminiLiveClient
from core.session_transcript_logger import SessionTranscriptLogger
from core.state import AgentMode
from config import AUDIO_RATE, POST_SESSION_TASK_TIMEOUT_SECONDS, WAIT_FOR_POST_SESSION_TASKS, WAKE_WORD
from memory_engine.active_context_builder import main_async as build_memory_context_async, get_local_today_date
from memory_engine.long_memory_extractor_agent import process_missing_sessions
from memory_engine.memory_config import OUTPUT_JSON_PATH

logger = logging.getLogger(__name__)

class AgentOrchestrator:
    def __init__(
        self, 
        audio_io: AudioIO, 
        vosk_model: Model, 
        gemini_client: GeminiLiveClient,
        transcript_logger: SessionTranscriptLogger
    ):
        self.audio_io = audio_io
        self.vosk_model = vosk_model
        self.gemini_client = gemini_client
        self.transcript_logger = transcript_logger
        self.running = False
        self._pending_memory_update_date: str | None = None
        self._pending_memory_update_session_id: str | None = None

    def _clear_pending_post_session_state(self) -> None:
        self._pending_memory_update_date = None
        self._pending_memory_update_session_id = None

    async def run_forever(self):
        """Main agent state management loop."""
        self.running = True
        logger.info("Orchestrator started. Entering main loop.")
        
        try:
            while self.running:
                try:
                    # 1. Sleep Mode (waiting for wake-word)
                    await self._wait_for_wake_word()
                    
                    # 2. New-day check and context update
                    await self._check_and_update_new_day()
                    
                    # 3. Active Mode (Gemini Live)
                    await self._run_active_session()
                finally:
                    # 4. Post-session memory processing (always runs, even on Ctrl+C)
                    await self._perform_post_session_tasks()
                
        except asyncio.CancelledError:
            logger.info("Orchestrator work cancelled.")
        finally:
            self.running = False

    async def _wait_for_wake_word(self):
        """Wake-word wait logic."""
        logger.info("[Sleep Mode] Waiting for command...")
        await asyncio.to_thread(self._run_vosk_loop)
        self.audio_io.play_beep()

    def _run_vosk_loop(self):
        """Synchronous Vosk loop to run in a thread."""
        target_word = WAKE_WORD.lower()
        rec = KaldiRecognizer(self.vosk_model, AUDIO_RATE)
        self.audio_io.routing_mode = AgentMode.WAKEWORD
        self.audio_io.clear_wakeword_queue()

        while self.audio_io.running:
            try:
                chunk = self.audio_io.ww_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if rec.AcceptWaveform(chunk):
                result = json.loads(rec.Result())
                text = result.get("text", "").strip()
                if text:
                    logger.info("[Vosk FINAL] %s", text)
            else:
                partial = json.loads(rec.PartialResult())
                text = partial.get("partial", "").strip()
            
            if target_word in text:
                logger.info("[Trigger] Heard: '%s'. Waking up!", text)
                return

    async def _check_and_update_new_day(self):
        """Logic for handling a new-day transition."""
        if not OUTPUT_JSON_PATH.exists():
            logger.info("Memory context not found, creating new one...")
            await build_memory_context_async()
            return

        try:
            def _read_date():
                with open(OUTPUT_JSON_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("context_date")
            
            last_date = await asyncio.to_thread(_read_date)
            today_date = get_local_today_date()

            if last_date != today_date:
                logger.info("Day change detected (%s -> %s). Rebuilding context...", last_date, today_date)
                await build_memory_context_async()
        except Exception as e:
            logger.error("Error during day-change check: %s", e)

    async def _run_active_session(self):
        """Start active Gemini conversation session."""
        self.audio_io.routing_mode = AgentMode.API
        
        # Save metadata before start for later extraction
        self._pending_memory_update_date = datetime.now().strftime("%Y-%m-%d")
        self.transcript_logger.start_session(datetime.now(timezone.utc))
        self._pending_memory_update_session_id = self.transcript_logger.current_session_id

        try:
            await self.gemini_client.connect_and_run()
        finally:
            await self.transcript_logger.flush_session_async()

    async def _perform_post_session_tasks(self):
        """Memory processing tasks after session ends."""
        pending_date = self._pending_memory_update_date
        if pending_date is None:
            return

        pending_session_id = self._pending_memory_update_session_id

        if not WAIT_FOR_POST_SESSION_TASKS:
            logger.info(
                "Post-session processing deferred by config for session_id=%s.",
                pending_session_id,
            )
            self._clear_pending_post_session_state()
            return

        deadline = None
        if POST_SESSION_TASK_TIMEOUT_SECONDS > 0:
            deadline = time.monotonic() + POST_SESSION_TASK_TIMEOUT_SECONDS

        logger.info(
            "Updating long-term memory (checking missed sessions) for session_id=%s...",
            pending_session_id,
        )
        try:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning(
                        "Skipping post-session tasks due to budget timeout for session_id=%s.",
                        pending_session_id,
                    )
                    self._clear_pending_post_session_state()
                    return
                await asyncio.wait_for(
                    process_missing_sessions(pending_date),
                    timeout=remaining,
                )
            else:
                await process_missing_sessions(pending_date)

            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning(
                        "Skipping active context rebuild because post-session budget exhausted for session_id=%s.",
                        pending_session_id,
                    )
                    self._clear_pending_post_session_state()
                    return
                await asyncio.wait_for(
                    build_memory_context_async(),
                    timeout=remaining,
                )
            else:
                await build_memory_context_async()

            self._clear_pending_post_session_state()
            logger.info("Memory successfully updated.")
        except asyncio.CancelledError:
            logger.info(
                "Post-session task cancelled for session_id=%s.",
                pending_session_id,
            )
            self._clear_pending_post_session_state()
            raise
        except asyncio.TimeoutError:
            logger.warning(
                "Post-session task interrupted by budget timeout for session_id=%s.",
                pending_session_id,
            )
            self._clear_pending_post_session_state()
        except Exception as e:
            logger.error("Error updating memory after session: %s", e)
