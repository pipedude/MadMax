import asyncio
import audioop
import fnmatch
import logging
import re
import time
from pathlib import Path

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from websockets.exceptions import ConnectionClosedError
from websockets.exceptions import ConnectionClosedOK

from config import (
    GOOGLE_API_KEY, GEMINI_MODEL_NAME, INSTRUCTIONS_FILE, SOUL_FILE, ENABLE_SOUL_IN_PROMPT, VOICE,
    VAD_START_SENSITIVITY, VAD_END_SENSITIVITY, VAD_SILENCE_DURATION_MS,
    AUDIO_RATE, LIVE_INPUT_RATE,
    VAD_PREFIX_PADDING_MS, INACTIVITY_TIMEOUT, READY_PHRASE, ENABLE_READY_PHRASE,
    LIVE_TOOL_RESUME_TIMEOUT_SECONDS,
    ENABLE_MANUAL_VAD, AUDIO_ENERGY_THRESHOLD,
    AGENT_FILES_DIR,
)
import json
from core.agent_tools import AgentToolExecutor
from memory_engine.memory_config import OUTPUT_JSON_PATH
from memory_engine.active_context_builder import build_compact_injection_context
from core.audio_io import AudioIO
from core.errors import FatalAPIError, InactivityTimeoutError, RealtimeAPIError, RecoverableConnectionError, ToolExecutionError
from core.session_transcript_logger import SessionTranscriptLogger

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

class GeminiLiveClient:
    def __init__(
        self,
        audio_io: AudioIO,
        session_transcript_logger: SessionTranscriptLogger,
        tool_executor: AgentToolExecutor | None = None,
    ):
        self.audio_io = audio_io
        self.session_transcript_logger = session_transcript_logger
        self.tool_executor = tool_executor
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        
        self.last_activity_time = None
        self.is_running = False
        self._current_user_transcript = ""
        self._current_agent_transcript = ""
        self._resample_state = None
        self._blocking_ops_count = 0
        self._input_audio_active = False
        self._input_audio_stream_ended = False
        self._last_user_speech_finished_at: float | None = None
        self._send_chunk_count = 0
        self._send_last_log_at = 0.0
        self._tool_resume_wait_started_at: float | None = None
        self._tool_resume_pending_ids: list[str] = []
        self._tool_resume_first_model_turn_logged = False
        self._tool_resume_first_audio_logged = False
        self._manual_speech_active = False
        self._manual_silence_chunks = 0
        
        # Instructions will be loaded asynchronously before start
        self.instructions = ""

    def _resolve_path(self, path_value: str) -> Path:
        path = Path(path_value)
        if not path.is_absolute():
            path = (_PROJECT_ROOT / path).resolve(strict=False)
        return path

    async def _load_base_instructions_async(self) -> str:
        instructions_path = self._resolve_path(INSTRUCTIONS_FILE)
        try:
            def _read():
                with instructions_path.open("r", encoding="utf-8") as file_handle:
                    return file_handle.read()
            return await asyncio.to_thread(_read)
        except FileNotFoundError:
            logger.warning("File %s not found. Using empty prompt.", instructions_path)
            return ""

    async def _load_soul_async(self) -> str:
        if not ENABLE_SOUL_IN_PROMPT:
            return ""
        soul_path = self._resolve_path(SOUL_FILE)
        try:
            def _read():
                with soul_path.open("r", encoding="utf-8") as file_handle:
                    return file_handle.read()
            return await asyncio.to_thread(_read)
        except FileNotFoundError:
            logger.debug("File %s not found. SOUL will not be injected.", soul_path)
            return ""

    async def _build_session_instructions_async(self) -> str:
        if not self.instructions:
            self.instructions = await self._load_base_instructions_async()

        parts = []

        # SOUL.md injection before instructions
        soul_text = await self._load_soul_async()
        if soul_text:
            parts.append(soul_text.strip())

        base_instructions = self.instructions.strip()
        if base_instructions:
            parts.append(base_instructions)

        # Asynchronous active_context.json injection (compact view)
        if OUTPUT_JSON_PATH.exists():
            try:
                def _read_context():
                    with OUTPUT_JSON_PATH.open("r", encoding="utf-8") as f:
                        return json.load(f)
                context_data = await asyncio.to_thread(_read_context)
                compact = build_compact_injection_context(context_data)
                parts.append(f"=== ACTIVE CONTEXT ===\n{compact}")
            except Exception as e:
                logger.error("Error injecting active_context.json: %s", e)

        return "\n\n".join(parts)

    def _reset_state(self):
        self.audio_io.reset_state()
        self.last_activity_time = asyncio.get_running_loop().time()
        self._current_user_transcript = ""
        self._current_agent_transcript = ""
        self._resample_state = None
        self._blocking_ops_count = 0
        self._input_audio_active = False
        self._input_audio_stream_ended = False
        self._last_user_speech_finished_at = None
        self._tool_resume_wait_started_at = None
        self._tool_resume_pending_ids = []
        self._tool_resume_first_model_turn_logged = False
        self._tool_resume_first_audio_logged = False
        self._manual_speech_active = False
        self._manual_silence_chunks = 0
        self._last_image_sent_name = None
        self._last_image_sent_time = 0.0

    async def _mark_model_response_started(self, session):
        was_not_receiving = not self.audio_io.is_receiving_response
        self.audio_io.is_receiving_response = True
        if was_not_receiving:
            self.audio_io.clear_mic_queue()
            if ENABLE_MANUAL_VAD:
                if self._manual_speech_active:
                    await session.send_realtime_input(activity_end=types.ActivityEnd())
                    self._manual_speech_active = False
                    self._manual_silence_chunks = 0
                    logger.debug("[Manual VAD] activity_end (model started responding)")
            elif self._input_audio_active and not self._input_audio_stream_ended:
                await session.send_realtime_input(audio_stream_end=True)
                self._input_audio_active = False
                self._input_audio_stream_ended = True
        self.last_activity_time = asyncio.get_running_loop().time()

    def _mark_model_response_finished(self):
        self.audio_io.is_receiving_response = False
        self.last_activity_time = asyncio.get_running_loop().time()

    def _flush_user_transcript(self):
        text = self._current_user_transcript.strip()
        if text:
            logger.info("user: %s", text)
            self.session_transcript_logger.log_user_message(text)
            self.last_activity_time = asyncio.get_running_loop().time()
        self._current_user_transcript = ""

    def _flush_agent_transcript(self):
        text = self._current_agent_transcript.strip()
        if text:
            logger.info("agent: %s", text)
            self.session_transcript_logger.log_agent_message(text)
            self.last_activity_time = asyncio.get_running_loop().time()
        self._current_agent_transcript = ""

    async def _try_send_image(self, session, text: str) -> None:
        if not text or not AGENT_FILES_DIR:
            return

        base_dir = Path(AGENT_FILES_DIR).resolve()
        if not base_dir.exists():
            return

        def _is_inside(path: Path) -> bool:
            try:
                return str(path.resolve()).startswith(str(base_dir) + "/")
            except (OSError, ValueError):
                return False

        # 1. Try explicit filename with extension first
        match = re.search(r"([^\s,;]+\.(?:jpg|jpeg|png|gif|webp))\b", text, re.IGNORECASE)
        target = None
        if match:
            raw_name = match.group(1)
            exact = (base_dir / raw_name).resolve()
            if _is_inside(exact):
                if exact.exists():
                    target = exact
                else:
                    stem = Path(raw_name).stem.lower()
                    for f in base_dir.iterdir():
                        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
                            if stem in f.stem.lower():
                                target = f
                                break
            if not target:
                await session.send_realtime_input(
                    text=f"Не нашёл файл {raw_name} в папке agent_files."
                )
                return
        else:
            # 2. No explicit extension — check if user mentioned a file stem
            # Require trigger words to avoid false positives on normal words
            text_lower = text.lower()
            trigger_words = (
                "посмотри", "открой", "покажи", "скинь", "фото", "фоточк",
                "картинк", "изображени", "файл", "look at", "open", "show",
                "image", "photo", "picture", "file", "send", "photo",
            )
            has_trigger = any(tw in text_lower for tw in trigger_words)
            if not has_trigger:
                return

            # Search for any file stem that appears in the text
            for f in base_dir.iterdir():
                if not f.is_file():
                    continue
                if f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".gif", ".webp"):
                    continue
                stem = f.stem.lower()
                # Also check simplified version (no hyphens/underscores)
                simplified = re.sub(r"[-_]", "", stem)
                text_simplified = re.sub(r"[-_]", "", text_lower)
                if stem in text_lower or simplified in text_simplified:
                    target = f
                    break
            if not target:
                return

        if not target or not target.exists():
            return

        # dedup: не слать одно и то же чаще чем раз в 5 сек
        now = time.monotonic()
        if self._last_image_sent_name == target.name and (now - self._last_image_sent_time) < 5.0:
            return

        mime_map = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp",
        }
        mime = mime_map.get(target.suffix.lower(), "image/jpeg")

        data = target.read_bytes()
        await session.send_realtime_input(
            video=types.Blob(data=data, mime_type=mime)
        )
        self._last_image_sent_name = target.name
        self._last_image_sent_time = now

    def _merge_transcript_chunk(self, current_text: str, incoming_text: str) -> str:
        current_text = current_text.strip()
        incoming_text = incoming_text.strip()

        if not incoming_text:
            return current_text
        if not current_text:
            return incoming_text
        if incoming_text == current_text:
            return current_text
        if incoming_text.startswith(current_text):
            return incoming_text
        if current_text.startswith(incoming_text):
            return current_text
        if current_text.endswith(incoming_text):
            return current_text
        return f"{current_text} {incoming_text}".strip()

    def _begin_blocking_operation(self):
        self._blocking_ops_count += 1
        self.last_activity_time = asyncio.get_running_loop().time()

    def _end_blocking_operation(self):
        if self._blocking_ops_count > 0:
            self._blocking_ops_count -= 1
        self.last_activity_time = asyncio.get_running_loop().time()

    def _arm_tool_resume_wait(self, function_response_ids: list[str]) -> None:
        self._tool_resume_wait_started_at = time.monotonic()
        self._tool_resume_pending_ids = function_response_ids
        self._tool_resume_first_model_turn_logged = False
        self._tool_resume_first_audio_logged = False

    def _clear_tool_resume_wait(self) -> None:
        self._tool_resume_wait_started_at = None
        self._tool_resume_pending_ids = []
        self._tool_resume_first_model_turn_logged = False
        self._tool_resume_first_audio_logged = False

    def _tool_resume_elapsed_ms(self) -> float | None:
        if self._tool_resume_wait_started_at is None:
            return None
        return (time.monotonic() - self._tool_resume_wait_started_at) * 1000.0

    def _recover_from_tool_resume_timeout(self) -> None:
        logger.warning(
            "Tool resume timeout detected: ids=%s timeout_s=%s receiving_response=%s playing_audio=%s. Resetting response state.",
            self._tool_resume_pending_ids,
            LIVE_TOOL_RESUME_TIMEOUT_SECONDS,
            self.audio_io.is_receiving_response,
            self.audio_io.is_playing_audio,
        )
        self.audio_io.clear_speaker_queue()
        self.audio_io.is_playing_audio = False
        self._current_agent_transcript = ""
        self._mark_model_response_finished()
        self._clear_tool_resume_wait()

    def _summarize_tool_payload(self, payload: object, *, limit: int = 1000) -> str:
        try:
            serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        except TypeError:
            serialized = repr(payload)
        if len(serialized) > limit:
            return f"{serialized[:limit]}..."
        return serialized

    async def _handle_tool_call(self, session, tool_call) -> None:
        function_calls = getattr(tool_call, "function_calls", None) or []
        if not function_calls:
            return
        if self.tool_executor is None or not self.tool_executor.has_tools():
            raise ToolExecutionError("Model requested a tool but executor is not configured.")

        self._begin_blocking_operation()
        try:
            function_responses: list[types.FunctionResponse] = []
            for function_call in function_calls:
                logger.info(
                    "Tool call requested: id=%s name=%s args=%s",
                    function_call.id,
                    function_call.name,
                    self._summarize_tool_payload(function_call.args),
                )
                try:
                    result = await self.tool_executor.execute(function_call.name, function_call.args)
                    logger.info(
                        "Tool call completed: id=%s name=%s result=%s",
                        function_call.id,
                        function_call.name,
                        self._summarize_tool_payload(result),
                    )
                except ToolExecutionError as exc:
                    logger.warning(
                        "Tool error id=%s name=%s args=%s error=%s",
                        function_call.id,
                        function_call.name,
                        self._summarize_tool_payload(function_call.args),
                        exc,
                    )
                    result = {
                        "ok": False,
                        "error": {
                            "code": "TOOL_EXECUTION_ERROR",
                            "message": str(exc),
                        },
                    }
                function_responses.append(
                    types.FunctionResponse(
                        id=function_call.id,
                        name=function_call.name,
                        response=result,
                    )
                )
            logger.info(
                "send_tool_response started: count=%s ids=%s",
                len(function_responses),
                [response.id for response in function_responses],
            )
            send_started_at = time.monotonic()
            await session.send_tool_response(function_responses=function_responses)
            send_elapsed_ms = (time.monotonic() - send_started_at) * 1000.0
            function_response_ids = [response.id for response in function_responses]
            logger.info(
                "send_tool_response finished: count=%s ids=%s duration_ms=%.1f",
                len(function_responses),
                function_response_ids,
                send_elapsed_ms,
            )
            self._arm_tool_resume_wait(function_response_ids)
        finally:
            self._end_blocking_operation()

    def _classify_runtime_error(self, exc: Exception) -> Exception:
        if isinstance(exc, FatalAPIError):
            return exc
        if isinstance(exc, RecoverableConnectionError):
            return exc
        if isinstance(exc, genai_errors.APIError):
            status = getattr(exc, "status", None)
            code = getattr(exc, "code", None)
            if status in {400, 401, 403, 404, 1008}:
                return FatalAPIError(f"Fatal Gemini API error ({status}): {exc}")
            if status in {408, 429, 500, 502, 503, 504, 1006, 1011} or code in {1006, 1011} or "1006" in str(exc) or "1011" in str(exc):
                return RecoverableConnectionError(f"Temporary Gemini API error ({status or code}): {exc}")
            return RealtimeAPIError(f"Unhandled Gemini API error ({status}): {exc}")
        if isinstance(exc, (TimeoutError, OSError, ConnectionError, ConnectionClosedError)):
            return RecoverableConnectionError(f"Gemini Live network error: {exc}")
        return RealtimeAPIError(f"Unhandled Gemini Live runtime error: {exc}")

    async def connect_and_run(self):
        session_instructions = await self._build_session_instructions_async()

        # Convert string config settings to SDK 1.73.1 Enum
        def get_start_sensitivity(val):
            val = val.upper()
            if val == "HIGH": return types.StartSensitivity.START_SENSITIVITY_HIGH
            if val == "LOW": return types.StartSensitivity.START_SENSITIVITY_LOW
            return types.StartSensitivity.START_SENSITIVITY_UNSPECIFIED

        def get_end_sensitivity(val):
            val = val.upper()
            if val == "HIGH": return types.EndSensitivity.END_SENSITIVITY_HIGH
            if val == "LOW": return types.EndSensitivity.END_SENSITIVITY_LOW
            return types.EndSensitivity.END_SENSITIVITY_UNSPECIFIED

        # VAD configuration for SDK 1.73.1
        vad_config = types.AutomaticActivityDetection(
            start_of_speech_sensitivity=get_start_sensitivity(VAD_START_SENSITIVITY),
            end_of_speech_sensitivity=get_end_sensitivity(VAD_END_SENSITIVITY),
            silence_duration_ms=VAD_SILENCE_DURATION_MS,
            prefix_padding_ms=VAD_PREFIX_PADDING_MS
        )

        # Build tool list: Google Search + function declarations from executor
        tool_list = [{"google_search": {}}]
        if self.tool_executor:
            tool_list.extend(self.tool_executor.get_tool_definitions())

        # Session configuration for SDK 1.73.1 using explicit types
        config = types.LiveConnectConfig(
            system_instruction=types.Content(
                parts=[types.Part(text=session_instructions)]
            ),
            response_modalities=["AUDIO"],
            tools=tool_list,
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=VOICE
                    )
                )
            ),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=None if ENABLE_MANUAL_VAD else vad_config
            )
        )

        retry_delay = 1
        max_delay = 30
        
        while True:
            session_tasks = []
            try:
                logger.info("Connecting to Gemini Live API (%s)...", GEMINI_MODEL_NAME)
                async with self.client.aio.live.connect(model=GEMINI_MODEL_NAME, config=config) as session:
                    self.is_running = True
                    self._reset_state()
                    logger.info("Connected. Max is ready to talk!")

                    if ENABLE_READY_PHRASE and READY_PHRASE:
                        self.audio_io.is_receiving_response = True
                        logger.info("Sending ready phrase prompt: '%s'", READY_PHRASE)
                        try:
                            await session.send_realtime_input(
                                text=f'Say exactly the following phrase without additions or changes: "{READY_PHRASE}"'
                            )
                            logger.info("Ready phrase prompt sent successfully")
                        except Exception as exc:
                            logger.error("Failed to send ready phrase: %s", exc)

                    send_task = asyncio.create_task(self._send_loop(session))
                    receive_task = asyncio.create_task(self._receive_loop(session))
                    watchdog_task = asyncio.create_task(self._watchdog())
                    session_tasks = [send_task, receive_task, watchdog_task]

                    done, pending = await asyncio.wait(
                        session_tasks,
                        return_when=asyncio.FIRST_COMPLETED
                    )

                    self.is_running = False
                    for task in pending:
                        task.cancel()
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)

                    for task in done:
                        exc = task.exception()
                        if exc:
                            raise exc
                    
                    raise RecoverableConnectionError("Session ended by server.")

            except InactivityTimeoutError:
                logger.info("Inactivity timer (%s sec) expired. Going to sleep.", INACTIVITY_TIMEOUT)
                return
            except asyncio.CancelledError:
                self.is_running = False
                raise
            except FatalAPIError:
                raise
            except Exception as e:
                classified_error = self._classify_runtime_error(e)
                if isinstance(classified_error, FatalAPIError):
                    raise classified_error
                if isinstance(classified_error, RecoverableConnectionError):
                    logger.warning("%s. Retry in %s sec...", classified_error, retry_delay)
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, max_delay)
                    continue
                raise classified_error
            finally:
                self.is_running = False
                for task in session_tasks:
                    if not task.done():
                        task.cancel()
                if session_tasks:
                    await asyncio.gather(*session_tasks, return_exceptions=True)

    async def _send_loop(self, session):
        """Send microphone audio."""
        logger.info("Send loop started.")
        while self.is_running:
            try:
                audio_data = await self.audio_io.mic_queue.get()
                # Barge-in support: do NOT block on is_receiving_response
                # if self.audio_io.is_receiving_response:
                #     logger.debug("[SendLoop] skip chunk: is_receiving_response=True")
                #     continue
                if self.audio_io.is_playing_audio:
                    logger.debug("[SendLoop] skip chunk: is_playing_audio=True")
                    continue

                rms = audioop.rms(audio_data, 2)
                is_speech = rms > AUDIO_ENERGY_THRESHOLD
                if is_speech:
                    self.last_activity_time = asyncio.get_running_loop().time()

                if ENABLE_MANUAL_VAD:
                    if is_speech:
                        self._manual_silence_chunks = 0
                        if not self._manual_speech_active:
                            await session.send_realtime_input(activity_start=types.ActivityStart())
                            self._manual_speech_active = True
                    else:
                        if self._manual_speech_active:
                            self._manual_silence_chunks += 1
                            silence_threshold = int(VAD_SILENCE_DURATION_MS / 1000 * 10)
                            if self._manual_silence_chunks >= silence_threshold:
                                await session.send_realtime_input(activity_end=types.ActivityEnd())
                                self._manual_speech_active = False

                if AUDIO_RATE != LIVE_INPUT_RATE:
                    audio_data, self._resample_state = audioop.ratecv(
                        audio_data,
                        2,
                        1,
                        AUDIO_RATE,
                        LIVE_INPUT_RATE,
                        self._resample_state,
                    )
                await session.send_realtime_input(
                    audio=types.Blob(
                        data=audio_data,
                        mime_type=f"audio/pcm;rate={LIVE_INPUT_RATE}",
                    )
                )
                self._input_audio_active = True
                self._input_audio_stream_ended = False
                self._send_chunk_count += 1
                now = time.monotonic()
                if now - self._send_last_log_at >= 2.0:
                    logger.info("[SendLoop] audio_chunks_sent=%s last_2s", self._send_chunk_count)
                    self._send_last_log_at = now
            except asyncio.CancelledError:
                break
            except ConnectionClosedOK:
                break

    async def _receive_loop(self, session):
        """Process incoming messages from Gemini."""
        logger.info("Receive loop started.")
        while self.is_running:
            try:
                async for message in session.receive():
                    if not self.is_running:
                        break
                    
                    # Full message-type logging for diagnostics
                    msg_types = []
                    if message.server_content:
                        sc = message.server_content
                        if sc.model_turn:
                            msg_types.append("model_turn")
                            if self._tool_resume_wait_started_at is not None and not self._tool_resume_first_model_turn_logged:
                                elapsed_ms = self._tool_resume_elapsed_ms()
                                logger.info(
                                    "Tool resume first model_turn: ids=%s elapsed_ms=%.1f",
                                    self._tool_resume_pending_ids,
                                    elapsed_ms or 0.0,
                                )
                                self._tool_resume_first_model_turn_logged = True
                        if sc.input_transcription:
                            msg_types.append(f"input_transcription(finished={sc.input_transcription.finished})")
                            if sc.input_transcription.finished:
                                self._last_user_speech_finished_at = time.monotonic()
                                logger.info("[ReceiveLoop] user speech finished")
                        if sc.model_turn and self._last_user_speech_finished_at is not None:
                            latency_ms = (time.monotonic() - self._last_user_speech_finished_at) * 1000
                            logger.info("[ReceiveLoop] model_turn started after %.1f ms", latency_ms)
                            self._last_user_speech_finished_at = None
                        if sc.output_transcription:
                            msg_types.append(f"output_transcription(finished={sc.output_transcription.finished})")
                        if sc.generation_complete:
                            msg_types.append("generation_complete")
                        if sc.turn_complete:
                            msg_types.append("turn_complete")
                        if sc.interrupted:
                            msg_types.append("interrupted")
                    if message.tool_call:
                        msg_types.append("tool_call")
                    if msg_types:
                        logger.info("[ReceiveLoop] message types: %s", ", ".join(msg_types))
                    else:
                        logger.debug("[ReceiveLoop] unknown message: %s", message)

                    if message.tool_call:
                        await self._handle_tool_call(session, message.tool_call)
                        continue
                        
                    # 1. Process user transcription
                    if message.server_content and message.server_content.input_transcription:
                        transcription = message.server_content.input_transcription
                        if transcription.text:
                            self._current_user_transcript = self._merge_transcript_chunk(
                                self._current_user_transcript,
                                transcription.text,
                            )
                        self.last_activity_time = asyncio.get_running_loop().time()
                        if transcription.finished and self._current_user_transcript:
                            await self._try_send_image(session, self._current_user_transcript)
                            self._flush_user_transcript()

                    # 2. Process agent response audio and transcription
                    if message.server_content and message.server_content.model_turn:
                        parts = message.server_content.model_turn.parts or []
                        for part in parts:
                            if part.inline_data:
                                if self._tool_resume_wait_started_at is not None and not self._tool_resume_first_audio_logged:
                                    elapsed_ms = self._tool_resume_elapsed_ms()
                                    logger.info(
                                        "Tool resume first audio chunk: ids=%s elapsed_ms=%.1f bytes=%s",
                                        self._tool_resume_pending_ids,
                                        elapsed_ms or 0.0,
                                        len(part.inline_data.data),
                                    )
                                    self._tool_resume_first_audio_logged = True
                                await self._mark_model_response_started(session)
                                self.audio_io.enqueue_speaker_audio(part.inline_data.data)

                    if message.server_content and message.server_content.output_transcription:
                        transcription = message.server_content.output_transcription
                        if transcription.text:
                            self._current_agent_transcript = self._merge_transcript_chunk(
                                self._current_agent_transcript,
                                transcription.text,
                            )
                        if transcription.finished and self._current_agent_transcript:
                            self._flush_agent_transcript()

                    if message.server_content and message.server_content.generation_complete:
                        self._mark_model_response_finished()

                    # 3. End of turn
                    if message.server_content and message.server_content.turn_complete:
                         if self._tool_resume_wait_started_at is not None:
                             elapsed_ms = self._tool_resume_elapsed_ms()
                             logger.info(
                                 "Tool resume turn_complete: ids=%s elapsed_ms=%.1f",
                                 self._tool_resume_pending_ids,
                                 elapsed_ms or 0.0,
                             )
                             self._clear_tool_resume_wait()
                         logger.info("[ReceiveLoop] turn_complete")
                         self._flush_user_transcript()
                         self._flush_agent_transcript()
                         self._mark_model_response_finished()

                    # 4. Handle interruptions (Barge-in)
                    if message.server_content and message.server_content.interrupted:
                        logger.debug("Interruption: user started speaking.")
                        self._clear_tool_resume_wait()
                        self._flush_agent_transcript()
                        self.audio_io.clear_speaker_queue()
                        self.audio_io.is_playing_audio = False
                        self._mark_model_response_finished()
            except genai_errors.APIError as exc:
                if exc.status == 1000:
                    break
                raise

    async def _watchdog(self):
        """Inactivity timer."""
        while self.is_running:
            await asyncio.sleep(1)
            now = asyncio.get_running_loop().time()
            if self._tool_resume_wait_started_at is not None:
                elapsed_s = time.monotonic() - self._tool_resume_wait_started_at
                if elapsed_s > LIVE_TOOL_RESUME_TIMEOUT_SECONDS:
                    self._recover_from_tool_resume_timeout()
            
            if self._blocking_ops_count > 0:
                continue
            if not self.audio_io.is_receiving_response and not self.audio_io.is_playing_audio and self.last_activity_time:
                if (now - self.last_activity_time) > INACTIVITY_TIMEOUT:
                    raise InactivityTimeoutError()
