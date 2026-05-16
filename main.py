import asyncio
import sys
import logging
from vosk import Model

from core.audio_io import AudioIO
from core.agent_tools import build_default_agent_tool_executor
from core.gemini_client import GeminiLiveClient
from core.session_transcript_logger import SessionTranscriptLogger
from core.orchestrator import AgentOrchestrator
from config import GOOGLE_API_KEY, VOSK_MODEL_PATH
from memory_engine.memory_config import DAILY_LOGS_DIR
from core.errors import AudioInitializationError, ConfigurationError, RealtimeAPIError, WakeWordModelError


logger = logging.getLogger(__name__)

async def main():
    if not GOOGLE_API_KEY:
        raise ConfigurationError("GOOGLE_API_KEY not found. Check your .env file.")

    logger.info("Loading local Vosk model (may take a couple of seconds)...")
    try:
        vosk_model = Model(VOSK_MODEL_PATH)
    except (OSError, RuntimeError, ValueError) as exc:
        raise WakeWordModelError(
            f"Failed to load Vosk model from '{VOSK_MODEL_PATH}'. Run ./setup.sh or download vosk-model-small-en-us-0.15, extract and rename the folder to '{VOSK_MODEL_PATH}'."
        ) from exc

    loop = asyncio.get_running_loop()
    audio_io = AudioIO(loop=loop)
    audio_io.start()
    
    transcript_logger = SessionTranscriptLogger(base_directory=DAILY_LOGS_DIR)
    tool_executor = build_default_agent_tool_executor()
    client = GeminiLiveClient(
        audio_io=audio_io,
        session_transcript_logger=transcript_logger,
        tool_executor=tool_executor,
    )
    
    # Orchestrator initialization
    orchestrator = AgentOrchestrator(
        audio_io=audio_io,
        vosk_model=vosk_model,
        gemini_client=client,
        transcript_logger=transcript_logger
    )

    try:
        await orchestrator.run_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        audio_io.close()

def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

if __name__ == "__main__":
    try:
        setup_logging()
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Program terminated by user.")
    except (ConfigurationError, WakeWordModelError, AudioInitializationError, RealtimeAPIError) as exc:
        logger.error("%s", exc)
        sys.exit(1)