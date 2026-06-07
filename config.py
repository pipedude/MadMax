import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# ==========================================
# BASIC API SETTINGS
# ==========================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-3.1-flash-live-preview")

# ==========================================
# AUDIO AND SYSTEM SETTINGS
# ==========================================
# Gemini 3.1 Live API works great with 24kHz
AUDIO_RATE = 24000
LIVE_INPUT_RATE = int(os.getenv("LIVE_INPUT_RATE", "16000"))
AUDIO_CHANNELS = 1
AUDIO_CHUNK = 960
AUDIO_PLAYBACK_DIAGNOSTICS = os.getenv("AUDIO_PLAYBACK_DIAGNOSTICS", "false").lower() in {"1", "true", "yes", "on"}

# ==========================================
# AGENT AND SESSION SETTINGS
# ==========================================
INSTRUCTIONS_FILE = "agent_instructions.md"
SOUL_FILE = "SOUL.md"
ENABLE_SOUL_IN_PROMPT = os.getenv("ENABLE_SOUL_IN_PROMPT", "true").lower() in {"1", "true", "yes", "on"}
VOICE = os.getenv("VOICE", "Puck") # Gemini has its own voices: Puck, Charon, Kore, Fenrir, Aoede

# ==========================================
# VAD SETTINGS (VOICE ACTIVITY DETECTOR)
# ==========================================
# Sensitivity: "HIGH", "LOW" or "OFF"
VAD_START_SENSITIVITY = os.getenv("VAD_START_SENSITIVITY", "LOW")
VAD_END_SENSITIVITY = os.getenv("VAD_END_SENSITIVITY", "LOW")
VAD_SILENCE_DURATION_MS = int(os.getenv("VAD_SILENCE_DURATION_MS", "1000"))
VAD_PREFIX_PADDING_MS = int(os.getenv("VAD_PREFIX_PADDING_MS", "300"))

# ==========================================
# WAKE-WORD (VOSK) AND SILENCE TIMER SETTINGS
# ==========================================
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH", "model_en")
WAKE_WORD = os.getenv("WAKE_WORD", "Max")
READY_PHRASE = os.getenv("READY_PHRASE", "Listening!")
ENABLE_READY_PHRASE = os.getenv("ENABLE_READY_PHRASE", "true").lower() in {"1", "true", "yes", "on"}

# Time in seconds after which the agent will sleep if there is no speech
INACTIVITY_TIMEOUT = int(os.getenv("INACTIVITY_TIMEOUT", "25"))
WAIT_FOR_POST_SESSION_TASKS = os.getenv("WAIT_FOR_POST_SESSION_TASKS", "true").lower() in {"1", "true", "yes", "on"}
LIVE_TOOL_RESUME_TIMEOUT_SECONDS = int(os.getenv("LIVE_TOOL_RESUME_TIMEOUT_SECONDS", "12"))
NON_LIVE_LLM_TIMEOUT_SECONDS = int(os.getenv("NON_LIVE_LLM_TIMEOUT_SECONDS", "60"))
POST_SESSION_TASK_TIMEOUT_SECONDS = int(
    os.getenv(
        "POST_SESSION_TASK_TIMEOUT_SECONDS",
        str(max(NON_LIVE_LLM_TIMEOUT_SECONDS + 5, 1)),
    )
)

# ==========================================
# MANUAL VAD (for testing)
# ==========================================
ENABLE_MANUAL_VAD = os.getenv("ENABLE_MANUAL_VAD", "false").lower() in {"1", "true", "yes", "on"}

# Local energy threshold for speech detection (RMS of 16-bit PCM)
AUDIO_ENERGY_THRESHOLD = int(os.getenv("AUDIO_ENERGY_THRESHOLD", "500"))

# ==========================================
# AGENT FILES (IMAGES)
# ==========================================
AGENT_FILES_DIR = os.getenv("AGENT_FILES_DIR", "./agent_files/")