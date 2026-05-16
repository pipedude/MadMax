from datetime import datetime
from pathlib import Path

# ==============================================================================
# ENVIRONMENT & BASE PATHS
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent

# Automatic timezone detection (e.g. 'UTC+07:00')
local_tz_offset = datetime.now().astimezone().strftime('%z')
# Format +0700 into +07:00
LONG_MEMORY_DEFAULT_TIMEZONE = f"UTC{local_tz_offset[:3]}:{local_tz_offset[3:]}"

# Daily log filename template (used for session search)
MARKDOWN_FILE_TEMPLATE = "{date}.md"

# Daily logs directory
DAILY_LOGS_DIR = BASE_DIR / "daily"


# ==============================================================================
# CONTEXT SUMMARIZATION (SHORT-TERM)
# ==============================================================================
# Path to active context file for the current session
OUTPUT_JSON_PATH = BASE_DIR / "active_context.json"

# Limit of recent utterances included in summarization
RECENT_REPLIES_LIMIT = 20

# Model for short-term context summarization
SUMMARIZE_CONTEXT_GEMINI_MODEL = "gemini-3.1-flash-lite"
SUMMARIZE_CONTEXT_GEMINI_THINKING_LEVEL = "minimal"


# ==============================================================================
# LLM MODELS (LONG-TERM MEMORY)
# ==============================================================================
# Main model for memory operation extraction
LONG_MEMORY_GEMINI_MODEL = "gemini-3.1-flash-lite"
LONG_MEMORY_GEMINI_THINKING_LEVEL = "minimal"

# Specialized model for conflict resolution and fact merging (Surgeon)
LONG_MEMORY_SURGEON_GEMINI_MODEL = "gemini-3.1-flash-lite"
LONG_MEMORY_SURGEON_GEMINI_THINKING_LEVEL = "minimal"


# ==============================================================================
# EXTRACTION RULES
# ==============================================================================
# Minimum confidence threshold at which a fact is extracted from text
LONG_MEMORY_MIN_CONFIDENCE_EXTRACT = 0.75

# Minimum number of meaningful letters in a token (ignore too-short words)
LONG_MEMORY_MIN_TOKEN_LENGTH = 3

# Maximum number of entities (people + places) injected into the extractor prompt
LONG_MEMORY_EXTRACTOR_ENTITY_CONTEXT_LIMIT = 50


# ==============================================================================
# MATCHING & DEDUPLICATION (NORMALIZER)
# ==============================================================================
# Exact match threshold (fuzzy matching) above which data is updated without LLM involvement
LONG_MEMORY_EXACT_MATCH_THRESHOLD = 0.98

# Forced similarity boost when one string fully contains another
LONG_MEMORY_FACT_CONTAINS_SIMILARITY_FLOOR = 0.95
LONG_MEMORY_GOAL_CONTAINS_SIMILARITY_FLOOR = 0.9


# ==============================================================================
# MERGE LOGIC (LLM SURGEON)
# ==============================================================================
# Maximum number of conflicts sent in a single batch to the LLM Surgeon
LONG_MEMORY_SURGEON_BATCH_SIZE = 15

# Self-perception history limit in reflections.json
LONG_MEMORY_MAX_PERSONA_HISTORY = 50

# Current roles history limit in reflections.json
LONG_MEMORY_MAX_PERSONA_ROLES = 20

# Number of persona roles and insights included in active context
ACTIVE_CONTEXT_PERSONA_ROLES_LIMIT = 3
ACTIVE_CONTEXT_PERSONA_INSIGHTS_LIMIT = 3

# Number of episodes included in active context
ACTIVE_CONTEXT_EPISODES_LIMIT = 3

# Experience records limit in experience.json
LONG_MEMORY_MAX_EXPERIENCE_RECORDS = 50

# Number of experience records included in active context
ACTIVE_CONTEXT_EXPERIENCE_LIMIT = 3

# Minimum confidence for experience to be included in active context
ACTIVE_CONTEXT_EXPERIENCE_MIN_CONFIDENCE = 0.85


# ==============================================================================
# IMPLICIT ENTITY CREATION
# ==============================================================================
# Whether to auto-create a person profile when mentioned in goals/episodes
LONG_MEMORY_PEOPLE_IMPLICIT_CREATE_FROM_GOAL = False
LONG_MEMORY_PEOPLE_IMPLICIT_CREATE_FROM_EPISODE_PARTICIPANTS = True

# Whether to auto-create a place when mentioned in experience/episodes
LONG_MEMORY_PLACES_IMPLICIT_CREATE_FROM_EXPERIENCE = False
LONG_MEMORY_PLACES_IMPLICIT_CREATE_FROM_EPISODES = False


# ==============================================================================
# PERSISTENCE & STORAGE (JSON DATABASE)
# ==============================================================================
MEMORY_DIR = BASE_DIR / "memory"

# Long-term memory database file paths
PEOPLE_PATH = MEMORY_DIR / "people.json"
PLACES_PATH = MEMORY_DIR / "places.json"
FACTS_PATH = MEMORY_DIR / "facts.json"
GOALS_PATH = MEMORY_DIR / "goals.json"
EXPERIENCE_PATH = MEMORY_DIR / "experience.json"
EPISODES_PATH = MEMORY_DIR / "episodes.log.jsonl"
REFLECTIONS_PATH = MEMORY_DIR / "reflections.json"
PROCESSED_SESSIONS_PATH = MEMORY_DIR / "processed_sessions.json"


# ==============================================================================
# BACKUP & SAFETY
# ==============================================================================
# Whether to create .bak copies before every write operation
LONG_MEMORY_BACKUP_ENABLED = True

# Maximum number of backups stored per file
LONG_MEMORY_MAX_BACKUPS = 2
