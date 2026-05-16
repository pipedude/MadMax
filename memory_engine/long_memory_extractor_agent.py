import argparse
import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from memory_engine.active_context_builder import Session, parse_day_markdown
from google import genai
from google.genai import types

from memory_engine.entity_policies import (
    get_generic_person_reference_terms_prompt_text,
    is_placeholder_person_reference,
)
from memory_engine.llm_client_utils import (
    build_non_live_genai_client,
    generate_content_with_diagnostics_async,
)
from config import GOOGLE_API_KEY
from memory_engine.memory_config import (
    BASE_DIR,
    FACTS_PATH,
    LONG_MEMORY_EXTRACTOR_ENTITY_CONTEXT_LIMIT,
    LONG_MEMORY_GEMINI_MODEL,
    LONG_MEMORY_GEMINI_THINKING_LEVEL,
    LONG_MEMORY_MIN_CONFIDENCE_EXTRACT,
    PEOPLE_PATH,
    PLACES_PATH,
)
from memory_engine.long_memory_apply import LongMemoryApply, normalize_precise_datetime
from memory_engine.long_memory_ops import (
    OPERATION_SCHEMAS,
    ExtractionResult,
    normalize_operation_item,
    validate_operations_payload,
)
from memory_engine.time_policy import timestamp_policy_for_prompt

_GENERIC_PERSON_REFS_PROMPT = get_generic_person_reference_terms_prompt_text()

logger = logging.getLogger(__name__)


def get_people_places_context_for_extractor(limit: int = LONG_MEMORY_EXTRACTOR_ENTITY_CONTEXT_LIMIT) -> str:
    """Builds a text context of existing people and places for injection into the extractor prompt."""
    lines: list[str] = []
    total = 0

    if PEOPLE_PATH.exists() and total < limit:
        try:
            data = json.loads(PEOPLE_PATH.read_text(encoding="utf-8"))
            people = data.get("people", [])
            if isinstance(people, list) and people:
                lines.append("=== KNOWN PEOPLE ===")
                for person in people[:limit - total]:
                    name = person.get("name", "")
                    person_id = person.get("person_id", "")
                    aliases = person.get("aliases", [])
                    alias_str = f", aliases: {', '.join(aliases)}" if aliases else ""
                    lines.append(f"- {person_id}: {name}{alias_str}")
                total += len(people[:limit - total])
        except Exception:
            pass

    if PLACES_PATH.exists() and total < limit:
        try:
            data = json.loads(PLACES_PATH.read_text(encoding="utf-8"))
            places = data.get("places", [])
            if isinstance(places, list) and places:
                lines.append("=== KNOWN PLACES ===")
                for place in places[:limit - total]:
                    name = place.get("name", "")
                    place_id = place.get("place_id", "")
                    place_type = place.get("type", "")
                    aliases = place.get("aliases", [])
                    alias_str = f", aliases: {', '.join(aliases)}" if aliases else ""
                    type_str = f" [{place_type}]" if place_type else ""
                    lines.append(f"- {place_id}: {name}{type_str}{alias_str}")
                total += len(places[:limit - total])
        except Exception:
            pass

    return "\n".join(lines) if lines else ""


def _write_processed_sessions_registry(items: list[dict[str, str]]) -> None:
    from memory_engine.memory_config import PROCESSED_SESSIONS_PATH

    PROCESSED_SESSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_SESSIONS_PATH.write_text(
        json.dumps({"processed_sessions": items}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def get_processed_session_ids() -> set[str]:
    from memory_engine.memory_config import PROCESSED_SESSIONS_PATH

    processed_ids: set[str] = set()
    if not PROCESSED_SESSIONS_PATH.exists():
        logger.debug("Processed sessions file not found at %s, starting fresh", PROCESSED_SESSIONS_PATH)
        return processed_ids
    try:
        payload = json.loads(PROCESSED_SESSIONS_PATH.read_text(encoding="utf-8"))
        items = payload.get("processed_sessions", [])
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                session_id = item.get("session_id")
                if isinstance(session_id, str) and session_id.strip():
                    processed_ids.add(session_id.strip())
        return processed_ids
    except json.JSONDecodeError as exc:
        logger.error("CRITICAL: %s is corrupted: %s. All sessions will be re-processed.", PROCESSED_SESSIONS_PATH, exc)
        return processed_ids
    except Exception as exc:
        logger.error("Failed to load processed sessions from %s: %s", PROCESSED_SESSIONS_PATH, exc)
        return processed_ids


def mark_session_processed(session_id: str) -> None:
    from memory_engine.memory_config import PROCESSED_SESSIONS_PATH

    normalized_session_id = session_id.strip()
    if not normalized_session_id:
        return

    existing_items: list[dict[str, str]] = []
    known_session_ids: set[str] = set()
    if PROCESSED_SESSIONS_PATH.exists():
        try:
            payload = json.loads(PROCESSED_SESSIONS_PATH.read_text(encoding="utf-8"))
            stored_items = payload.get("processed_sessions", [])
            if isinstance(stored_items, list):
                for item in stored_items:
                    if not isinstance(item, dict):
                        continue
                    stored_session_id = item.get("session_id")
                    if not isinstance(stored_session_id, str) or not stored_session_id.strip():
                        continue
                    clean_session_id = stored_session_id.strip()
                    if clean_session_id in known_session_ids:
                        continue
                    known_session_ids.add(clean_session_id)
                    processed_at = item.get("processed_at")
                    if not isinstance(processed_at, str) or not processed_at.strip():
                        processed_at = datetime.now().astimezone().isoformat(timespec="seconds")
                    existing_items.append({
                        "session_id": clean_session_id,
                        "processed_at": processed_at,
                    })
        except Exception as e:
            logger.error("Error preparing processed_sessions.json entry: %s", e)

    if normalized_session_id in known_session_ids:
        return

    existing_items.append({
        "session_id": normalized_session_id,
        "processed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    })
    _write_processed_sessions_registry(existing_items)


async def process_missing_sessions(day_date: str) -> list[dict[str, Any]]:
    """
    Finds in the day file all sessions that have not yet been extracted into long-term memory,
    and runs the extraction process for them.
    """
    from memory_engine.memory_config import DAILY_LOGS_DIR, MARKDOWN_FILE_TEMPLATE
    
    day_file = DAILY_LOGS_DIR / MARKDOWN_FILE_TEMPLATE.format(date=day_date)
    if not day_file.exists():
        return []

    # 1. Get list of already processed sessions
    processed_ids = get_processed_session_ids()
    
    # 2. Parse all sessions from the day file
    day_markdown = day_file.read_text(encoding="utf-8")
    all_sessions = parse_day_markdown(day_markdown)
    
    reports = []
    extractor = LongMemoryExtractor()

    # 3. Process only missed sessions
    for session in all_sessions:
        if session.session_id not in processed_ids:
            logger.info("New session found for processing: %s", session.session_id)
            try:
                extractor_input = ExtractionInput.from_markdown_session(str(day_file), session.session_id)
                extraction_result = await extractor.extract_async(extractor_input)
                extraction_meta = extraction_result.get("_meta", {}) if isinstance(extraction_result, dict) else {}
                failed_categories = extraction_meta.get("category_failures", []) if isinstance(extraction_meta, dict) else []

                apply_engine = LongMemoryApply()
                report = await apply_engine.apply_payload(extraction_result, transcript=extractor_input.transcript)
                report["session_id"] = session.session_id
                report["extraction_meta"] = extraction_meta
                logger.info(
                    "Session processing result: session_id=%s raw_ops=%s filtered_ops=%s failed_categories=%s apply_counts=%s",
                    session.session_id,
                    extraction_meta.get("raw_operations_count", 0),
                    extraction_meta.get("filtered_operations_count", len(extraction_result.get("operations", [])) if isinstance(extraction_result, dict) else 0),
                    [item.get("category") for item in failed_categories if isinstance(item, dict)],
                    report.get("counts", {}),
                )
                if failed_categories:
                    logger.warning(
                        "Session %s will not be marked as processed: extraction completed with category errors=%s",
                        session.session_id,
                        [item.get("category") for item in failed_categories if isinstance(item, dict)],
                    )
                else:
                    mark_session_processed(session.session_id)
                    processed_ids.add(session.session_id)
                reports.append(report)
            except asyncio.CancelledError:
                logger.info("Session processing %s cancelled", session.session_id)
                raise
            except Exception as e:
                logger.error("Error processing session %s: %s", session.session_id, e)
    
    return reports


RAW_MODEL_OUTPUT_DEBUG_PATH = BASE_DIR / "tmp_extraction_raw_response.txt"
RAW_MODEL_OUTPUT_RETRY_DEBUG_PATH = BASE_DIR / "tmp_extraction_raw_response_retry.txt"
EXTRACTION_ERROR_LOG_PATH = BASE_DIR / "tmp_extraction_error.log"
FILTER_TRACE_DEBUG_PATH = BASE_DIR / "tmp_extraction_filter_trace.json"


@dataclass(frozen=True)
class ExtractionInput:
    transcript: str
    session_id: str
    start_time: str
    end_time: str

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ExtractionInput":
        if args.day_markdown_file is not None:
            if not args.session_id:
                raise ValueError("session_id is required when using --day-markdown-file")
            return cls.from_markdown_session(args.day_markdown_file, args.session_id)

        transcript = args.transcript
        if args.transcript_file is not None:
            transcript = Path(args.transcript_file).read_text(encoding="utf-8")
        if transcript is None:
            raise ValueError("Transcript is required")
        if not args.session_id:
            raise ValueError("session_id is required")
        if not args.start_time:
            raise ValueError("start_time is required")
        if not args.end_time:
            raise ValueError("end_time is required")
        return cls(
            transcript=transcript,
            session_id=args.session_id,
            start_time=normalize_precise_datetime(args.start_time) or args.start_time,
            end_time=normalize_precise_datetime(args.end_time) or args.end_time,
        )

    @classmethod
    def from_markdown_session(cls, day_markdown_file: str, session_id: str) -> "ExtractionInput":
        day_markdown = Path(day_markdown_file).read_text(encoding="utf-8")
        sessions = parse_day_markdown(day_markdown)
        session = find_session_by_id(sessions, session_id)
        if session is None:
            raise ValueError(f"Could not find session_id {session_id} in {day_markdown_file}")
        return cls(
            transcript=build_transcript_from_session(session),
            session_id=session.session_id,
            start_time=normalize_precise_datetime(session.started_at) or session.started_at,
            end_time=normalize_precise_datetime(session.ended_at) or session.ended_at,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtractionInput":
        return cls(
            transcript=str(data.get("transcript", "")),
            session_id=str(data.get("session_id", "")),
            start_time=normalize_precise_datetime(str(data.get("start_time", ""))) or str(data.get("start_time", "")),
            end_time=normalize_precise_datetime(str(data.get("end_time", ""))) or str(data.get("end_time", "")),
        )


@dataclass(frozen=True)
class CategoryExtractionResult:
    category: str
    operations: list[dict[str, Any]]
    had_error: bool = False
    error: str | None = None


def format_session_reply_line(reply: Any) -> str:
    speaker_label = reply.speaker
    if getattr(reply, "name", None):
        speaker_label = f"{speaker_label} ({reply.name})"
    return f"{speaker_label}: {reply.text}"


def build_transcript_from_session(session: Session) -> str:
    return "\n".join(format_session_reply_line(reply) for reply in session.replies)


def find_session_by_id(sessions: list[Session], session_id: str) -> Session | None:
    for session in sessions:
        if session.session_id == session_id:
            return session
    return None


class LongMemoryExtractor:
    def __init__(self) -> None:
        self._client: genai.Client | None = None

    def _get_client(self) -> genai.Client:
        if self._client is None:
            if not GOOGLE_API_KEY:
                raise RuntimeError("GOOGLE_API_KEY is not set in configs or .env")
            self._client = build_non_live_genai_client()
        return self._client

    def _build_base_prompt(self, extraction_input: ExtractionInput, category_instruction: str, existing_context: str = "") -> str:
        context_block = f"\n<existing_entities>\n{existing_context}\n</existing_entities>\n" if existing_context else ""
        return f"""
<role_instruction>
You are a Memory Extractor for a robot. Your task is to analyze a dialogue transcript and extract atomic, useful and reliable facts, goals and events for the robot's long-term memory.
This call is STRICTLY focused on the following category:
{category_instruction}
</role_instruction>{context_block}

<output_format>
Return the result STRICTLY in JSON format:
{{
  "operations": [ ... ]
}}
- Do not add markdown markup (no ```json).
- Do not add explanatory text before or after the JSON.
- If there is no useful data to extract, return {{"operations": []}}.
</output_format>

<general_rules>
- Do not invent information (names, facts, places) not present in the transcript.
- Do not return fields not provided by the schema.
- Service fields (operation names, enum values, keys) must be written STRICTLY in English.
- User content (names, descriptions, place names) keep in the original language (usually English).
- Preserve names and titles exactly as they appear in the transcript. Do not translate or transliterate them.
- PRINCIPLE: better to under-extract than to extract a doubtful or noisy fragment.
</general_rules>

<temporal_rules>
- NEVER save relative time ("tomorrow", "yesterday", "next week", "next Saturday").
{timestamp_policy_for_prompt()}
- Use the start_time field from the <input_data> section as the reference point ("today").
- Example: if start_time: 2026-04-26 (Sunday), and the text says "next Saturday" -> write "2026-05-02".
</temporal_rules>

<filtering_rules>
- Keep only: stable facts, practically useful information, real future plans, useful robot experience or significant episodes.
- IGNORE: jokes, emotional noise, household bustle, weak guesses.
- STRICTLY IGNORE one-off reference queries, translation of individual phrases, questions about word meanings, formulations like "can I / how / what does / what does / can I / posso" and similar one-time informational requests, unless they confirm a stable personal context of the user.
- STRICTLY IGNORE temporary household states (who went where, who is eating what, who is sitting/standing/lying where), unless it is related to safety.
- ATTENTION: Do not confuse "temporary details" with "important environmental changes".
- RECORD: objects and states that represent a risk (hazard), obstacles or important spatial changes.
- Do not turn into facts: hypotheses, medical guesses, probabilistic interpretations, current plans ("preparing a surprise"), secret intentions.
- If the transcript shows signs of uncertainty (maybe, seems, probably, I'm not sure), such facts must CATEGORICALLY NOT be extracted.
- Current plans, preparation of surprises, secret joint intentions or coordination of actions are always GOAL or EPISODE, not FACT.
- If a statement was made only in the agent's reply and not confirmed by the user's words, such a fact must NOT be extracted.
</filtering_rules>

<few_shot_examples>
- Example 1: transcript contains only the question "Posso andare in fabbrica a Marghera?" without additional personal context -> return {{"operations": []}}.
- Example 2: transcript contains "Today we want to watch Furiosa" -> acceptable add_goal_candidate about watching the movie.
- Example 3: transcript contains "A box with wires is lying in the garage near the door" -> place can be extracted only if the box or the area near the door is important as a landmark, obstacle or risk.
</few_shot_examples>

<confidence_rules>
Use STRICTLY the fixed scale:
- 0.95: Rare. A direct, unambiguous declaration of a fact.
- 0.85: Default. Fact is confirmed by context or explicit agreement.
- 0.75: Practically useful inference or cautious interpretation.
- In case of any doubts choose the lower value. If confidence is below {LONG_MEMORY_MIN_CONFIDENCE_EXTRACT} — do not extract at all. Never set 1.0.
</confidence_rules>

<input_data>
session_id: {extraction_input.session_id}
start_time: {extraction_input.start_time}
end_time: {extraction_input.end_time}

transcript:
{extraction_input.transcript}
</input_data>
"""

    def _get_category_instructions(self, category: str) -> str:
        instructions = {
            "entities": f"""ENTITIES (PEOPLE AND PLACES)

<schema_definition>
1. upsert_person_candidate: person_name (req), aliases, role, last_seen_at, source_session_id.
2. upsert_place_candidate: place_name (req), aliases, place_type, parent_place_name, last_confirmed_at, source_session_id.
   - Allowed place_type: room, area, object, furniture, hazard, obstacle.
</schema_definition>

<entity_specific_rules>
<rule_person>
If a person is mentioned and linked to a fact/goal/episode, create upsert_person_candidate.
- DO NOT CREATE duplicates. The <existing_entities> section lists already known people with their IDs and aliases.
- If the transcript mentions a name/alias that OBVIOUSLY refers to an already known person (e.g. "Johnny" for "John", or "Masha" for "Maria"), use upsert_person_candidate with the ORIGINAL person_name from the list, and record the new form in aliases.
- Example: the list has "John", the text says "Johnny went to the store" → upsert_person_candidate(person_name="John", aliases=["Johnny"]).
- Example: the list has "Mikky", the text says "Mikky was given a book" → upsert_person_candidate(person_name="Mikky", aliases=[]). Do not create a new "Mikky".
- If an alias exactly matches the name of another person — this is a new person.
- Do not fill role if it is not explicitly named.
- STRICTLY FORBIDDEN to extract a role if its name or direct synonym is absent from the transcript.
- Do not invent roles like "owner", "family member", "father", "mother" if they are not in the text.
- Create a person only if the person is explicitly identified by name or a stable proper designation.
- DO NOT create a person for generic nameless references: {_GENERIC_PERSON_REFS_PROMPT}.
- Exception: the aggregated label "User's Family" is allowed if the conversation is explicitly about the family as a group.
- If a participant is not identified by name, they may be mentioned in the episode summary, but a separate person entity must not be created.
</rule_person>

<rule_place>
- DO NOT CREATE duplicate places. The <existing_entities> section lists already known places with their IDs and aliases.
- If the transcript mentions a name that OBVIOUSLY refers to an already known place, use upsert_place_candidate with the ORIGINAL place_name from the list.
- A place is not just a room or zone, but also any significant stationary object, piece of furniture or risk area.
- Use hierarchy via parent_place_name.
  Example: place_name: "Mug", place_type: "object", parent_place_name: "Nightstand".
- MANDATORY: extract objects and zones if they are related to safety, risk, obstacles or serve as important landmarks.
- Allowed place_type: room, area, object, furniture, hazard, obstacle.
- STRICTLY FORBIDDEN to create records for generic room names (kitchen, garage, bathroom) if they are merely mentioned as background and are not described with new details or unique properties.
- Do not create a place for locations that appeared only in a single reference, translation or informational question from the user, if the transcript does not indicate that this is a real context of the user's life, home, route, project or environment.
- Do not create a place for a generic name without a stable link to context: for example "factory", "store", "cafe", "street", if this is only part of a one-off question and does not describe a significant known location.
- If you are in doubt whether a place is a real long-term context of the user, do not extract it.
- Negative example: "Posso andare in fabbrica a Marghera?" without additional life context → do not create a place at all.
- Positive example: "A box with wires is lying in the garage near the door" → create a place only if the box near the door is important as a landmark, obstacle or risk.
- STRICTLY FORBIDDEN to use placeholder names: unknown, n/a, somewhere, none. If a place is not identified concretely — skip the operation entirely.
</rule_place>
</entity_specific_rules>""",
            
            "facts": f"""FACTS

<schema_definition>
1. upsert_fact_candidate: subject_name (req), fact_type (req), description (req), confidence, source_session_id.
   - fact_type ENUM: preference, relation, profile, household, behavior, other.
</schema_definition>

<entity_specific_rules>
<rule_fact>
- One fact = one candidate. Do not mix different properties in a single description.
- Example: "Mikky is 8 years old" and "Mikky loves space" -> two different candidates.
- Stable occupations, hobbies, preferred regular activities and long-term interests are fact_type=profile, not household.
- Use household only for domestic, home, organizational or family-household circumstances, not for hobbies and interests.
- If you hesitate between profile and household, choose profile only when the description refers to a stable personal trait, interest, habit or regular activity of the person.
  - A fact must have one main anchor subject (person or entity).
  - STRICTLY FORBIDDEN to extract facts about subjects that do not participate in the dialogue and are not mentioned as key participants of events.
  - Do not use generic subjects like {_GENERIC_PERSON_REFS_PROMPT} for fact.
  - A fact must be an atomic thought, not a description of the whole scene or project.
  - Do not create a new fact just because it was formulated by the agent, retelling already known memory or a tool result.
  - STRICTLY IGNORE temporary actions: "drinking coffee", "sitting at the table", "looking out the window", "talking on the phone".
  - Do not create a fact if it is a paraphrase of existing memory without new information.
 - Do not create a separate fact just to repeat an alias, nickname or code name of a person, if it is better stored as aliases on the person.
 - behavior: only biological/psychological constants (allergy, phobia, chronic reactions). NOT communication styles, response formats, dialogue rules — that is experience.
 - preference: only static tastes and interests ("loves space", "doesn't like noise").
 - Positive example: "Mikky is into astronomy" -> upsert_fact_candidate with fact_type=profile.
 - Positive example: "John has a pollen allergy" -> upsert_fact_candidate with fact_type=behavior.
 - Negative example: "John prefers short answers during voice testing" -> this is experience (a rule for the robot), not fact.
 - Negative example: "Mikky has the code name Falcon" with a simultaneous person alias -> do not create a separate fact just for the alias.
 - Negative example: "Mikky is drinking coffee right now" -> do not create a fact.
</rule_fact>
</entity_specific_rules>""",

            "goals": """GOALS AND PLANS

<schema_definition>
1. add_goal_candidate: description (req), person_name, due_at, source_session_id, status.
   - status ENUM: active, inactive.
</schema_definition>

<entity_specific_rules>
<rule_goal>
Only future tasks, reminders or promises. If the action has ALREADY been done in the conversation — this is not a goal.
Fill person_name only if the task relates to a specific person.
- Do not create a goal from a one-off informational question, translation request or an attempt to formulate a phrase in another language.
- Questions like "can I...", "how...", "what does...", "how to say...", "what does...", "can I...", "posso..." are not goals by themselves, unless the user explicitly speaks about the intention to do it later as their real plan.
- Do not rephrase a reference question into an artificial goal like "find out the possibility...", if the transcript contains no explicit intention, promise, reminder or future plan.
- A goal is allowed only when the user directly expresses an intention, commitment, plan or a request for a reminder.
- Default status = active. Set status inactive ONLY if the user explicitly asked to cancel, close or abandon an existing task.
- Example: "Forget about watching Furiosa" or "Cancel the wings task" -> add_goal_candidate(..., status="inactive").
- Negative example: "Posso andare in fabbrica a Marghera?" -> do not create a goal if this is just a reference question.
- Positive example: "Today we want to watch Furiosa" -> add_goal_candidate(..., status="active").
</rule_goal>
</entity_specific_rules>""",

            "experience": """EXPERIENCE (BEHAVIOR RULES)

<schema_definition>
1. upsert_experience_candidate: action (req), object (req), place_name, reason, effect, status, confidence, source_session_id.
    - status ENUM: active, inactive.
</schema_definition>

<entity_specific_rules>
<rule_experience>
- Experience is a learned behavior rule of the robot, stemming from past interaction and applicable in the future.
- This is a rule FOR ITSELF (the robot), not knowledge ABOUT a person or the world. Knowledge about people is facts.
- Physical experience: collision avoidance, safe distances, navigation, reaction to physical triggers.
  Example: "if the cat flattened its ears -> back away 2 meters".
- Social/communication experience: how the robot should better behave in communication with a specific person or in a specific situation. Includes response style, dialogue formats, preferred length or tone of responses.
  Example: "if the user yawns -> offer tea".
  Example: "if Mikky is upset -> ask what happened, don't joke".
  Example: "if John asks for voice testing -> answer briefly".
- FORBIDDEN: simple knowledge about a person without a rule for the robot ("Mikky loves space" -> this is FACT, not experience).
- FORBIDDEN: reminders, tasks, promises ("remind Elisa about things" -> GOAL).
- FORBIDDEN: technical self-reports (calibration, motors, software updates), unless they yield a practical action rule.
- Default status = active. Set status inactive ONLY if the user explicitly asked to forget this rule.
</rule_experience>
</entity_specific_rules>""",

            "episodes": f"""EPISODES

<schema_definition>
1. add_episode_candidate: summary (req), participants, place_name, time, source_session_id.
</schema_definition>

<entity_specific_rules>
<rule_episode>
Add an episode ONLY if a significant event occurred with a concrete result, change or decision.
- Episode = not just "what happened", but "what changed" or "what decision was made".
- Good episode: "Mikky suggested using Raspberry Pi 5 for the robot, I agreed to help with 3D modeling" (result: a concrete decision).
- Good episode: "Elisa left for the pool, I stayed home alone to work" (result: change in participant composition).
- Bad episode: "Scooter ride" (no result — this is just a fact, store as fact, not episode).
- Bad episode: "Discussion of movies" (too general, no specifics — not an episode).
- Bad episode: "Ordered wings from Burger King" (no significant result — just an action).
- Summary must be concrete: who, what they did, what the result was. One sentence, not a list.
- participants: only external people/entities (WITHOUT the robot itself).
- Do not add unidentified placeholder labels like {_GENERIC_PERSON_REFS_PROMPT} without a name to participants.
- place_name: only if there is one clearly dominant place. If there are many places or a list — leave empty. Never write a comma-separated list of places.
- DO NOT duplicate: if an episode obviously describes the same event already in memory (the same walk with the same people), skip it.
</rule_episode>
</entity_specific_rules>""",

            "persona": """PERSONA AND REFLECTION

<schema_definition>
1. upsert_persona_candidate: trait, self_perception, source_session_id.
</schema_definition>

<entity_specific_rules>
<rule_persona_roles>
- trait: Record not just the role name, but "Role + Context". Briefly describe in connection with which events this role was adopted.
- Format: "Role name (brief explanation of reason or situation)".
- Example: "Garage mystery explorer (after 3D-scanning sectors for gift storage)" instead of just "Explorer".
</rule_persona_roles>

<rule_persona_self>
- self_perception: This is not a dry fact, but "Insight + Reflection". Write in first person. Describe the chain of reasoning that led to this conclusion.
- Format: "I came to the conclusion that... [insight], because [reason/observation]".
- Example: "I realize I am part of this family, because John entrusted me with a secret from Mikky, and I felt responsibility for the success of the common cause" instead of "I am part of the family".
- Extract this only if the agent in the transcript directly states their state, new role or awareness of their place in the world.
</rule_persona_self>
</entity_specific_rules>"""
        }
        return instructions.get(category, "")

    def _build_retry_prompt(
        self,
        extraction_input: ExtractionInput,
        category_instruction: str,
        invalid_response_text: str,
        validation_error: str,
        existing_context: str = "",
    ) -> str:
        return f"""
<instruction>
The previous JSON response failed technical validation. Fix it, keeping the format and structure STRICTLY according to the schema.
- Do not add new data not present in the transcript.
- Fix only the JSON structure, field names, operation types (op) or enum values.
- If an operation looks doubtful according to the original prompt rules, better DELETE it and return fewer operations than keep a doubtful object.
- Keep the semantic constraints of the original category: do not turn one-off reference questions into long-term memory just for the sake of valid JSON.
- Return STRICTLY clean JSON without markdown and explanations.
</instruction>

<error_details>
{validation_error}
</error_details>

<original_prompt>
{self._build_base_prompt(extraction_input, category_instruction, existing_context)}
</original_prompt>

<invalid_response>
{invalid_response_text}
</invalid_response>
"""

    async def _extract_category_async(self, extraction_input: ExtractionInput, category: str) -> list[dict[str, Any]]:
        category_instruction = self._get_category_instructions(category)
        existing_context = ""
        if category == "entities":
            existing_context = get_people_places_context_for_extractor()
        prompt = self._build_base_prompt(extraction_input, category_instruction, existing_context)
        
        async def call_llm(p: str, op_name: str) -> str:
            response = await generate_content_with_diagnostics_async(
                client=self._get_client(),
                model=LONG_MEMORY_GEMINI_MODEL,
                contents=p,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(
                        thinking_level=LONG_MEMORY_GEMINI_THINKING_LEVEL
                    ),
                ),
                logger=logger,
                operation_name=op_name,
            )
            return response.text.strip() if response.text else ""

        try:
            response_text = await call_llm(prompt, f"long_memory_extract_{category}")
        except Exception as exc:
            logger.error("LLM critical error in category %s: %s", category, exc)
            save_extraction_error_log(
                f"stage=call_llm category={category} session_id={extraction_input.session_id} error={exc}"
            )
            return CategoryExtractionResult(category=category, operations=[], had_error=True, error=str(exc))
            
        if not response_text:
            logger.error("Empty model response for category %s session_id=%s", category, extraction_input.session_id)
            save_extraction_error_log(
                f"stage=empty_response category={category} session_id={extraction_input.session_id}"
            )
            return CategoryExtractionResult(category=category, operations=[], had_error=True, error="empty_model_response")

        try:
            payload = self._parse_model_response(response_text)
            normalized_payload = normalize_and_clean_operations_payload(payload)
            validated = validate_operations_payload(normalized_payload)
            operations = validated.get("operations", [])
            logger.info(
                "Category extraction processed: session_id=%s category=%s operations=%s",
                extraction_input.session_id,
                category,
                len(operations),
            )
            return CategoryExtractionResult(category=category, operations=operations)
        except (json.JSONDecodeError, ValueError) as exc:
            save_raw_model_output(response_text)
            logger.warning("JSON validation error for category %s, trying retry...", category)
            save_extraction_error_log(
                f"stage=initial_validation category={category} session_id={extraction_input.session_id} error={exc}"
            )
            
            retry_prompt = self._build_retry_prompt(extraction_input, category_instruction, response_text, str(exc), existing_context)
            retry_response_text = await call_llm(retry_prompt, f"long_memory_extract_{category}_retry")
            
            if not retry_response_text:
                save_retry_raw_model_output(retry_response_text)
                logger.error("Empty retry model response for category %s session_id=%s", category, extraction_input.session_id)
                save_extraction_error_log(
                    f"stage=empty_retry_response category={category} session_id={extraction_input.session_id}"
                )
                return CategoryExtractionResult(category=category, operations=[], had_error=True, error="empty_retry_response")
            
            try:
                payload = self._parse_model_response(retry_response_text)
                normalized_payload = normalize_and_clean_operations_payload(payload)
                validated = validate_operations_payload(normalized_payload)
                operations = validated.get("operations", [])
                logger.info(
                    "Category extraction processed after retry: session_id=%s category=%s operations=%s",
                    extraction_input.session_id,
                    category,
                    len(operations),
                )
                return CategoryExtractionResult(category=category, operations=operations)
            except Exception as retry_exc:
                save_retry_raw_model_output(retry_response_text)
                logger.error("Retry also failed for category %s: %s", category, retry_exc)
                save_extraction_error_log(
                    f"stage=retry_validation category={category} session_id={extraction_input.session_id} error={retry_exc}"
                )
                return CategoryExtractionResult(category=category, operations=[], had_error=True, error=str(retry_exc))

    def _parse_model_response(self, text: str) -> dict[str, Any]:
        """Extracts JSON from the model response, stripping markdown markup."""
        text = text.strip()
        # Try to find JSON inside ```json ... ```
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
        
        # If no blocks found, try to parse the whole text (Gemini with response_mime_type often returns clean JSON)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Last attempt: find something resembling a JSON object { ... }
            match = re.search(r"(\{.*\})", text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            raise

    async def extract_async(self, extraction_input: ExtractionInput) -> dict[str, list[dict[str, Any]]]:
        if not extraction_input.transcript.strip():
            return {
                **ExtractionResult().to_dict(),
                "_meta": {
                    "session_id": extraction_input.session_id,
                    "category_reports": [],
                    "category_failures": [],
                    "raw_operations_count": 0,
                    "filtered_operations_count": 0,
                },
            }

        categories = ["entities", "facts", "goals", "experience", "episodes", "persona"]
        
        tasks = [self._extract_category_async(extraction_input, cat) for cat in categories]
        results = await asyncio.gather(*tasks)
        
        all_operations: list[dict[str, Any]] = []
        category_reports: list[dict[str, Any]] = []
        category_failures: list[dict[str, str]] = []
        for result in results:
            all_operations.extend(result.operations)
            category_reports.append({
                "category": result.category,
                "operations_count": len(result.operations),
                "had_error": result.had_error,
                "error": result.error,
            })
            if result.had_error:
                category_failures.append({
                    "category": result.category,
                    "error": result.error or "unknown_error",
                })

        validated_payload = {"operations": all_operations}
        filtered_payload = post_filter_operations_payload(validated_payload, extraction_input)
        filtered_operations = filtered_payload.get("operations", [])
        logger.info(
            "Extraction completed: session_id=%s raw_ops=%s filtered_ops=%s failed_categories=%s",
            extraction_input.session_id,
            len(all_operations),
            len(filtered_operations),
            [item["category"] for item in category_failures],
        )
        return {
            "operations": filtered_operations,
            "_meta": {
                "session_id": extraction_input.session_id,
                "category_reports": category_reports,
                "category_failures": category_failures,
                "raw_operations_count": len(all_operations),
                "filtered_operations_count": len(filtered_operations),
            },
        }

    def extract(self, extraction_input: ExtractionInput) -> dict[str, list[dict[str, Any]]]:
        # Synchronous wrapper for backward compatibility, if needed
        return asyncio.run(self.extract_async(extraction_input))

def save_raw_model_output(response_text: str) -> None:
    RAW_MODEL_OUTPUT_DEBUG_PATH.write_text(response_text, encoding="utf-8")


def save_retry_raw_model_output(response_text: str) -> None:
    RAW_MODEL_OUTPUT_RETRY_DEBUG_PATH.write_text(response_text, encoding="utf-8")


def save_extraction_error_log(message: str) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with EXTRACTION_ERROR_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def normalize_and_clean_operations_payload(payload: dict[str, Any] | list[Any]) -> dict[str, list[dict[str, Any]]]:
    operations = payload if isinstance(payload, list) else payload.get("operations")
    if not isinstance(operations, list):
        raise ValueError("Payload must contain an operations list")
    normalized_operations: list[dict[str, Any]] = []
    for item in operations:
        if not isinstance(item, dict):
            normalized_operations.append(item)
            continue
        normalized_item = normalize_operation_item(item)
        op = str(normalized_item.get("op", "")).strip()
        data = normalized_item.get("data", {})
        if not isinstance(data, dict):
            normalized_operations.append({"op": op, "data": data})
            continue
        schema = OPERATION_SCHEMAS.get(op)
        cleaned_data = dict(data)
        if schema is not None:
            for field_name in schema["optional"]:
                value = cleaned_data.get(field_name)
                if value is None:
                    cleaned_data.pop(field_name, None)
                    continue
                if isinstance(value, str) and not value.strip():
                    cleaned_data.pop(field_name, None)
        normalized_operations.append({"op": op, "data": cleaned_data})
    return {"operations": normalized_operations}


def post_filter_operations_payload(
    payload: dict[str, list[dict[str, Any]]],
    extraction_input: ExtractionInput,
) -> dict[str, list[dict[str, Any]]]:
    operations = payload.get("operations", [])
    trace_entries: list[dict[str, Any]] = []
    filter_steps: list[tuple[str, Any]] = [
        ("enrich_operations_with_session_context", lambda current: enrich_operations_with_session_context(current, extraction_input)),
        ("drop_unidentified_person_operations", drop_unidentified_person_operations),
        ("clean_episode_operations", clean_episode_operations),
        ("drop_generic_subject_facts", drop_generic_subject_facts),
        ("drop_exact_duplicate_existing_facts", drop_exact_duplicate_existing_facts),
        ("drop_low_confidence_operations", drop_low_confidence_operations),
    ]
    filtered_operations = operations
    for step_name, step in filter_steps:
        next_operations = step(filtered_operations)
        trace_entries.append(build_filter_trace_entry(step_name, filtered_operations, next_operations))
        filtered_operations = next_operations
    save_filter_trace(trace_entries)
    return validate_operations_payload({"operations": filtered_operations})


def save_filter_trace(trace_entries: list[dict[str, Any]]) -> None:
    FILTER_TRACE_DEBUG_PATH.write_text(json.dumps({"steps": trace_entries}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_filter_trace_entry(
    step_name: str,
    before_operations: list[dict[str, Any]],
    after_operations: list[dict[str, Any]],
) -> dict[str, Any]:
    removed_operations = operations_difference(before_operations, after_operations)
    added_operations = operations_difference(after_operations, before_operations)
    return {
        "step": step_name,
        "before_count": len(before_operations),
        "after_count": len(after_operations),
        "removed_count": len(removed_operations),
        "added_or_modified_count": len(added_operations),
        "removed_operations": removed_operations[:5],
        "added_or_modified_operations": added_operations[:5],
    }


def operations_difference(
    left_operations: list[dict[str, Any]],
    right_operations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    serialized_right_counts: dict[str, int] = {}
    for operation in right_operations:
        serialized = serialize_operation(operation)
        serialized_right_counts[serialized] = serialized_right_counts.get(serialized, 0) + 1
    difference: list[dict[str, Any]] = []
    for operation in left_operations:
        serialized = serialize_operation(operation)
        remaining = serialized_right_counts.get(serialized, 0)
        if remaining > 0:
            serialized_right_counts[serialized] = remaining - 1
            continue
        difference.append(operation)
    return difference


def serialize_operation(operation: dict[str, Any]) -> str:
    return json.dumps(operation, ensure_ascii=False, sort_keys=True)


def enrich_operations_with_session_context(
    operations: list[dict[str, Any]],
    extraction_input: ExtractionInput,
) -> list[dict[str, Any]]:
    enriched_operations: list[dict[str, Any]] = []
    for operation in operations:
        data = dict(operation.get("data", {}))
        # Inject source_session_id into all operations if the model did not return it
        if not is_non_empty_text(data.get("source_session_id")):
            data["source_session_id"] = extraction_input.session_id
        
        if operation.get("op") == "add_episode_candidate":
            normalized_time = normalize_precise_datetime(data.get("time"))
            if normalized_time is not None:
                data["time"] = normalized_time
            elif not is_non_empty_text(data.get("time")):
                data["time"] = extraction_input.start_time
        enriched_operations.append({"op": operation.get("op"), "data": data})
    return enriched_operations


def drop_unidentified_person_operations(operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered_operations: list[dict[str, Any]] = []
    for operation in operations:
        op = operation.get("op")
        data = dict(operation.get("data", {}))
        if op == "upsert_person_candidate":
            person_name = data.get("person_name")
            if isinstance(person_name, str) and is_allowed_person_entity_name(person_name):
                filtered_operations.append({"op": op, "data": data})
            continue
        if op == "add_goal_candidate":
            person_name = data.get("person_name")
            if isinstance(person_name, str) and not is_allowed_person_entity_name(person_name):
                data.pop("person_name", None)
            filtered_operations.append({"op": op, "data": data})
            continue
        if op == "add_episode_candidate":
            participants = data.get("participants")
            if isinstance(participants, list):
                cleaned_participants = [
                    participant
                    for participant in participants
                    if isinstance(participant, str)
                    and is_allowed_person_entity_name(participant)
                    and not is_agent_self_name(participant)
                ]
                cleaned_participants = dedupe_strings(cleaned_participants)
                if cleaned_participants:
                    data["participants"] = cleaned_participants
                else:
                    data.pop("participants", None)
            filtered_operations.append({"op": op, "data": data})
            continue
        filtered_operations.append(operation)
    return filtered_operations


def clean_episode_operations(operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered_operations: list[dict[str, Any]] = []
    for operation in operations:
        if operation.get("op") != "add_episode_candidate":
            filtered_operations.append(operation)
            continue
        data = dict(operation.get("data", {}))
        participants = data.get("participants")
        if isinstance(participants, list):
            cleaned_participants = [
                participant
                for participant in participants
                if isinstance(participant, str) and not is_agent_self_name(participant)
            ]
            cleaned_participants = dedupe_strings(cleaned_participants)
            if cleaned_participants:
                data["participants"] = cleaned_participants
            else:
                data.pop("participants", None)
        place_name = data.get("place_name")
        if isinstance(place_name, str):
            if is_placeholder_place_name(place_name):
                data.pop("place_name", None)
            elif is_multi_place_value(place_name):
                primary = extract_primary_place(place_name)
                if primary:
                    data["place_name"] = primary
                else:
                    data.pop("place_name", None)
        filtered_operations.append({"op": operation["op"], "data": data})
    return filtered_operations


def drop_generic_subject_facts(operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered_operations: list[dict[str, Any]] = []
    for operation in operations:
        if operation.get("op") != "upsert_fact_candidate":
            filtered_operations.append(operation)
            continue
        data = operation.get("data", {})
        subject_name = data.get("subject_name")
        if not isinstance(subject_name, str):
            filtered_operations.append(operation)
            continue
        if is_generic_fact_subject(subject_name):
            continue
        filtered_operations.append(operation)
    return filtered_operations


def drop_exact_duplicate_existing_facts(operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    existing_facts = load_existing_fact_records()
    filtered_operations: list[dict[str, Any]] = []
    for operation in operations:
        if operation.get("op") != "upsert_fact_candidate":
            filtered_operations.append(operation)
            continue
        data = operation.get("data", {})
        subject_name = data.get("subject_name")
        fact_type = data.get("fact_type")
        description = data.get("description")
        if not isinstance(subject_name, str) or not isinstance(fact_type, str) or not isinstance(description, str):
            filtered_operations.append(operation)
            continue
        if is_exact_duplicate_existing_fact(subject_name, fact_type, description, existing_facts):
            continue
        filtered_operations.append(operation)
    return filtered_operations


def drop_low_confidence_operations(operations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    filtered_operations: list[dict[str, Any]] = []
    for operation in operations:
        data = operation.get("data", {})
        confidence = data.get("confidence")
        if confidence is not None:
            try:
                if float(confidence) < LONG_MEMORY_MIN_CONFIDENCE_EXTRACT:
                    continue
            except (ValueError, TypeError):
                pass
        filtered_operations.append(operation)
    return filtered_operations


def normalize_text_value(value: str) -> str:
    return " ".join(value.strip().split()).casefold()


def load_existing_fact_records() -> list[dict[str, Any]]:
    if not FACTS_PATH.exists():
        return []
    try:
        data = json.loads(FACTS_PATH.read_text(encoding="utf-8"))
        return data.get("facts", [])
    except Exception:
        return []


def is_exact_duplicate_existing_fact(subject: str, fact_type: str, description: str, existing_facts: list[dict[str, Any]]) -> bool:
    s_key = normalize_text_value(subject)
    t_key = normalize_text_value(fact_type)
    d_key = normalize_text_value(description)
    for fact in existing_facts:
        if normalize_text_value(str(fact.get("subject", ""))) == s_key and \
           normalize_text_value(str(fact.get("fact_type", ""))) == t_key and \
           normalize_text_value(str(fact.get("description", ""))) == d_key:
            return True
    return False


def is_non_empty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def dedupe_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = normalize_text_value(value)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(" ".join(value.strip().split()))
    return result


def is_agent_self_name(value: str) -> bool:
    normalized_value = normalize_text_value(value)
    return normalized_value in {
        "max",
        "madmax",
        "robot",
        "assistant",
        "agent",
    }


def is_allowed_person_entity_name(value: str) -> bool:
    normalized_value = normalize_text_value(value)
    if normalized_value == "user's family":
        return True
    if is_placeholder_person_reference(value):
        return False
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        return False
    if any(char.isdigit() for char in cleaned):
        return False
    return True


def is_multi_place_value(value: str) -> bool:
    normalized_value = normalize_text_value(value)
    if "," in normalized_value or ";" in normalized_value or "/" in normalized_value:
        return True
    return bool(re.search(r"\band\b", normalized_value))


def extract_primary_place(value: str) -> str:
    """Extracts the first place from a list (e.g. 'Kitchen and hall' -> 'Kitchen')"""
    # First split by hard separators
    parts = re.split(r"[,;/]", value)
    first_part = parts[0].strip()
    # Then split by the conjunction 'and' as a separate word
    sub_parts = re.split(r"\s+and\s+", first_part)
    return sub_parts[0].strip()


def is_placeholder_place_name(value: str) -> bool:
    normalized_value = normalize_text_value(value)
    return normalized_value in {"unknown", "n/a", "none"}


def is_generic_fact_subject(subject_name: str) -> bool:
    normalized_subject = normalize_text_value(subject_name)
    if not normalized_subject:
        return True
    if normalized_subject == "user's family":
        return False
    if is_placeholder_person_reference(subject_name):
        return True
    return False

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id")
    parser.add_argument("--start-time")
    parser.add_argument("--end-time")
    transcript_group = parser.add_mutually_exclusive_group(required=True)
    transcript_group.add_argument("--transcript")
    transcript_group.add_argument("--transcript-file")
    transcript_group.add_argument("--day-markdown-file")
    return parser


async def main_async() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    extraction_input = ExtractionInput.from_args(args)
    result = await LongMemoryExtractor().extract_async(extraction_input)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main_async())
