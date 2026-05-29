import json
import re
import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memory_engine.memory_config import (
    DAILY_LOGS_DIR,
    MARKDOWN_FILE_TEMPLATE,
    OUTPUT_JSON_PATH,
    RECENT_REPLIES_LIMIT,
    REFLECTIONS_PATH,
    GOALS_PATH,
    EXPERIENCE_PATH,
    EPISODES_PATH,
    PLACES_PATH,
    ACTIVE_CONTEXT_PERSONA_ROLES_LIMIT,
    ACTIVE_CONTEXT_PERSONA_INSIGHTS_LIMIT,
    ACTIVE_CONTEXT_EPISODES_LIMIT,
    ACTIVE_CONTEXT_EXPERIENCE_LIMIT,
    ACTIVE_CONTEXT_EXPERIENCE_MIN_CONFIDENCE,
)
from memory_engine.summarize_context_agent import ContextAgent

logger = logging.getLogger(__name__)


DATE_HEADER_RE = re.compile(r"^#\s+(\d{4}-\d{2}-\d{2})$")
SESSION_HEADER_RE = re.compile(r"^##\s+Session\s+(.+)$")
SESSION_ID_RE = re.compile(r"^session_id:\s*(.+)$")
STARTED_AT_RE = re.compile(r"^started_at:\s*(.+)$")
ENDED_AT_RE = re.compile(r"^ended_at:\s*(.+)$")
REPLY_RE = re.compile(r"^(user|agent)(\s*\([^)]*\))?:\s*(.+)$")


@dataclass
class Reply:
    speaker: str
    name: str | None
    text: str

    def to_dict(self) -> dict[str, str | None]:
        data: dict[str, str | None] = {
            "speaker": self.speaker,
            "text": self.text,
        }
        if self.name is not None:
            data["name"] = self.name
        return data


@dataclass
class Session:
    session_id: str
    started_at: str
    ended_at: str
    replies: list[Reply]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "dialog": [reply.to_dict() for reply in self.replies],
            "ended_at": self.ended_at,
        }


def get_long_term_injections(context_date: str) -> dict[str, Any]:
    injections = {
        "persona": {"roles": [], "insights": []},
        "agenda": [],
        "experience": [],
        "episodes": []
    }

    # 1. Persona (Reflections)
    if REFLECTIONS_PATH.exists():
        try:
            with open(REFLECTIONS_PATH, "r", encoding="utf-8") as f:
                ref = json.load(f)
                # Take last 3 roles
                roles = ref.get("current_roles", [])
                injections["persona"]["roles"] = roles[-ACTIVE_CONTEXT_PERSONA_ROLES_LIMIT:] if isinstance(roles, list) else []
                # Take last 3 insights
                insights = ref.get("self_perception", [])
                injections["persona"]["insights"] = insights[-ACTIVE_CONTEXT_PERSONA_INSIGHTS_LIMIT:] if isinstance(insights, list) else []
        except json.JSONDecodeError as exc:
            logger.error("CRITICAL: %s is corrupted: %s. Persona will be EMPTY.", REFLECTIONS_PATH, exc)
        except Exception as exc:
            logger.error("Failed to load long-term persona injections from %s: %s", REFLECTIONS_PATH, exc)
    else:
        logger.debug("Reflections file not found at %s, skipping persona injection", REFLECTIONS_PATH)

    # 2. Agenda (Goals)
    if GOALS_PATH.exists():
        try:
            with open(GOALS_PATH, "r", encoding="utf-8") as f:
                goals_data = json.load(f)
                goals = goals_data.get("goals", [])
                if isinstance(goals, list):
                    for g in goals:
                        if g.get("status") != "active":
                            continue
                        due_at = g.get("due_at")
                        # Condition: today OR no deadline
                        if not due_at or due_at == context_date:
                            injections["agenda"].append({
                                "description": g.get("description"),
                                "due_at": due_at
                            })
        except json.JSONDecodeError as exc:
            logger.error("CRITICAL: %s is corrupted: %s. Agenda will be EMPTY.", GOALS_PATH, exc)
        except Exception as exc:
            logger.error("Failed to load long-term agenda injections from %s: %s", GOALS_PATH, exc)
    else:
        logger.debug("Goals file not found at %s, skipping agenda injection", GOALS_PATH)

    # 3. Experience
    if EXPERIENCE_PATH.exists():
        try:
            place_lookup: dict[str, str] = {}
            if PLACES_PATH.exists():
                with open(PLACES_PATH, "r", encoding="utf-8") as places_file:
                    places_data = json.load(places_file)
                    places = places_data.get("places", [])
                    if isinstance(places, list):
                        for place in places:
                            if not isinstance(place, dict):
                                continue
                            place_id = place.get("place_id")
                            place_name = place.get("name")
                            if isinstance(place_id, str) and place_id.strip() and isinstance(place_name, str) and place_name.strip():
                                place_lookup[place_id.strip()] = place_name.strip()
            with open(EXPERIENCE_PATH, "r", encoding="utf-8") as f:
                exp_data = json.load(f)
                experience = exp_data.get("experience", [])
                if isinstance(experience, list):
                    # Take last N active records with confidence >= configured threshold
                    candidates = []
                    for e in experience:
                        if e.get("status") != "active":
                            continue
                        confidence = e.get("confidence", 0)
                        if isinstance(confidence, (int, float)) and confidence >= ACTIVE_CONTEXT_EXPERIENCE_MIN_CONFIDENCE:
                            place_name = e.get("place_name")
                            if not isinstance(place_name, str) or not place_name.strip():
                                place_id = e.get("place_id")
                                if isinstance(place_id, str) and place_id.strip():
                                    place_name = place_lookup.get(place_id.strip())
                            candidates.append({
                                "action": e.get("action"),
                                "object": e.get("object"),
                                "place_name": place_name,
                                "reason": e.get("reason"),
                                "effect": e.get("effect"),
                                "confidence": confidence,
                            })
                    injections["experience"] = candidates[-ACTIVE_CONTEXT_EXPERIENCE_LIMIT:]
        except json.JSONDecodeError as exc:
            logger.error("CRITICAL: %s is corrupted: %s. Experience memory will be EMPTY.", EXPERIENCE_PATH, exc)
        except Exception as exc:
            logger.error("Failed to load long-term experience injections from %s: %s", EXPERIENCE_PATH, exc)
    else:
        logger.debug("Experience file not found at %s, skipping experience injection", EXPERIENCE_PATH)

    # 4. Episodes (Key Events)
    if EPISODES_PATH.exists():
        try:
            episodes_raw: list[dict[str, Any]] = []
            with open(EPISODES_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        item = json.loads(stripped)
                        episodes_raw.append(item)
                    except json.JSONDecodeError:
                        continue
            # Take last 3 by time
            episodes_raw.sort(key=lambda e: e.get("time") or "", reverse=True)
            injections["episodes"] = episodes_raw[:ACTIVE_CONTEXT_EPISODES_LIMIT]
        except Exception as exc:
            logger.error("Failed to load episodes from %s: %s", EPISODES_PATH, exc)
    else:
        logger.debug("Episodes file not found at %s, skipping episodes injection", EPISODES_PATH)

    return injections


def parse_day_markdown(day_markdown: str) -> list[Session]:
    sessions: list[Session] = []
    lines = day_markdown.splitlines()
    current_session_id = ""
    current_started_at = ""
    current_ended_at = ""
    current_replies: list[Reply] = []
    inside_session = False

    def flush_session() -> None:
        nonlocal current_session_id, current_started_at, current_ended_at, current_replies
        if current_session_id:
            sessions.append(
                Session(
                    session_id=current_session_id,
                    started_at=current_started_at,
                    ended_at=current_ended_at,
                    replies=current_replies,
                )
            )
        current_session_id = ""
        current_started_at = ""
        current_ended_at = ""
        current_replies = []

    for line in lines:
        header_match = SESSION_HEADER_RE.match(line)
        if header_match:
            continue

        if line == "--- SESSION START ---":
            inside_session = True
            continue

        if line == "--- SESSION END ---":
            inside_session = False
            continue

        if not inside_session and current_session_id:
            ended_at_match = ENDED_AT_RE.match(line)
            if ended_at_match:
                current_ended_at = ended_at_match.group(1).strip()
                flush_session()
            continue

        if not inside_session:
            continue

        if not current_session_id:
            session_id_match = SESSION_ID_RE.match(line)
            if session_id_match:
                current_session_id = session_id_match.group(1).strip()
                continue

        if not current_started_at:
            started_at_match = STARTED_AT_RE.match(line)
            if started_at_match:
                current_started_at = started_at_match.group(1).strip()
                continue

        reply_match = REPLY_RE.match(line)
        if reply_match and current_session_id:
            speaker = reply_match.group(1).strip()
            raw_name = reply_match.group(2)
            name = raw_name.strip()[1:-1].strip() if raw_name else None
            text = reply_match.group(3).strip()
            current_replies.append(
                Reply(
                    speaker=speaker,
                    name=name,
                    text=text,
                )
            )

    if current_session_id:
        flush_session()

    return sessions


def extract_day_date(day_markdown: str) -> str:
    for line in day_markdown.splitlines():
        date_match = DATE_HEADER_RE.match(line.strip())
        if date_match:
            return date_match.group(1)
    raise ValueError("Could not extract day date from markdown header")


def build_last_context(sessions: list[Session], limit: int) -> list[dict[str, Any]]:
    if not sessions:
        return []

    selected_session_indexes: set[int] = set()
    collected = 0

    for index in range(len(sessions) - 1, -1, -1):
        session = sessions[index]
        reply_count = len(session.replies)
        if reply_count == 0:
            continue
        selected_session_indexes.add(index)
        collected += reply_count
        if collected >= limit:
            break

    ordered_sessions: list[dict[str, Any]] = []
    for index, session in enumerate(sessions):
        if index in selected_session_indexes:
            ordered_sessions.append(session.to_dict())
    return ordered_sessions


def count_replies(sessions: list[Session]) -> int:
    return sum(len(session.replies) for session in sessions)


def load_existing_active_context() -> dict[str, Any]:
    if not OUTPUT_JSON_PATH.exists():
        return {}
    return json.loads(OUTPUT_JSON_PATH.read_text(encoding="utf-8"))


def _pretty_list_of_dicts(items: list[Any], field_order: list[str]) -> list[str]:
    """Formats a list of dicts into readable lines like:
       - key1: val1 | key2: val2
    Unknown fields are appended after known ones."""
    out: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            out.append(f"  - {item}")
            continue
        parts = []
        for key in field_order:
            val = item.get(key)
            if val is not None and val != "":
                parts.append(f"{key}: {val}")
        for key, val in item.items():
            if key not in field_order and val is not None and val != "":
                parts.append(f"{key}: {val}")
        if parts:
            out.append("  - " + " | ".join(parts))
    return out


def build_compact_injection_context(context_data: dict[str, Any]) -> str:
    """Builds a compact representation of the active context for injection into the system prompt.
    Includes only meaningful fields, excluding service metadata."""
    lines: list[str] = []

    date = context_data.get("context_date", "")
    time = context_data.get("current_time", "")
    if date:
        lines.append(f"Date: {date}")
    if time:
        lines.append(f"Current Time: {time}")
    if date or time:
        lines.append("")

    summary_yesterday = context_data.get("summary_yesterday", "")
    if summary_yesterday:
        lines.append("--- Yesterday's Summary ---")
        lines.append(str(summary_yesterday))
        lines.append("")

    summary_today = context_data.get("summary_today", "")
    if summary_today:
        lines.append("--- Today's Summary ---")
        lines.append(str(summary_today))
        lines.append("")

    reply_count_today = context_data.get("reply_count_today")
    if reply_count_today is not None:
        lines.append(f"Reply Count Today: {reply_count_today}")
    summary_reply_count = context_data.get("summary_reply_count")
    if summary_reply_count is not None:
        lines.append(f"Summary Reply Count: {summary_reply_count}")
    if reply_count_today is not None or summary_reply_count is not None:
        lines.append("")

    last_context = context_data.get("last_context", [])
    if isinstance(last_context, list) and last_context:
        lines.append("--- Recent Context ---")
        for session in last_context:
            if not isinstance(session, dict):
                continue
            session_id = session.get("session_id", "unknown")
            started_at = session.get("started_at", "")
            ended_at = session.get("ended_at", "")
            time_range = f"{started_at} → {ended_at}" if ended_at else started_at
            lines.append(f"Session {session_id} ({time_range}):")
            dialog = session.get("dialog", [])
            if isinstance(dialog, list):
                for msg in dialog:
                    if isinstance(msg, dict):
                        speaker = msg.get("speaker", "?")
                        text = msg.get("text", "")
                        lines.append(f"  {speaker}: {text}")
            lines.append("")

    injections = context_data.get("long_term_injections", {})
    if isinstance(injections, dict) and injections:
        lines.append("--- Long Term Memory ---")
        persona = injections.get("persona", {})
        if isinstance(persona, dict):
            roles = persona.get("roles", [])
            if isinstance(roles, list) and roles:
                lines.append("Persona Roles:")
                for role in roles:
                    if isinstance(role, dict):
                        trait = role.get("trait", "")
                        if trait:
                            lines.append(f"  - {trait}")
            insights = persona.get("insights", [])
            if isinstance(insights, list) and insights:
                lines.append("Persona Insights:")
                for insight in insights:
                    if isinstance(insight, dict):
                        text = insight.get("insight", "")
                        if text:
                            lines.append(f"  - {text}")

        agenda = injections.get("agenda", [])
        if isinstance(agenda, list) and agenda:
            lines.append("Agenda:")
            lines.extend(_pretty_list_of_dicts(agenda, ["description", "due_at"]))
            lines.append("")

        experience = injections.get("experience", [])
        if isinstance(experience, list) and experience:
            lines.append("Experience:")
            lines.extend(_pretty_list_of_dicts(
                experience,
                ["action", "object", "place_name", "reason", "effect", "confidence"]
            ))
            lines.append("")

        episodes = injections.get("episodes", [])
        if isinstance(episodes, list) and episodes:
            lines.append("Key Episodes:")
            for ep in episodes:
                if isinstance(ep, dict):
                    summary = ep.get("summary", "")
                    time = ep.get("time", "")
                    if summary:
                        prefix = f"[{time}] " if time else ""
                        lines.append(f"  - {prefix}{summary}")
            lines.append("")

        known_keys = {"persona", "agenda", "experience", "episodes"}
        for key, value in injections.items():
            if key in known_keys:
                continue
            if isinstance(value, list) and value:
                lines.append(f"{key.replace('_', ' ').title()}:")
                for item in value:
                    if isinstance(item, dict):
                        parts = [f"{k}: {v}" for k, v in item.items()
                                 if v is not None and v != ""]
                        if parts:
                            lines.append("  - " + " | ".join(parts))
                    else:
                        lines.append(f"  - {item}")
                lines.append("")
            elif isinstance(value, dict) and value:
                lines.append(f"{key.replace('_', ' ').title()}:")
                for k, v in value.items():
                    if v is not None and v != "":
                        lines.append(f"  {k}: {v}")
                lines.append("")

    return "\n".join(lines)


def get_local_today_date() -> str:
    return datetime.now().astimezone().date().isoformat()


def get_local_current_time() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def get_day_markdown_path(day_date: str) -> Path:
    return DAILY_LOGS_DIR / MARKDOWN_FILE_TEMPLATE.format(date=day_date)


def get_latest_previous_markdown_path(reference_date: str) -> Path | None:
    previous_candidates: list[tuple[str, Path]] = []
    for path in DAILY_LOGS_DIR.glob("*.md"):
        match = re.match(r"^(\d{4}-\d{2}-\d{2})\.md$", path.name)
        if not match:
            continue
        file_date = match.group(1)
        if file_date < reference_date:
            previous_candidates.append((file_date, path))

    if not previous_candidates:
        return None

    previous_candidates.sort(key=lambda item: item[0], reverse=True)
    return previous_candidates[0][1]


async def finalize_reference_day_summary_async(
    reference_markdown_path: Path,
    existing_active_context: dict[str, Any],
    context_agent: ContextAgent,
) -> str:
    reference_day_markdown = reference_markdown_path.read_text(encoding="utf-8")
    reference_date = extract_day_date(reference_day_markdown)
    active_context_date = existing_active_context.get("context_date")

    if active_context_date != reference_date:
        return await context_agent.summarize_day_async(reference_day_markdown)

    previous_summary = existing_active_context.get("summary_today", "")
    previous_reply_count = existing_active_context.get("reply_count_today", 0)
    previous_summary_reply_count = existing_active_context.get("summary_reply_count", 0)

    if previous_summary and previous_reply_count <= previous_summary_reply_count:
        return previous_summary

    return await context_agent.summarize_day_async(reference_day_markdown)


async def build_active_context_async(
    day_markdown: str,
    existing_active_context: dict[str, Any],
    source_markdown_path: Path,
    context_date: str,
    current_time: str,
    summary_yesterday: str,
) -> dict[str, Any]:
    if not day_markdown.strip():
        return {
            "context_date": context_date,
            "current_time": current_time,
            "source_markdown_path": str(source_markdown_path),
            "last_context": [],
            "summary_yesterday": summary_yesterday,
            "reply_count_today": 0,
            "summary_today": "",
            "summary_reply_count": 0,
            "long_term_injections": get_long_term_injections(context_date),
        }

    sessions = parse_day_markdown(day_markdown)
    last_context = build_last_context(sessions, RECENT_REPLIES_LIMIT)
    reply_count_today = count_replies(sessions)
    
    previous_summary = existing_active_context.get("summary_today", "")
    previous_summary_reply_count = existing_active_context.get("summary_reply_count", 0)

    if not previous_summary:
        should_refresh_summary = reply_count_today >= RECENT_REPLIES_LIMIT
    else:
        new_replies_since_summary = reply_count_today - previous_summary_reply_count
        should_refresh_summary = new_replies_since_summary >= RECENT_REPLIES_LIMIT

    if should_refresh_summary:
        context_agent = ContextAgent()
        summary_today = await context_agent.summarize_day_async(day_markdown)
        summary_reply_count = reply_count_today
    else:
        summary_today = previous_summary
        summary_reply_count = previous_summary_reply_count if previous_summary else 0

    return {
        "context_date": context_date,
        "current_time": current_time,
        "source_markdown_path": str(source_markdown_path),
        "last_context": last_context,
        "summary_yesterday": summary_yesterday,
        "reply_count_today": reply_count_today,
        "summary_today": summary_today,
        "summary_reply_count": summary_reply_count,
        "long_term_injections": get_long_term_injections(context_date),
    }


async def main_async() -> None:
    local_today_date = get_local_today_date()
    local_current_time = get_local_current_time()
    source_markdown_path = get_day_markdown_path(local_today_date)
    if source_markdown_path.exists():
        day_markdown = source_markdown_path.read_text(encoding="utf-8")
        markdown_context_date = extract_day_date(day_markdown)
        if markdown_context_date != local_today_date:
            raise ValueError(
                f"Markdown date {markdown_context_date} does not match local device date {local_today_date}"
            )
    else:
        day_markdown = ""

    existing_active_context = load_existing_active_context()
    previous_context_date = existing_active_context.get("context_date")
    is_new_day = previous_context_date != local_today_date
    previous_markdown_path = get_latest_previous_markdown_path(local_today_date)

    context_agent = ContextAgent()
    if is_new_day:
        if previous_markdown_path is not None:
            summary_yesterday = await finalize_reference_day_summary_async(
                previous_markdown_path,
                existing_active_context,
                context_agent,
            )
        else:
            summary_yesterday = ""
        existing_active_context = {
            "summary_today": "",
            "summary_reply_count": 0,
        }
    else:
        summary_yesterday = existing_active_context.get("summary_yesterday", "")
        if not summary_yesterday and previous_markdown_path is not None:
            summary_yesterday = await finalize_reference_day_summary_async(
                previous_markdown_path,
                existing_active_context,
                context_agent,
            )

    active_context = await build_active_context_async(
        day_markdown,
        existing_active_context,
        source_markdown_path,
        local_today_date,
        local_current_time,
        summary_yesterday,
    )
    await asyncio.to_thread(
        OUTPUT_JSON_PATH.write_text,
        json.dumps(active_context, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved active context to %s", OUTPUT_JSON_PATH)
    logger.info("last_context sessions: %d", len(active_context['last_context']))
    logger.info("reply_count_today: %d", active_context['reply_count_today'])
    logger.info("summary_reply_count: %d", active_context['summary_reply_count'])


if __name__ == "__main__":
    import asyncio
    asyncio.run(main_async())
