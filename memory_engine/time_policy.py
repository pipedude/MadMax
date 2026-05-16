from __future__ import annotations

"""Central timestamp policy for long-term memory.

Precise timestamps represent an exact moment in local device time and must use
ISO 8601 datetime with the current OS timezone offset.

Deadline-style fields represent a calendar date without time-of-day and must use
ISO 8601 date-only format.
"""

PRECISE_TIMESTAMP_FORMAT = "YYYY-MM-DDTHH:MM:SS±HH:MM"
DATE_ONLY_FORMAT = "YYYY-MM-DD"

PRECISE_TIMESTAMP_FIELDS: tuple[str, ...] = (
    "episode.time",
    "updated_at",
    "created_at",
    "last_seen_at",
    "last_confirmed_at",
    "started_at",
    "ended_at",
    "processed_at",
)

DATE_ONLY_FIELDS: tuple[str, ...] = (
    "due_at",
)


def precise_timestamp_fields_for_prompt() -> str:
    return ", ".join(("time", "last_seen_at", "last_confirmed_at"))


def timestamp_policy_for_prompt() -> str:
    return (
        f"- For the due_at field, keep the exact date in {DATE_ONLY_FORMAT} format.\n"
        f"- For precise-time fields ({precise_timestamp_fields_for_prompt()}), keep the local timestamp in {PRECISE_TIMESTAMP_FORMAT} format.\n"
        "- Do not use a bare date for episode.time, last_seen_at, or last_confirmed_at."
    )
