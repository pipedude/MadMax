from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class TranscriptEntry:
    role: str
    text: str


class SessionTranscriptLogger:
    def __init__(self, base_directory: str | Path) -> None:
        self.base_directory = Path(base_directory)
        if not self.base_directory.is_absolute():
            self.base_directory = (_PROJECT_ROOT / self.base_directory).resolve(strict=False)
        self._session_started_at: datetime | None = None
        self._session_id: str | None = None
        self._entries: list[TranscriptEntry] = []

    @property
    def current_session_id(self) -> str | None:
        return self._session_id

    def start_session(self, started_at: datetime) -> None:
        self._session_started_at = self._to_utc(started_at)
        self._session_id = f"sess_{uuid4().hex[:12]}"
        self._entries = []

    def log_user_message(self, text: str) -> None:
        self._append_entry("user", text)

    def log_agent_message(self, text: str) -> None:
        self._append_entry("agent", text)

    def flush_session(self) -> None:
        self._flush_sync()

    async def flush_session_async(self) -> None:
        await asyncio.to_thread(self._flush_sync)

    def _flush_sync(self) -> None:
        if self._session_started_at is None or self._session_id is None:
            self.reset_session()
            return

        has_user_entries = any(
            entry.role == "user" and entry.text.strip()
            for entry in self._entries
        )
        if not has_user_entries:
            self.reset_session()
            return

        target_directory = self.base_directory
        target_directory.mkdir(parents=True, exist_ok=True)

        ended_at = datetime.now(timezone.utc)
        duration_seconds = max(0, int((ended_at - self._session_started_at).total_seconds()))
        file_date = self._session_started_at.astimezone().strftime("%Y-%m-%d")
        session_display_time = self._session_started_at.astimezone().strftime("%H:%M:%S")
        target_file_path = target_directory / f"{file_date}.md"
        day_header = f"# {file_date}"
        session_lines = [
            f"## Session {session_display_time}",
            "",
            "--- SESSION START ---",
            f"session_id: {self._session_id}",
            f"started_at: {self._format_local(self._session_started_at)}",
            "",
        ]
        session_lines.extend(
            f"{entry.role}: {entry.text}"
            for entry in self._entries
        )
        if self._entries:
            session_lines.append("")
        session_lines.extend(
            [
                "--- SESSION END ---",
                f"session_id: {self._session_id}",
                f"ended_at: {self._format_local(ended_at)}",
                f"duration: {duration_seconds}s",
            ]
        )
        session_block = "\n".join(session_lines).rstrip()

        file_exists = target_file_path.exists()
        file_has_content = file_exists and target_file_path.stat().st_size > 0

        with target_file_path.open("a", encoding="utf-8") as transcript_file:
            if not file_has_content:
                transcript_file.write(f"{day_header}\n\n{session_block}\n")
            else:
                transcript_file.write(f"\n{session_block}\n")

        self.reset_session()
        return

    def reset_session(self) -> None:
        self._session_started_at = None
        self._session_id = None
        self._entries = []

    def _append_entry(self, role: str, text: str) -> None:
        normalized_text = text.strip()
        if self._session_started_at is None or not normalized_text:
            return
        self._entries.append(TranscriptEntry(role=role, text=normalized_text))

    def _to_utc(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.astimezone(timezone.utc)
        return value.astimezone(timezone.utc)

    def _format_local(self, value: datetime) -> str:
        # Format as local time without redundant timezones
        return value.astimezone().strftime("%Y-%m-%dT%H:%M:%S")
