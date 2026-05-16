import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from google.genai import types

from memory_engine.entity_policies import is_placeholder_person_reference
from memory_engine.llm_client_utils import build_non_live_genai_client, generate_content_with_diagnostics_async
from memory_engine.long_memory_normalize import LongMemoryNormalizer, clean_optional_text, clean_text, normalize_text, unique_strings
from memory_engine.long_memory_ops import validate_operations_payload
from memory_engine.memory_config import (
    EPISODES_PATH,
    EXPERIENCE_PATH,
    FACTS_PATH,
    GOALS_PATH,
    LONG_MEMORY_BACKUP_ENABLED,
    LONG_MEMORY_MAX_BACKUPS,
    LONG_MEMORY_MAX_EXPERIENCE_RECORDS,
    LONG_MEMORY_MAX_PERSONA_HISTORY,
    LONG_MEMORY_MAX_PERSONA_ROLES,
    LONG_MEMORY_PEOPLE_IMPLICIT_CREATE_FROM_EPISODE_PARTICIPANTS,
    LONG_MEMORY_PEOPLE_IMPLICIT_CREATE_FROM_GOAL,
    LONG_MEMORY_PLACES_IMPLICIT_CREATE_FROM_EPISODES,
    LONG_MEMORY_PLACES_IMPLICIT_CREATE_FROM_EXPERIENCE,
    LONG_MEMORY_SURGEON_BATCH_SIZE,
    LONG_MEMORY_SURGEON_GEMINI_MODEL,
    LONG_MEMORY_SURGEON_GEMINI_THINKING_LEVEL,
    PEOPLE_PATH,
    PLACES_PATH,
    REFLECTIONS_PATH,
)

logger = logging.getLogger(__name__)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


ApplyStatus = Literal["created", "updated", "unchanged", "skipped", "pending", "appended"]


@dataclass
class ApplyAction:
    op: str
    status: ApplyStatus
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {
            "op": self.op,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass
class ApplyReport:
    actions: list[ApplyAction] = field(default_factory=list)

    def add(self, op: str, status: ApplyStatus, detail: str) -> None:
        self.actions.append(ApplyAction(op=op, status=status, detail=detail))

    def to_dict(self) -> dict[str, Any]:
        counts = {
            "created": 0,
            "updated": 0,
            "appended": 0,
            "skipped": 0,
            "unchanged": 0,
            "pending": 0,
        }
        for action in self.actions:
            counts[action.status] = counts.get(action.status, 0) + 1
        return {
            "applied_at": now_iso(),
            "counts": counts,
            "actions": [action.to_dict() for action in self.actions],
        }


SURGEON_SYSTEM_PROMPT = """You are a robot memory conflict resolution module.
Your task is to compare old data (old_data) and new data (new_data) and decide how to merge them.

For each item in conflict_batch choose one action:
- "UPDATE": new data refute or replace old data (e.g. "likes" changed to "doesn't like").
- "MERGE": new data complement old data (e.g. reason is clarified or details are added).
- "APPEND": data have completely different meanings (different objects or different action types), both objects must be kept separately.
- "IGNORE": new data carry the same or lesser meaning than old data.

STRICT RULES:
1. TYPE CORRECTION: If you see that fact_type in old_data is less precise than in new data (e.g. was household, became profile), you MUST suggest the correct type in merged_fields.fact_type.
2. ENTITY MERGING: If in one batch you see conflicts about "John" and "Johnny", and understand from context that this is the same person — treat them as one entity.
3. CONTRADICTION RESOLUTION: If new data directly contradict old data (e.g. "smokes" vs "quit smoking"), always choose UPDATE and fill merged_fields with final values.
4. DUPLICATE MINIMIZATION: If data can be merged without loss of meaning — merge them (MERGE).
5. TRANSCRIPT USAGE: If a <transcript> block is present in the prompt, use it as the primary context for resolving the conflict.
6. CURRENT CONTEXT PRIORITY: In case of direct contradiction between old_data and new_data, rely on wording and temporal markers from the current session transcript.
7. DO NOT IGNORE CONTEXT: Consider temporal/context cues from the transcript (e.g. "before", "now", "no longer", "today", "yesterday"), even if old_data and new_data superficially look similar.
8. RELATION SYNONYMS: For fact_type=relation or similar family facts treat synonymous roles as THE SAME role: "spouse" = "wife/husband", "father" = "dad", "mother" = "mom", "son" = "child" (if context is unambiguous). If old_data and new_data describe the same relation with synonymous wording — choose MERGE or IGNORE, not APPEND.

If UPDATE or MERGE is chosen, return merged_fields with final fields by category:
- fact: description is required, fact_type is optional.
- goal: description is required, status is optional.
- experience: action and object are required, reason/effect/status are optional.
Do not return service fields like confidence, place_id, source_session_id, updated_at, created_at in merged_fields.
Return the result in JSON format matching the schema.
"""


def normalize_precise_datetime(value: Any) -> str | None:
    cleaned_value = clean_optional_text(value)
    if cleaned_value is None:
        return None
    local_timezone = datetime.now().astimezone().tzinfo
    try:
        parsed_datetime = datetime.fromisoformat(cleaned_value)
    except ValueError:
        try:
            parsed_date = date.fromisoformat(cleaned_value)
        except ValueError:
            return None
        return datetime(
            parsed_date.year,
            parsed_date.month,
            parsed_date.day,
            tzinfo=local_timezone,
        ).isoformat(timespec="seconds")
    if parsed_datetime.tzinfo is None:
        parsed_datetime = parsed_datetime.replace(tzinfo=local_timezone)
    else:
        parsed_datetime = parsed_datetime.astimezone(local_timezone)
    return parsed_datetime.isoformat(timespec="seconds")


def normalize_date_only(value: Any) -> str | None:
    cleaned_value = clean_optional_text(value)
    if cleaned_value is None:
        return None
    try:
        return date.fromisoformat(cleaned_value).isoformat()
    except ValueError:
        normalized_datetime = normalize_precise_datetime(cleaned_value)
        if normalized_datetime is None:
            return None
        return datetime.fromisoformat(normalized_datetime).date().isoformat()


def parse_iso_datetime(value: str | None) -> datetime | None:
    normalized_value = normalize_precise_datetime(value)
    if normalized_value is None:
        return None
    return datetime.fromisoformat(normalized_value)


def normalize_confidence(value: Any) -> float:
    if value is None:
        return 0.85
    if not isinstance(value, (int, float)):
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.85
    return float(value)


class LongMemoryApply:
    def __init__(self) -> None:
        self.client = None
        self._consumed = False
        self.people_payload = self._load_collection(PEOPLE_PATH, "people")
        self.places_payload = self._load_collection(PLACES_PATH, "places")
        self.facts_payload = self._load_collection(FACTS_PATH, "facts")
        self.goals_payload = self._load_collection(GOALS_PATH, "goals")
        self.experience_payload = self._load_collection(EXPERIENCE_PATH, "experience")
        self.reflections = self._load_json(REFLECTIONS_PATH)
        self.normalizer = LongMemoryNormalizer()
        self.people: list[dict[str, Any]] = self.people_payload["people"]
        self.places: list[dict[str, Any]] = self.places_payload["places"]
        self.facts: list[dict[str, Any]] = self.facts_payload["facts"]
        self.goals: list[dict[str, Any]] = self.goals_payload["goals"]
        self.experience: list[dict[str, Any]] = self.experience_payload["experience"]
        self.existing_episode_keys = self._load_existing_episode_keys()
        self.pending_episode_entries: list[dict[str, Any]] = []
        self.conflict_queue: list[dict[str, Any]] = []

    def _get_client(self):
        if self.client is None:
            self.client = build_non_live_genai_client()
        return self.client

    async def apply_payload(self, payload: dict[str, Any] | list[Any], transcript: str | None = None) -> dict[str, Any]:
        if self._consumed:
            raise RuntimeError("LongMemoryApply instance is single-use. Create a new one for each payload.")
        validated_payload = validate_operations_payload(payload)
        report = ApplyReport()
        backup_paths: list[tuple[Path, Path]] = []

        try:
            # Step 1: First pass (Sorting and Deferral)
            for operation in validated_payload["operations"]:
                op = operation["op"]
                data = operation["data"]
                if op == "upsert_person_candidate":
                    status, detail = self._apply_person(data)
                elif op == "upsert_place_candidate":
                    status, detail = self._apply_place(data)
                elif op == "upsert_fact_candidate":
                    status, detail = self._apply_fact(data)
                elif op == "add_goal_candidate":
                    status, detail = self._apply_goal(data)
                elif op == "upsert_experience_candidate":
                    status, detail = self._apply_experience(data)
                elif op == "add_episode_candidate":
                    status, detail = self._apply_episode(data)
                elif op == "upsert_persona_candidate":
                    status, detail = self._apply_persona(data)
                else:
                    status, detail = "skipped", f"Unsupported operation: {op}"
                report.add(op, status, detail)

            # Step 2: Second pass (Conflict resolution via LLM Surgeon)
            if self.conflict_queue:
                await self._resolve_batch_conflicts_async(report, transcript=transcript)

            if LONG_MEMORY_BACKUP_ENABLED:
                self._backup_files()
                for path in [PEOPLE_PATH, PLACES_PATH, FACTS_PATH, GOALS_PATH, EXPERIENCE_PATH, REFLECTIONS_PATH]:
                    bak = path.with_suffix(path.suffix + ".bak")
                    if bak.exists():
                        backup_paths.append((path, bak))
            await self._save_collection_async(PEOPLE_PATH, self.people_payload)
            await self._save_collection_async(PLACES_PATH, self.places_payload)
            await self._save_collection_async(FACTS_PATH, self.facts_payload)
            await self._save_collection_async(GOALS_PATH, self.goals_payload)
            await self._save_collection_async(EXPERIENCE_PATH, self.experience_payload)
            await self._save_json_async(REFLECTIONS_PATH, self.reflections)
            await self._append_episodes_async()
        except Exception:
            if backup_paths:
                logger.exception("apply_payload failed; rolling back memory files from .bak")
                for original_path, bak_path in backup_paths:
                    try:
                        shutil.copy2(bak_path, original_path)
                        logger.info("Restored %s from %s", original_path.name, bak_path.name)
                    except Exception as restore_exc:
                        logger.error("Failed to restore %s from %s: %s", original_path.name, bak_path.name, restore_exc)
            else:
                logger.exception("apply_payload failed")
            raise
        finally:
            self._consumed = True
            self.pending_episode_entries = []
            self.conflict_queue = []

        report_dict = report.to_dict()
        logger.info("Apply payload finished: counts=%s", report_dict.get("counts", {}))
        return report_dict

    async def _resolve_batch_conflicts_async(self, report: ApplyReport, transcript: str | None = None) -> None:
        if not self.conflict_queue:
            return

        # Smart batching: group by subject (person) so the Surgeon sees context for one entity
        groups: dict[str, list[dict[str, Any]]] = {}
        for conflict in self.conflict_queue:
            if conflict.get("category") == "experience":
                old_data = conflict.get("old_data", {})
                new_data = conflict.get("new_data", {})
                place_id = clean_optional_text(new_data.get("place_id")) or clean_optional_text(old_data.get("place_id"))
                if place_id:
                    subj_id = f"experience_place:{place_id}"
                else:
                    action = clean_optional_text(new_data.get("action")) or clean_optional_text(old_data.get("action"))
                    obj = clean_optional_text(new_data.get("object")) or clean_optional_text(old_data.get("object"))
                    action_key = normalize_text(action or "")
                    object_key = normalize_text(obj or "")
                    if action_key or object_key:
                        subj_id = f"experience:{action_key}|{object_key}"
                    else:
                        subj_id = "common"
            else:
                # Try to find subject_id (person_id or name)
                subj_id = conflict.get("old_data", {}).get("person_id") or \
                          conflict.get("old_data", {}).get("fact_id") or \
                          conflict.get("old_data", {}).get("goal_id") or "common"
            groups.setdefault(subj_id, []).append(conflict)

        # Build chunks from groups
        chunks: list[list[dict[str, Any]]] = []
        current_chunk: list[dict[str, Any]] = []
        max_batch = LONG_MEMORY_SURGEON_BATCH_SIZE

        for group_items in groups.values():
            if len(current_chunk) + len(group_items) > max_batch and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
            
            # If one group exceeds max_batch — split it (edge case)
            if len(group_items) > max_batch:
                for i in range(0, len(group_items), max_batch):
                    chunks.append(group_items[i:i + max_batch])
            else:
                current_chunk.extend(group_items)
        
        if current_chunk:
            chunks.append(current_chunk)

        # Run all batches in parallel
        tasks = [self._process_chunk_async(chunk, report, transcript=transcript) for chunk in chunks]
        await asyncio.gather(*tasks)

        # Clear the queue after processing
        self.conflict_queue = []

    async def _process_chunk_async(self, chunk: list[dict[str, Any]], report: ApplyReport, transcript: str | None = None) -> None:
        resolutions = await self._generate_resolutions_batch_async(chunk, transcript=transcript)
        self._apply_resolutions(resolutions, chunk, report)

    async def _generate_resolutions_batch_async(self, chunk: list[dict[str, Any]], transcript: str | None = None) -> list[dict[str, Any]]:
        response_schema = {
            "type": "OBJECT",
            "properties": {
                "resolutions": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "task_id": {"type": "STRING"},
                            "action": {
                                "type": "STRING",
                                "enum": ["UPDATE", "MERGE", "APPEND", "IGNORE"]
                            },
                            "merged_fields": {
                                "type": "OBJECT",
                                "properties": {
                                    "description": {"type": "STRING"},
                                    "fact_type": {"type": "STRING"},
                                    "status": {"type": "STRING"},
                                    "action": {"type": "STRING"},
                                    "object": {"type": "STRING"},
                                    "reason": {"type": "STRING"},
                                    "effect": {"type": "STRING"}
                                }
                            }
                        },
                        "required": ["task_id", "action"]
                    }
                }
            },
            "required": ["resolutions"]
        }

        transcript_block = ""
        if transcript:
            transcript_block = f"<transcript>\n{transcript}\n</transcript>\n\n"

        prompt = (
            f"{transcript_block}"
            f"Conflicts to resolve:\n{json.dumps(chunk, ensure_ascii=False, indent=2)}"
        )

        try:
            config_params = {
                "system_instruction": SURGEON_SYSTEM_PROMPT,
                "response_mime_type": "application/json",
                "response_schema": response_schema,
                "temperature": 0.0,
            }

            if LONG_MEMORY_SURGEON_GEMINI_THINKING_LEVEL:
                config_params["thinking_config"] = {
                    "include_thoughts": True,
                    "thinking_level": LONG_MEMORY_SURGEON_GEMINI_THINKING_LEVEL
                }

            response = await generate_content_with_diagnostics_async(
                client=self._get_client(),
                model=LONG_MEMORY_SURGEON_GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(**config_params),
                logger=logger,
                operation_name="long_memory_surgeon",
            )
            
            if not response.text:
                return []
            
            result = json.loads(response.text)
            return result.get("resolutions", [])
        except Exception as exc:
            logger.error("Error in LLM-Surgeon: %s", exc)
            raise RuntimeError(f"LLM Surgeon failed: {exc}") from exc

    def _apply_resolutions(self, resolutions: list[dict[str, Any]], chunk: list[dict[str, Any]], report: ApplyReport) -> None:
        # Build a map for quick lookup of original conflict data
        chunk_map = {c["task_id"]: c for c in chunk}
        valid_actions = {"IGNORE", "APPEND", "UPDATE", "MERGE"}
        
        for res in resolutions:
            task_id = str(res.get("task_id", "")).strip()
            action = str(res.get("action", "")).strip().upper()
            merged_fields = res.get("merged_fields")

            if action not in valid_actions:
                logger.warning("Surgeon returned invalid action '%s' for task_id=%s", action, task_id)
                report.add("resolve_unknown", "skipped", f"Invalid surgeon action '{action}' for task_id={task_id}")
                continue
            
            conflict = chunk_map.get(task_id)
            if not conflict:
                logger.warning("Surgeon returned unknown task_id: %s", task_id)
                report.add("resolve_unknown", "skipped", f"Unknown surgeon task_id '{task_id}'")
                continue
            
            category = conflict["category"]
            old_data = conflict["old_data"]
            new_data = conflict["new_data"]
            
            if action == "IGNORE":
                self._update_timestamp_only(category, old_data, new_data)
                report.add(f"resolve_{category}", "unchanged", f"LLM ignored conflict for {task_id}")
            
            elif action == "APPEND":
                status, detail = self._create_new_record_from_conflict(category, conflict)
                report.add(f"resolve_{category}", status, detail)
            
            elif action in ["UPDATE", "MERGE"]:
                if not isinstance(merged_fields, dict):
                    logger.warning("Surgeon returned %s without valid merged_fields for task_id=%s, using fallback", action, task_id)
                    merged_fields = {}

                self._update_record_text(category, old_data, new_data, merged_fields)
                report.add(f"resolve_{category}", "updated", f"LLM {action} {category} for {task_id}")

            else:
                logger.warning("Surgeon reached unexpected action branch '%s' for task_id=%s", action, task_id)
                report.add(f"resolve_{category}", "skipped", f"Unexpected surgeon action '{action}' for {task_id}")

    def _update_timestamp_only(self, category: str, old_data: dict[str, Any], new_data: dict[str, Any] | None = None) -> None:
        collection = self._get_collection_by_category(category)
        id_key = self._get_id_key_by_category(category)
        target_id = old_data.get(id_key)
        
        for item in collection:
            if item.get(id_key) == target_id:
                if new_data is not None and new_data.get("confidence") is not None:
                    item["confidence"] = max(
                        normalize_confidence(item.get("confidence")),
                        normalize_confidence(new_data.get("confidence")),
                    )
                item["updated_at"] = now_iso()
                break

    def _create_new_record_from_conflict(self, category: str, conflict: dict[str, Any]) -> tuple[ApplyStatus, str]:
        new_data = conflict["new_data"]
        old_data = conflict["old_data"]
        if category == "fact":
            existing = next((f for f in self.facts if f["fact_id"] == old_data["fact_id"]), None)
            if existing:
                # Search for a duplicate across the ENTIRE collection (including old_target) to avoid creating duplicates
                decision = self.normalizer.find_fact_match(
                    self.facts,
                    existing["subject"],
                    new_data.get("fact_type") or existing["fact_type"],
                    new_data["description"],
                    existing.get("person_id"),
                )
                if decision.status == "exact" and decision.matched is not None:
                    self._update_record_text(
                        "fact",
                        decision.matched,
                        new_data,
                        {},
                    )
                    return "updated", f"LLM APPEND deduped into existing fact '{new_data['description']}'"
                if decision.status == "conflict":
                    logger.warning("Skipping surgeon APPEND for fact due to duplicate conflict: %s", new_data.get("description"))
                    return "skipped", f"Skipped APPEND for fact due to duplicate conflict: '{new_data.get('description')}'"
                fact = {
                    "fact_id": generate_id("fact"),
                    "subject": existing["subject"],
                    "fact_type": new_data.get("fact_type") or existing["fact_type"],
                    "description": new_data["description"],
                    "confidence": new_data["confidence"],
                    "updated_at": now_iso(),
                }
                if existing.get("person_id"):
                    fact["person_id"] = existing["person_id"]
                if new_data.get("source_session_id"):
                    fact["source_session_id"] = new_data["source_session_id"]
                self.facts.append(fact)
                return "created", f"LLM appended new fact '{new_data['description']}'"
            return "skipped", "Could not append fact: original conflict target not found"
        
        elif category == "goal":
            existing = next((g for g in self.goals if g["goal_id"] == old_data["goal_id"]), None)
            if existing:
                decision = self.normalizer.find_goal_match(
                    self.goals,
                    new_data["description"],
                    new_data.get("person_id"),
                    new_data.get("status") or "active",
                )
                if decision.status == "exact" and decision.matched is not None:
                    self._update_record_text("goal", decision.matched, new_data, {})
                    return "updated", f"LLM APPEND deduped into existing goal '{new_data['description']}'"
                if decision.status == "conflict":
                    logger.warning("Skipping surgeon APPEND for goal due to duplicate conflict: %s", new_data.get("description"))
                    return "skipped", f"Skipped APPEND for goal due to duplicate conflict: '{new_data.get('description')}'"
                goal = {
                    "goal_id": generate_id("goal"),
                    "description": new_data["description"],
                    "status": new_data["status"],
                    "person_id": new_data.get("person_id"),
                    "due_at": new_data.get("due_at"),
                    "created_at": now_iso(),
                    "updated_at": now_iso(),
                }
                if new_data.get("source_session_id"):
                    goal["source_session_id"] = new_data["source_session_id"]
                self.goals.append(goal)
                return "created", f"LLM appended new goal '{new_data['description']}'"
            return "skipped", "Could not append goal: original conflict target not found"
        
        elif category == "experience":
            existing = next((e for e in self.experience if e["exp_id"] == old_data["exp_id"]), None)
            if existing:
                decision = self.normalizer.find_experience_match(
                    self.experience,
                    new_data["action"],
                    new_data["object"],
                    new_data.get("place_id"),
                )
                if decision.status == "exact" and decision.matched is not None:
                    self._update_record_text("experience", decision.matched, new_data, {})
                    return "updated", f"LLM APPEND deduped into existing experience '{new_data['action']}' / '{new_data['object']}'"
                if decision.status == "conflict":
                    logger.warning(
                        "Skipping surgeon APPEND for experience due to duplicate conflict: %s / %s",
                        new_data.get("action"),
                        new_data.get("object"),
                    )
                    return "skipped", f"Skipped APPEND for experience due to duplicate conflict: '{new_data.get('action')}' / '{new_data.get('object')}'"
                exp = {
                    "exp_id": generate_id("exp"),
                    "action": new_data["action"],
                    "object": new_data["object"],
                    "status": new_data.get("status", "active"),
                    "confidence": new_data.get("confidence", 0.85),
                    "created_at": now_iso(),
                    "updated_at": now_iso(),
                }
                if new_data.get("place_id") is not None:
                    exp["place_id"] = new_data["place_id"]
                if new_data.get("reason") is not None:
                    exp["reason"] = new_data["reason"]
                if new_data.get("effect") is not None:
                    exp["effect"] = new_data["effect"]
                if new_data.get("source_session_id"):
                    exp["source_session_id"] = new_data["source_session_id"]
                self.experience.append(exp)
                if len(self.experience) > LONG_MEMORY_MAX_EXPERIENCE_RECORDS:
                    self.experience = self.experience[-LONG_MEMORY_MAX_EXPERIENCE_RECORDS:]
                return "created", f"LLM appended new experience '{new_data['action']}' / '{new_data['object']}'"
            return "skipped", "Could not append experience: original conflict target not found"

        return "skipped", f"Unsupported conflict category for APPEND: {category}"

    def _update_record_text(self, category: str, old_data: dict[str, Any], new_data: dict[str, Any], merged_fields: dict[str, Any]) -> None:
        collection = self._get_collection_by_category(category)
        id_key = self._get_id_key_by_category(category)
        target_id = old_data.get(id_key)
        
        for item in collection:
            if item.get(id_key) == target_id:
                if category == "fact":
                    description = merged_fields.get("description")
                    if description is None:
                        description = new_data.get("description")
                    if description is None:
                        description = old_data.get("description")
                    if description is not None:
                        item["description"] = description
                    fact_type = merged_fields.get("fact_type")
                    if fact_type is None:
                        fact_type = new_data.get("fact_type")
                    if fact_type is None:
                        fact_type = old_data.get("fact_type")
                    if fact_type is not None:
                        item["fact_type"] = fact_type
                    if new_data.get("confidence") is not None:
                        item["confidence"] = max(
                            normalize_confidence(item.get("confidence")),
                            normalize_confidence(new_data.get("confidence")),
                        )
                elif category == "goal":
                    description = merged_fields.get("description")
                    if description is None:
                        description = new_data.get("description")
                    if description is None:
                        description = old_data.get("description")
                    if description is not None:
                        item["description"] = description
                    status = merged_fields.get("status")
                    if status is None:
                        status = new_data.get("status")
                    if status is None:
                        status = old_data.get("status")
                    if status is not None:
                        item["status"] = status
                    if new_data.get("due_at") is not None:
                        item["due_at"] = new_data["due_at"]
                
                elif category == "experience":
                    action = merged_fields.get("action")
                    if action is None:
                        action = new_data.get("action")
                    if action is None:
                        action = old_data.get("action")
                    if action is not None:
                        item["action"] = action
                    obj = merged_fields.get("object")
                    if obj is None:
                        obj = new_data.get("object")
                    if obj is None:
                        obj = old_data.get("object")
                    if obj is not None:
                        item["object"] = obj
                    place_id = clean_optional_text(new_data.get("place_id")) or clean_optional_text(old_data.get("place_id"))
                    if place_id is not None:
                        item["place_id"] = place_id
                    reason = merged_fields.get("reason")
                    if reason is None:
                        reason = new_data.get("reason")
                    if reason is None:
                        reason = old_data.get("reason")
                    if reason is not None:
                        item["reason"] = reason
                    effect = merged_fields.get("effect")
                    if effect is None:
                        effect = new_data.get("effect")
                    if effect is None:
                        effect = old_data.get("effect")
                    if effect is not None:
                        item["effect"] = effect
                    status = merged_fields.get("status")
                    if status is None:
                        status = new_data.get("status")
                    if status is None:
                        status = old_data.get("status")
                    if status is not None:
                        item["status"] = status
                    if new_data.get("confidence") is not None:
                        item["confidence"] = max(
                            normalize_confidence(item.get("confidence")),
                            normalize_confidence(new_data.get("confidence")),
                        )
                
                # Update source_session_id to the latest
                if new_data.get("source_session_id"):
                    item["source_session_id"] = new_data["source_session_id"]
                
                item["updated_at"] = now_iso()
                break

    def _get_collection_by_category(self, category: str) -> list[dict[str, Any]]:
        if category == "fact": return self.facts
        if category == "goal": return self.goals
        if category == "experience": return self.experience
        return []

    def _get_id_key_by_category(self, category: str) -> str:
        if category == "fact": return "fact_id"
        if category == "goal": return "goal_id"
        if category == "experience": return "exp_id"
        return "id"

    def _apply_person(self, data: dict[str, Any]) -> tuple[ApplyStatus, str]:
        person_name = clean_text(data["person_name"])
        existing = self._find_person_by_name_or_alias(person_name)
        aliases = unique_strings(data.get("aliases", []))
        role = clean_optional_text(data.get("role"))
        last_seen_at = normalize_precise_datetime(data.get("last_seen_at"))
        source_session_id = clean_optional_text(data.get("source_session_id"))

        if existing is None:
            person = {
                "person_id": generate_id("person"),
                "name": person_name,
                "aliases": aliases,
            }
            if role is not None:
                person["role"] = role
            if last_seen_at is not None:
                person["last_seen_at"] = last_seen_at
            if source_session_id is not None:
                person["source_session_id"] = source_session_id
            self.people.append(person)
            return "created", f"Created person '{person_name}'"

        changed = False
        changed = self._merge_aliases(existing, aliases + self._candidate_alias_from_primary_name(existing.get("name"), person_name)) or changed
        if role is not None and not clean_optional_text(existing.get("role")):
            existing["role"] = role
            changed = True
        if last_seen_at is not None and self._update_latest_timestamp(existing, "last_seen_at", last_seen_at):
            changed = True
        
        # Update the session of the last mention if it arrived
        if source_session_id is not None:
            existing["source_session_id"] = source_session_id
            # We do not count this as 'changed' for the updated/unchanged status to avoid spam,
            # but we will update the data. Although logically this is also a change.
            changed = True

        if changed:
            return "updated", f"Updated person '{existing['name']}'"
        return "unchanged", f"No changes for person '{existing['name']}'"

    def _apply_place(self, data: dict[str, Any]) -> tuple[ApplyStatus, str]:
        place_name = clean_text(data["place_name"])
        existing = self._find_place_by_name_or_alias(place_name)
        aliases = unique_strings(data.get("aliases", []))
        place_type = clean_optional_text(data.get("place_type"))
        parent_place_name = clean_optional_text(data.get("parent_place_name"))
        last_confirmed_at = normalize_precise_datetime(data.get("last_confirmed_at"))
        source_session_id = clean_optional_text(data.get("source_session_id"))

        if existing is None:
            place = {
                "place_id": generate_id("place"),
                "name": place_name,
                "aliases": aliases,
            }
            if place_type is not None:
                place["type"] = place_type
            if parent_place_name is not None:
                parent_id, _ = self._resolve_or_create_place_id(parent_place_name, allow_implicit=True)
                if parent_id is not None:
                    place["parent"] = parent_id
            if last_confirmed_at is not None:
                place["last_confirmed_at"] = last_confirmed_at
            if source_session_id is not None:
                place["source_session_id"] = source_session_id
            self.places.append(place)
            return "created", f"Created place '{place_name}'"

        changed = False
        changed = self._merge_aliases(existing, aliases + self._candidate_alias_from_primary_name(existing.get("name"), place_name)) or changed
        if place_type is not None and not clean_optional_text(existing.get("type")):
            existing["type"] = place_type
            changed = True
        if parent_place_name is not None and not clean_optional_text(existing.get("parent")):
            parent_id, _ = self._resolve_or_create_place_id(parent_place_name, allow_implicit=True)
            if parent_id is not None:
                existing["parent"] = parent_id
                changed = True
        if last_confirmed_at is not None and self._update_latest_timestamp(existing, "last_confirmed_at", last_confirmed_at):
            changed = True
        
        if source_session_id is not None:
            existing["source_session_id"] = source_session_id
            changed = True

        if changed:
            return "updated", f"Updated place '{existing['name']}'"
        return "unchanged", f"No changes for place '{existing['name']}'"

    def _apply_fact(self, data: dict[str, Any]) -> tuple[ApplyStatus, str]:
        subject = clean_text(data["subject_name"])
        fact_type = clean_text(data["fact_type"])
        description = clean_text(data["description"])
        confidence = normalize_confidence(data.get("confidence"))
        source_session_id = clean_optional_text(data.get("source_session_id"))
        
        # Try to find the person's ID if they exist
        person_id, _ = self._resolve_or_create_person_id(subject, allow_implicit=False)
        
        # Post-filter: if description contains an alias of an existing person — skip
        person = self._find_person_by_name_or_alias(subject)
        if person:
            description_lower = description.lower()
            for alias in person.get("aliases", []):
                if alias.lower() in description_lower:
                    return "skipped", f"Fact skipped: description contains alias '{alias}' of {subject}"
        
        decision = self.normalizer.find_fact_match(self.facts, subject, fact_type, description, person_id=person_id)

        if decision.status == "none":
            fact = {
                "fact_id": generate_id("fact"),
                "subject": subject,
                "fact_type": fact_type,
                "description": description,
                "confidence": confidence,
                "updated_at": now_iso(),
            }
            if person_id is not None:
                fact["person_id"] = person_id
            if source_session_id is not None:
                fact["source_session_id"] = source_session_id
            self.facts.append(fact)
            return "created", f"Created fact for '{subject}'"

        if decision.status == "exact":
            existing = decision.matched
            if existing is None:
                return "skipped", "Internal error: exact match but no record"
            changed = False
            if confidence > float(existing.get("confidence", 0.0)):
                existing["confidence"] = confidence
                changed = True
            if source_session_id is not None and existing.get("source_session_id") != source_session_id:
                existing["source_session_id"] = source_session_id
                changed = True
            if existing.get("fact_type") != fact_type:
                existing["fact_type"] = fact_type
                changed = True
            if existing.get("description") != description:
                existing["description"] = description
                changed = True
            if changed or "updated_at" not in existing:
                existing["updated_at"] = now_iso()
                changed = True
            if changed:
                return "updated", f"Updated fact for '{subject}' (exact match)"
            return "unchanged", f"No changes for fact '{description}'"

        if decision.status == "conflict":
            existing = decision.matched
            if existing is None:
                 return "skipped", "Internal error: conflict match but no record"
            task_id = generate_id("task")
            self.conflict_queue.append({
                "task_id": task_id,
                "category": "fact",
                "old_data": {
                    "fact_id": existing["fact_id"],
                    "fact_type": existing["fact_type"],
                    "description": existing["description"],
                    "person_id": existing.get("person_id")
                },
                "new_data": {
                    "fact_type": fact_type,
                    "description": description,
                    "confidence": confidence,
                    "source_session_id": source_session_id
                }
            })
            return "pending", f"Fact conflict queued for LLM: {task_id}"

        return "skipped", f"Unknown decision status: {decision.status}"

    def _apply_goal(self, data: dict[str, Any]) -> tuple[ApplyStatus, str]:
        description = clean_text(data["description"])
        person_name = clean_optional_text(data.get("person_name"))
        due_at = normalize_date_only(data.get("due_at"))
        source_session_id = clean_optional_text(data.get("source_session_id"))
        status = clean_optional_text(data.get("status")) or "active"
        person_id = None
        implicit_person_created = False
        if person_name is not None:
            person_id, implicit_person_created = self._resolve_or_create_person_id(
                person_name,
                allow_implicit=LONG_MEMORY_PEOPLE_IMPLICIT_CREATE_FROM_GOAL,
            )
        decision = self.normalizer.find_goal_match(self.goals, description, person_id, status)

        if decision.status == "none":
            goal = {
                "goal_id": generate_id("goal"),
                "description": description,
                "status": status,
                "created_at": now_iso(),
                "updated_at": now_iso(),
            }
            if person_id is not None:
                goal["person_id"] = person_id
            if due_at is not None:
                goal["due_at"] = due_at
            if source_session_id is not None:
                goal["source_session_id"] = source_session_id
            self.goals.append(goal)
            detail = f"Created goal '{description}'"
            if implicit_person_created and person_name is not None:
                detail += f" and implicit person '{person_name}'"
            return "created", detail

        if decision.status == "exact":
            existing = decision.matched
            if existing is None:
                return "skipped", "Internal error: exact goal match but no record"
            changed = False
            if person_id is not None and not clean_optional_text(existing.get("person_id")):
                existing["person_id"] = person_id
                changed = True
            if due_at is not None and not clean_optional_text(existing.get("due_at")):
                existing["due_at"] = due_at
                changed = True
            if source_session_id is not None and clean_optional_text(existing.get("source_session_id")) != source_session_id:
                existing["source_session_id"] = source_session_id
                changed = True
            if status != clean_optional_text(existing.get("status")):
                existing["status"] = status
                changed = True
            if changed:
                existing["updated_at"] = now_iso()
                return "updated", f"Updated goal '{description}' (exact match)"
            return "unchanged", f"No changes for goal '{description}'"

        if decision.status == "conflict":
            existing = decision.matched
            if existing is None:
                return "skipped", "Internal error: conflict goal match but no record"
            task_id = generate_id("task")
            self.conflict_queue.append({
                "task_id": task_id,
                "category": "goal",
                "old_data": {
                    "goal_id": existing["goal_id"],
                    "description": existing["description"]
                },
                "new_data": {
                    "description": description,
                    "person_id": person_id,
                    "due_at": due_at,
                    "status": status
                }
            })
            return "pending", f"Goal conflict queued for LLM: {task_id}"

        return "skipped", f"Unknown goal decision status: {decision.status}"

    def _apply_experience(self, data: dict[str, Any]) -> tuple[ApplyStatus, str]:
        action = clean_text(data["action"])
        obj = clean_text(data["object"])
        place_name = clean_optional_text(data.get("place_name"))
        reason = clean_optional_text(data.get("reason"))
        effect = clean_optional_text(data.get("effect"))
        status = clean_optional_text(data.get("status")) or "active"
        confidence = normalize_confidence(data.get("confidence"))
        source_session_id = clean_optional_text(data.get("source_session_id"))
        place_id = None
        implicit_place_created = False
        if place_name is not None:
            place_id, implicit_place_created = self._resolve_or_create_place_id(
                place_name,
                allow_implicit=LONG_MEMORY_PLACES_IMPLICIT_CREATE_FROM_EXPERIENCE,
            )
        decision = self.normalizer.find_experience_match(self.experience, action, obj, place_id)

        if decision.status == "none":
            item = {
                "exp_id": generate_id("exp"),
                "action": action,
                "object": obj,
                "status": status,
                "confidence": confidence,
                "created_at": now_iso(),
                "updated_at": now_iso(),
            }
            if place_id is not None:
                item["place_id"] = place_id
            if reason is not None:
                item["reason"] = reason
            if effect is not None:
                item["effect"] = effect
            if source_session_id is not None:
                item["source_session_id"] = source_session_id
            self.experience.append(item)
            if len(self.experience) > LONG_MEMORY_MAX_EXPERIENCE_RECORDS:
                self.experience = self.experience[-LONG_MEMORY_MAX_EXPERIENCE_RECORDS:]
            detail = f"Created experience '{action}' for '{obj}'"
            if implicit_place_created and place_name is not None:
                detail += f" and implicit place '{place_name}'"
            return "created", detail

        if decision.status == "exact":
            existing = decision.matched
            if existing is None:
                return "skipped", "Internal error: exact experience match but no record"
            changed = False
            if reason is not None and clean_optional_text(existing.get("reason")) != reason:
                existing["reason"] = reason
                changed = True
            if effect is not None and clean_optional_text(existing.get("effect")) != effect:
                existing["effect"] = effect
                changed = True
            if status != clean_optional_text(existing.get("status")):
                existing["status"] = status
                changed = True
            if confidence > float(existing.get("confidence", 0.0)):
                existing["confidence"] = confidence
                changed = True
            if source_session_id is not None and existing.get("source_session_id") != source_session_id:
                existing["source_session_id"] = source_session_id
                changed = True
            if changed:
                existing["updated_at"] = now_iso()
                return "updated", f"Updated experience '{action}' for '{obj}' (exact match)"
            return "unchanged", f"No changes for experience '{action}' for '{obj}'"

        if decision.status == "conflict":
            existing = decision.matched
            if existing is None:
                return "skipped", "Internal error: conflict match but no record"
            task_id = generate_id("task")
            self.conflict_queue.append({
                "task_id": task_id,
                "category": "experience",
                "old_data": {
                    "exp_id": existing["exp_id"],
                    "action": existing["action"],
                    "object": existing["object"],
                    "place_id": existing.get("place_id"),
                    "reason": existing.get("reason"),
                    "effect": existing.get("effect"),
                    "status": existing.get("status"),
                },
                "new_data": {
                    "action": action,
                    "object": obj,
                    "place_id": place_id,
                    "reason": reason,
                    "effect": effect,
                    "status": status,
                    "confidence": confidence,
                    "source_session_id": source_session_id,
                }
            })
            return "pending", f"Queued experience conflict for '{action}' / '{obj}' (score={decision.score:.2f})"

        return "skipped", f"Unknown experience decision status: {decision.status}"

    def _apply_episode(self, data: dict[str, Any]) -> tuple[ApplyStatus, str]:
        summary = clean_text(data["summary"])
        time = normalize_precise_datetime(data.get("time")) or now_iso()
        source_session_id = clean_optional_text(data.get("source_session_id"))
        participants = unique_strings(data.get("participants", []))
        place_name = clean_optional_text(data.get("place_name"))
        participant_ids: list[str] = []
        for participant_name in participants:
            person_id, _ = self._resolve_or_create_person_id(
                participant_name,
                allow_implicit=LONG_MEMORY_PEOPLE_IMPLICIT_CREATE_FROM_EPISODE_PARTICIPANTS,
            )
            if person_id is not None:
                participant_ids.append(person_id)
        place_id = None
        if place_name is not None:
            place_id, _ = self._resolve_or_create_place_id(
                place_name,
                allow_implicit=LONG_MEMORY_PLACES_IMPLICIT_CREATE_FROM_EPISODES,
            )

        episode_key = self._build_episode_key(time, summary, source_session_id, participant_ids)
        if episode_key in self.existing_episode_keys:
            return "unchanged", f"Duplicate episode skipped for '{summary}'"

        episode = {
            "episode_id": generate_id("episode"),
            "time": time,
            "participants": participant_ids,
            "summary": summary,
        }
        if place_id is not None:
            episode["place_id"] = place_id
        if source_session_id is not None:
            episode["source_session_id"] = source_session_id
        self.pending_episode_entries.append(episode)
        self.existing_episode_keys.add(episode_key)
        return "appended", f"Queued episode '{summary}'"

    def _apply_persona(self, data: dict[str, Any]) -> tuple[ApplyStatus, str]:
        trait = clean_optional_text(data.get("trait"))
        self_perception = clean_optional_text(data.get("self_perception"))
        source_session_id = clean_optional_text(data.get("source_session_id"))
        
        changed = False
        details = []
        
        if trait:
            if "current_roles" not in self.reflections:
                self.reflections["current_roles"] = []
            
            existing_traits = [r["trait"] for r in self.reflections["current_roles"]]
            
            # Split by comma (ignoring commas inside parentheses) and the conjunction " and "
            raw_traits = self._split_traits(trait)
            traits_to_add = []
            for rt in raw_traits:
                sub_parts = re.split(r"\s+and\s+", rt)
                for sp in sub_parts:
                    if sp.strip():
                        traits_to_add.append(sp.strip())
            
            for t in traits_to_add:
                t_norm = normalize_text(t)
                if not any(normalize_text(et) == t_norm for et in existing_traits):
                    self.reflections["current_roles"].append({
                        "trait": t,
                        "created_at": now_iso(),
                        "origin_session": source_session_id or "unknown"
                    })
                    existing_traits.append(t)
                    changed = True
                    details.append(f"Added trait '{t}'")
                else:
                    details.append(f"Trait '{t}' already exists")
            # Enforce the role limit from config
            roles = self.reflections["current_roles"]
            if len(roles) > LONG_MEMORY_MAX_PERSONA_ROLES:
                self.reflections["current_roles"] = roles[-LONG_MEMORY_MAX_PERSONA_ROLES:]
        
        if self_perception:
            if "self_perception" not in self.reflections or not isinstance(self.reflections["self_perception"], list):
                # Migrate old string format to new list format
                old_val = self.reflections.get("self_perception")
                self.reflections["self_perception"] = []
                if isinstance(old_val, str) and old_val.strip():
                    self.reflections["self_perception"].append({
                        "insight": old_val,
                        "created_at": now_iso(),
                        "origin_session": "migration"
                    })
            
            # Add new insight if it differs from the last one
            history = self.reflections["self_perception"]
            is_new = True
            if history:
                last_insight = history[-1].get("insight", "")
                if normalize_text(last_insight) == normalize_text(self_perception):
                    is_new = False
            
            if is_new:
                history.append({
                    "insight": self_perception,
                    "created_at": now_iso(),
                    "origin_session": source_session_id or "unknown"
                })
                # Enforce the history limit from config
                if len(history) > LONG_MEMORY_MAX_PERSONA_HISTORY:
                    self.reflections["self_perception"] = history[-LONG_MEMORY_MAX_PERSONA_HISTORY:]
                
                changed = True
                details.append(f"Added self-perception insight: '{self_perception}'")
            else:
                details.append("Self-perception insight already exists as latest")
        
        if not changed:
            return "unchanged", "; ".join(details) if details else "No persona data provided"
        
        return "updated", "; ".join(details)

    def _find_person_by_name_or_alias(self, person_name: str) -> dict[str, Any] | None:
        decision = self.normalizer.find_person_match(self.people, person_name)
        return decision.matched

    def _find_place_by_name_or_alias(self, place_name: str) -> dict[str, Any] | None:
        decision = self.normalizer.find_place_match(self.places, place_name)
        return decision.matched

    def _find_fact(self, subject: str, fact_type: str, description: str) -> dict[str, Any] | None:
        decision = self.normalizer.find_fact_match(self.facts, subject, fact_type, description)
        return decision.matched

    def _find_goal(self, description: str, person_id: str | None, status: str) -> dict[str, Any] | None:
        decision = self.normalizer.find_goal_match(self.goals, description, person_id, status)
        return decision.matched

    def _merge_aliases(self, record: dict[str, Any], new_aliases: list[str]) -> bool:
        if "aliases" not in record:
            record["aliases"] = []
        old_aliases_norm = {normalize_text(a) for a in record["aliases"]}
        added = False
        for alias in new_aliases:
            alias_clean = clean_text(alias)
            if alias_clean and normalize_text(alias_clean) not in old_aliases_norm:
                record["aliases"].append(alias_clean)
                old_aliases_norm.add(normalize_text(alias_clean))
                added = True
        return added

    def _find_experience(self, action: str, obj: str, place_id: str | None) -> dict[str, Any] | None:
        decision = self.normalizer.find_experience_match(self.experience, action, obj, place_id)
        return decision.matched

    def _resolve_or_create_person_id(self, person_name: str, allow_implicit: bool = True) -> tuple[str | None, bool]:
        if is_placeholder_person_reference(person_name):
            return None, False
        decision = self.normalizer.find_person_match(self.people, person_name)
        if decision.matched is not None:
            return str(decision.matched["person_id"]), False
        if not allow_implicit:
            return None, False
        person = {
            "person_id": generate_id("person"),
            "name": clean_text(person_name),
            "aliases": [],
        }
        self.people.append(person)
        return str(person["person_id"]), True

    def _resolve_or_create_place_id(self, place_name: str, allow_implicit: bool = False) -> tuple[str | None, bool]:
        decision = self.normalizer.find_place_match(self.places, place_name)
        if decision.matched is not None:
            return str(decision.matched["place_id"]), False
        if not allow_implicit:
            return None, False
        place = {
            "place_id": generate_id("place"),
            "name": clean_text(place_name),
            "aliases": [],
        }
        self.places.append(place)
        return str(place["place_id"]), True

    def _load_existing_episode_keys(self) -> set[str]:
        if not EPISODES_PATH.exists():
            return set()
        keys: set[str] = set()
        for line in EPISODES_PATH.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            keys.add(
                self._build_episode_key(
                    clean_optional_text(item.get("time")) or "",
                    clean_optional_text(item.get("summary")) or "",
                    clean_optional_text(item.get("source_session_id")),
                    item.get("participants", []),
                )
            )
        return keys

    async def _append_episodes_async(self) -> None:
        if not self.pending_episode_entries:
            return

        def _write() -> None:
            with EPISODES_PATH.open("a", encoding="utf-8") as file_handle:
                for episode in self.pending_episode_entries:
                    file_handle.write(json.dumps(episode, ensure_ascii=False) + "\n")
                file_handle.flush()
                os.fsync(file_handle.fileno())

        await asyncio.to_thread(_write)

    async def _save_json_async(self, path: Path, payload: dict[str, Any]) -> None:
        def _write() -> None:
            fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
            try:
                with os.fdopen(fd, 'w', encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
                os.replace(tmp_path, path)
            except Exception:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise
        await asyncio.to_thread(_write)

    async def _save_collection_async(self, path: Path, payload: dict[str, Any]) -> None:
        def _write() -> None:
            fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
            try:
                with os.fdopen(fd, 'w', encoding="utf-8") as f:
                    f.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
                os.replace(tmp_path, path)
            except Exception:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise
        await asyncio.to_thread(_write)

    def _load_json(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            logger.debug("Memory collection not found at %s, starting empty", path)
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.error("CRITICAL: %s is corrupted: %s. Collection will be EMPTY.", path, exc)
            return {}
        except Exception as exc:
            logger.error("Failed to load memory collection from %s: %s", path, exc)
            return {}

    def _load_collection(self, path: Path, root_key: str) -> dict[str, Any]:
        if not path.exists():
            return {root_key: []}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Expected object in {path}")
        items = payload.get(root_key)
        if not isinstance(items, list):
            raise ValueError(f"Expected list '{root_key}' in {path}")
        return payload

    def _backup_files(self) -> None:
        files_to_backup = [PEOPLE_PATH, PLACES_PATH, FACTS_PATH, GOALS_PATH, EXPERIENCE_PATH, REFLECTIONS_PATH]
        for path in files_to_backup:
            if path.exists():
                try:
                    self._rotate_backups(path)
                except Exception as e:
                    logger.warning("Failed to create backup for %s: %s", path, e)

    def _rotate_backups(self, path: Path) -> None:
        max_backups = max(int(LONG_MEMORY_MAX_BACKUPS), 0)
        if max_backups <= 0:
            return
        bak_path = path.with_suffix(path.suffix + ".bak")
        if max_backups > 1:
            oldest_numbered = Path(f"{bak_path}.{max_backups - 1}")
            if oldest_numbered.exists():
                oldest_numbered.unlink()
            for index in range(max_backups - 2, 0, -1):
                source = Path(f"{bak_path}.{index}")
                target = Path(f"{bak_path}.{index + 1}")
                if source.exists():
                    os.replace(source, target)
            if bak_path.exists():
                os.replace(bak_path, Path(f"{bak_path}.1"))
        shutil.copy2(path, bak_path)

    def _update_latest_timestamp(self, record: dict[str, Any], field_name: str, candidate_value: str) -> bool:
        existing_value = clean_optional_text(record.get(field_name))
        normalized_candidate_value = normalize_precise_datetime(candidate_value)
        if normalized_candidate_value is None:
            return False
        existing_dt = parse_iso_datetime(existing_value)
        candidate_dt = parse_iso_datetime(normalized_candidate_value)
        if existing_value is None:
            record[field_name] = normalized_candidate_value
            return True
        if candidate_dt is not None and existing_dt is not None:
            if candidate_dt > existing_dt:
                record[field_name] = normalized_candidate_value
                return True
            return False
        if candidate_dt is not None and existing_dt is None:
            record[field_name] = normalized_candidate_value
            return True
        if candidate_dt is None and existing_dt is None and normalized_candidate_value > existing_value:
            record[field_name] = normalized_candidate_value
            return True
        return False

    def _candidate_alias_from_primary_name(self, existing_name: Any, candidate_name: str) -> list[str]:
        existing_name_text = clean_optional_text(existing_name)
        if existing_name_text is None:
            return []
        if normalize_text(existing_name_text) == normalize_text(candidate_name):
            return []
        return [candidate_name]

    def _build_episode_key(
        self,
        time: str,
        summary: str,
        source_session_id: str | None,
        participant_ids: list[str] | None = None,
    ) -> str:
        normalized_time = normalize_precise_datetime(time) or normalize_text(time)
        # Normalize summary: remove frequent stop words for better deduplication
        text = normalize_text(summary)
        stop_words = {"joint", "together", "today", "yesterday"}
        tokens = [t for t in text.split() if t not in stop_words]
        normalized_summary = " ".join(tokens)
        parts = [
            normalized_time,
            normalized_summary,
            normalize_text(source_session_id or ""),
        ]
        if participant_ids:
            parts.append("|".join(sorted(participant_ids)))
        return "|".join(parts)

    def _split_traits(self, text: str) -> list[str]:
        """Splits text by commas, ignoring commas inside parentheses."""
        parts = []
        current = ""
        depth = 0
        for char in text:
            if char == "(":
                depth += 1
                current += char
            elif char == ")":
                depth -= 1
                current += char
            elif char == "," and depth == 0:
                if current.strip():
                    parts.append(current.strip())
                current = ""
            else:
                current += char
        if current.strip():
            parts.append(current.strip())
        return parts


async def apply_memory_operations_async(payload: dict[str, Any] | list[Any], transcript: str | None = None) -> dict[str, Any]:
    applier = LongMemoryApply()
    return await applier.apply_payload(payload, transcript=transcript)


def apply_memory_operations(payload: dict[str, Any] | list[Any], transcript: str | None = None) -> dict[str, Any]:
    """Synchronous wrapper for CLI."""
    return asyncio.run(apply_memory_operations_async(payload, transcript=transcript))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operations-file", required=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    payload = json.loads(Path(args.operations_file).read_text(encoding="utf-8"))
    result = apply_memory_operations(payload)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
