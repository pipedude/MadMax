import asyncio
import json
from pathlib import Path
from typing import Any

from memory_engine.long_memory_normalize import LongMemoryNormalizer, clean_optional_text, normalize_text, normalized_tokens
from memory_engine.memory_config import EPISODES_PATH, EXPERIENCE_PATH, FACTS_PATH, GOALS_PATH, PEOPLE_PATH, PLACES_PATH


class LongMemoryQueryService:
    def __init__(self) -> None:
        self._normalizer = LongMemoryNormalizer()
        self._people_mtime: float | None = None
        self._places_mtime: float | None = None
        self._goals_mtime: float | None = None
        self._facts_mtime: float | None = None
        self._episodes_mtime: float | None = None
        self._experience_mtime: float | None = None
        self._people: list[dict[str, Any]] = []
        self._places: list[dict[str, Any]] = []
        self._goals: list[dict[str, Any]] = []
        self._facts: list[dict[str, Any]] = []
        self._episodes: list[dict[str, Any]] = []
        self._experience: list[dict[str, Any]] = []
        self._person_lookup: dict[str, dict[str, Any]] = {}

    async def lookup_person(
        self,
        *,
        query: str,
        include_related: bool = True,
        facts_limit: int = 5,
        goals_limit: int = 5,
        episodes_limit: int = 5,
    ) -> dict[str, Any]:
        person = await self._resolve_person_async(query)
        if person is None:
            return {
                "ok": True,
                "result": {
                    "matched": False,
                    "query": query,
                    "person": None,
                    "facts": [],
                    "goals": [],
                    "episodes": [],
                },
            }

        result = {
            "matched": True,
            "query": query,
            "person": self._build_person_card(person),
            "facts": [],
            "goals": [],
            "episodes": [],
        }
        if not include_related:
            return {"ok": True, "result": result}

        person_id = clean_optional_text(person.get("person_id"))
        
        # Load everything needed asynchronously
        await asyncio.gather(
            self._ensure_facts_loaded_async(),
            self._ensure_goals_loaded_async(),
            self._ensure_episodes_loaded_async()
        )

        facts = self._get_facts_for_person(person_id=person_id, person_name=str(person.get("name", "")))
        goals = self._get_goals_for_person(person_id=person_id)
        episodes = self._get_episodes_for_person(person_id=person_id)
        
        result["facts"] = facts[: self._bounded_limit(facts_limit)]
        result["goals"] = goals[: self._bounded_limit(goals_limit)]
        result["episodes"] = episodes[: self._bounded_limit(episodes_limit)]
        return {"ok": True, "result": result}

    async def lookup_goals(
        self,
        *,
        query: str | None = None,
        person_name: str | None = None,
        status: str = "active",
        limit: int = 5,
    ) -> dict[str, Any]:
        await self._ensure_goals_loaded_async()
        person = await self._resolve_person_async(person_name) if person_name else None
        if person_name and person is None:
            return {
                "ok": True,
                "result": {
                    "query": query,
                    "person_name": person_name,
                    "status": status,
                    "matched": False,
                    "owner": None,
                    "goals": [],
                },
            }
        person_id = clean_optional_text(person.get("person_id")) if person else None
        query_key = normalize_text(query or "")
        query_tokens = normalized_tokens(query or "")

        matches: list[dict[str, Any]] = []
        for goal in self._goals:
            goal_status = clean_optional_text(goal.get("status")) or "active"
            if status != "any" and goal_status != status:
                continue
            goal_person_id = clean_optional_text(goal.get("person_id"))
            if person_id and goal_person_id != person_id:
                continue
            description = str(goal.get("description", ""))
            if query_key:
                description_key = normalize_text(description)
                if query_key not in description_key:
                    description_tokens = normalized_tokens(description)
                    if not query_tokens or not (query_tokens & description_tokens):
                        continue
            matches.append(self._build_goal_card(goal))

        matches.sort(key=lambda item: (item.get("updated_at") or "", item.get("created_at") or ""), reverse=True)
        return {
            "ok": True,
            "result": {
                "query": query,
                "person_name": person_name,
                "status": status,
                "matched": bool(matches),
                "owner": self._build_person_card(person) if person else None,
                "goals": matches[: self._bounded_limit(limit)],
            },
        }

    async def recent_episodes(
        self,
        *,
        query: str | None = None,
        person_name: str | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        await self._ensure_episodes_loaded_async()
        person = await self._resolve_person_async(person_name) if person_name else None
        if person_name and person is None:
            return {
                "ok": True,
                "result": {
                    "query": query,
                    "person_name": person_name,
                    "matched": False,
                    "person": None,
                    "episodes": [],
                },
            }
        person_id = clean_optional_text(person.get("person_id")) if person else None
        query_key = normalize_text(query or "")
        query_tokens = normalized_tokens(query or "")

        matches: list[dict[str, Any]] = []
        for episode in self._episodes:
            participants = [clean_optional_text(value) for value in episode.get("participants", [])]
            participants = [value for value in participants if value]
            if person_id and person_id not in participants:
                continue
            summary = str(episode.get("summary", ""))
            if query_key:
                summary_key = normalize_text(summary)
                if query_key not in summary_key:
                    summary_tokens = normalized_tokens(summary)
                    if not query_tokens or not (query_tokens & summary_tokens):
                        continue
            matches.append(self._build_episode_card(episode))

        matches.sort(key=lambda item: item.get("time") or "", reverse=True)
        return {
            "ok": True,
            "result": {
                "query": query,
                "person_name": person_name,
                "matched": bool(matches),
                "person": self._build_person_card(person) if person else None,
                "episodes": matches[: self._bounded_limit(limit)],
            },
        }

    async def lookup_experience(
        self,
        *,
        query: str | None = None,
        place_name: str | None = None,
        status: str = "active",
        limit: int = 5,
    ) -> dict[str, Any]:
        await self._ensure_experience_loaded_async()
        place = await self._resolve_place_async(place_name) if place_name else None
        if place_name and place is None:
            return {
                "ok": True,
                "result": {
                    "query": query,
                    "place_name": place_name,
                    "status": status,
                    "matched": False,
                    "place": None,
                    "experience": [],
                },
            }

        place_id = clean_optional_text(place.get("place_id")) if place else None
        query_key = normalize_text(query or "")
        query_tokens = normalized_tokens(query or "")

        matches: list[dict[str, Any]] = []
        for experience in self._experience:
            experience_status = clean_optional_text(experience.get("status")) or "active"
            if status != "any" and experience_status != status:
                continue
            experience_place_id = clean_optional_text(experience.get("place_id"))
            if place_id and experience_place_id != place_id:
                continue

            if query_key:
                haystacks = [
                    str(experience.get("action", "")),
                    str(experience.get("object", "")),
                    str(experience.get("reason", "")),
                    str(experience.get("effect", "")),
                ]
                matched_by_query = False
                for haystack in haystacks:
                    haystack_key = normalize_text(haystack)
                    if query_key and query_key in haystack_key:
                        matched_by_query = True
                        break
                    haystack_tokens = normalized_tokens(haystack)
                    if query_tokens and (query_tokens & haystack_tokens):
                        matched_by_query = True
                        break
                if not matched_by_query:
                    continue

            matches.append(self._build_experience_card(experience))

        matches.sort(key=lambda item: (item.get("updated_at") or "", item.get("created_at") or ""), reverse=True)
        return {
            "ok": True,
            "result": {
                "query": query,
                "place_name": place_name,
                "status": status,
                "matched": bool(matches),
                "place": self._build_place_card(place) if place else None,
                "experience": matches[: self._bounded_limit(limit)],
            },
        }

    async def _resolve_person_async(self, query: str | None) -> dict[str, Any] | None:
        if not query:
            return None
        await self._ensure_people_loaded_async()
        
        # 1. Use unified logic from Normalizer
        decision = self._normalizer.find_person_match(self._people, query)
        if decision.matched is not None:
            return decision.matched

        # 2. Fallback to fuzzy token search (simplified)
        query_key = normalize_text(query)
        query_tokens = normalized_tokens(query)
        
        best_person: dict[str, Any] | None = None
        best_score = 0
        
        for person in self._people:
            # Check name or alias match
            names = [str(person.get("name", "")).lower()] + [str(a).lower() for alias in person.get("aliases", []) for a in ([alias] if isinstance(alias, str) else alias)]
            
            score = 0
            for name in names:
                if query_key in name or name in query_key:
                    score = max(score, 10)
                else:
                    name_tokens = normalized_tokens(name)
                    overlap = len(query_tokens & name_tokens)
                    if overlap > 0:
                        score = max(score, overlap)
            
            if score > best_score:
                best_score = score
                best_person = person

        return best_person if best_score > 0 else None

    async def _ensure_people_loaded_async(self) -> None:
        mtime = self._get_mtime(PEOPLE_PATH)
        if mtime == self._people_mtime:
            return
        self._people = await asyncio.to_thread(self._load_json_file, PEOPLE_PATH, "people")
        self._people_mtime = mtime

    async def _ensure_places_loaded_async(self) -> None:
        mtime = self._get_mtime(PLACES_PATH)
        if mtime == self._places_mtime:
            return
        self._places = await asyncio.to_thread(self._load_json_file, PLACES_PATH, "places")
        self._places_mtime = mtime

    async def _ensure_goals_loaded_async(self) -> None:
        mtime = self._get_mtime(GOALS_PATH)
        if mtime == self._goals_mtime:
            return
        self._goals = await asyncio.to_thread(self._load_json_file, GOALS_PATH, "goals")
        self._goals_mtime = mtime

    async def _ensure_facts_loaded_async(self) -> None:
        mtime = self._get_mtime(FACTS_PATH)
        if mtime == self._facts_mtime:
            return
        self._facts = await asyncio.to_thread(self._load_json_file, FACTS_PATH, "facts")
        self._facts_mtime = mtime

    async def _ensure_episodes_loaded_async(self) -> None:
        mtime = self._get_mtime(EPISODES_PATH)
        if mtime == self._episodes_mtime:
            return
        self._episodes = await asyncio.to_thread(self._load_jsonl_file, EPISODES_PATH)
        self._episodes_mtime = mtime

    async def _ensure_experience_loaded_async(self) -> None:
        mtime = self._get_mtime(EXPERIENCE_PATH)
        if mtime == self._experience_mtime:
            return
        self._experience = await asyncio.to_thread(self._load_json_file, EXPERIENCE_PATH, "experience")
        self._experience_mtime = mtime

    def _get_facts_for_person(self, *, person_id: str | None, person_name: str) -> list[dict[str, Any]]:
        person_key = normalize_text(person_name)
        matches: list[dict[str, Any]] = []
        for fact in self._facts:
            fact_person_id = clean_optional_text(fact.get("person_id"))
            if person_id and fact_person_id == person_id:
                matches.append(self._build_fact_card(fact))
                continue
            if not person_id and normalize_text(str(fact.get("subject", ""))) == person_key:
                matches.append(self._build_fact_card(fact))
        matches.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
        return matches

    def _get_goals_for_person(self, *, person_id: str | None) -> list[dict[str, Any]]:
        if person_id is None:
            return []
        matches = [self._build_goal_card(goal) for goal in self._goals if clean_optional_text(goal.get("person_id")) == person_id]
        matches.sort(key=lambda item: (item.get("updated_at") or "", item.get("created_at") or ""), reverse=True)
        return matches

    def _get_episodes_for_person(self, *, person_id: str | None) -> list[dict[str, Any]]:
        if person_id is None:
            return []
        matches = []
        for episode in self._episodes:
            participants = [clean_optional_text(value) for value in episode.get("participants", [])]
            if person_id in participants:
                matches.append(self._build_episode_card(episode))
        matches.sort(key=lambda item: item.get("time") or "", reverse=True)
        return matches

    def _build_person_card(self, person: dict[str, Any] | None) -> dict[str, Any] | None:
        if person is None:
            return None
        return {
            "person_id": clean_optional_text(person.get("person_id")),
            "name": clean_optional_text(person.get("name")),
            "aliases": [alias for alias in person.get("aliases", []) if isinstance(alias, str) and alias.strip()],
            "last_seen_at": clean_optional_text(person.get("last_seen_at")),
            "source_session_id": clean_optional_text(person.get("source_session_id")),
        }

    def _build_place_card(self, place: dict[str, Any] | None) -> dict[str, Any] | None:
        if place is None:
            return None
        return {
            "place_id": clean_optional_text(place.get("place_id")),
            "name": clean_optional_text(place.get("name")),
            "type": clean_optional_text(place.get("type")),
            "aliases": [alias for alias in place.get("aliases", []) if isinstance(alias, str) and alias.strip()],
            "parent_place_id": clean_optional_text(place.get("parent")),
            "last_confirmed_at": clean_optional_text(place.get("last_confirmed_at")),
            "source_session_id": clean_optional_text(place.get("source_session_id")),
        }

    def _build_goal_card(self, goal: dict[str, Any]) -> dict[str, Any]:
        return {
            "goal_id": clean_optional_text(goal.get("goal_id")),
            "description": clean_optional_text(goal.get("description")),
            "status": clean_optional_text(goal.get("status")) or "active",
            "due_at": clean_optional_text(goal.get("due_at")),
            "person_id": clean_optional_text(goal.get("person_id")),
            "person_name": self._get_person_name(clean_optional_text(goal.get("person_id"))),
            "created_at": clean_optional_text(goal.get("created_at")),
            "updated_at": clean_optional_text(goal.get("updated_at")),
            "source_session_id": clean_optional_text(goal.get("source_session_id")),
        }

    def _build_fact_card(self, fact: dict[str, Any]) -> dict[str, Any]:
        return {
            "fact_id": clean_optional_text(fact.get("fact_id")),
            "subject": clean_optional_text(fact.get("subject")),
            "fact_type": clean_optional_text(fact.get("fact_type")),
            "description": clean_optional_text(fact.get("description")),
            "confidence": fact.get("confidence"),
            "person_id": clean_optional_text(fact.get("person_id")),
            "updated_at": clean_optional_text(fact.get("updated_at")),
            "source_session_id": clean_optional_text(fact.get("source_session_id")),
        }

    def _build_episode_card(self, episode: dict[str, Any]) -> dict[str, Any]:
        participant_ids = [clean_optional_text(value) for value in episode.get("participants", [])]
        participant_ids = [value for value in participant_ids if value]
        return {
            "episode_id": clean_optional_text(episode.get("episode_id")),
            "time": clean_optional_text(episode.get("time")),
            "summary": clean_optional_text(episode.get("summary")),
            "participants": participant_ids,
            "participant_names": [name for name in (self._get_person_name(person_id) for person_id in participant_ids) if name],
            "source_session_id": clean_optional_text(episode.get("source_session_id")),
        }

    def _build_experience_card(self, experience: dict[str, Any]) -> dict[str, Any]:
        place_id = clean_optional_text(experience.get("place_id"))
        return {
            "exp_id": clean_optional_text(experience.get("exp_id")),
            "action": clean_optional_text(experience.get("action")),
            "object": clean_optional_text(experience.get("object")),
            "place_id": place_id,
            "place_name": self._get_place_name(place_id),
            "reason": clean_optional_text(experience.get("reason")),
            "effect": clean_optional_text(experience.get("effect")),
            "status": clean_optional_text(experience.get("status")) or "active",
            "confidence": experience.get("confidence"),
            "created_at": clean_optional_text(experience.get("created_at")),
            "updated_at": clean_optional_text(experience.get("updated_at")),
            "source_session_id": clean_optional_text(experience.get("source_session_id")),
        }

    async def _resolve_place_async(self, query: str | None) -> dict[str, Any] | None:
        if not query:
            return None
        await self._ensure_places_loaded_async()

        decision = self._normalizer.find_place_match(self._places, query)
        if decision.matched is not None:
            return decision.matched

        query_key = normalize_text(query)
        query_tokens = normalized_tokens(query)
        best_place: dict[str, Any] | None = None
        best_score = 0

        for place in self._places:
            names = [str(place.get("name", ""))] + [str(alias) for alias in place.get("aliases", []) if isinstance(alias, str)]
            score = 0
            for name in names:
                normalized_name = normalize_text(name)
                if query_key in normalized_name or normalized_name in query_key:
                    score = max(score, 10)
                else:
                    name_tokens = normalized_tokens(name)
                    overlap = len(query_tokens & name_tokens)
                    if overlap > 0:
                        score = max(score, overlap)
            if score > best_score:
                best_score = score
                best_place = place

        return best_place if best_score > 0 else None

    def _get_person_name(self, person_id: str | None) -> str | None:
        if person_id is None:
            return None
        self._ensure_people_loaded()
        for person in self._people:
            if clean_optional_text(person.get("person_id")) == person_id:
                return clean_optional_text(person.get("name"))
        return None

    def _get_place_name(self, place_id: str | None) -> str | None:
        if place_id is None:
            return None
        self._ensure_places_loaded()
        for place in self._places:
            if clean_optional_text(place.get("place_id")) == place_id:
                return clean_optional_text(place.get("name"))
        return None

    def _ensure_people_loaded(self) -> None:
        mtime = self._get_mtime(PEOPLE_PATH)
        if mtime == self._people_mtime:
            return
        payload = self._load_json_file(PEOPLE_PATH, "people")
        self._people = payload
        self._people_mtime = mtime
        self._rebuild_people_index()

    def _ensure_places_loaded(self) -> None:
        mtime = self._get_mtime(PLACES_PATH)
        if mtime == self._places_mtime:
            return
        self._places = self._load_json_file(PLACES_PATH, "places")
        self._places_mtime = mtime

    def _ensure_goals_loaded(self) -> None:
        mtime = self._get_mtime(GOALS_PATH)
        if mtime == self._goals_mtime:
            return
        self._goals = self._load_json_file(GOALS_PATH, "goals")
        self._goals_mtime = mtime

    def _ensure_facts_loaded(self) -> None:
        mtime = self._get_mtime(FACTS_PATH)
        if mtime == self._facts_mtime:
            return
        self._facts = self._load_json_file(FACTS_PATH, "facts")
        self._facts_mtime = mtime

    def _ensure_episodes_loaded(self) -> None:
        mtime = self._get_mtime(EPISODES_PATH)
        if mtime == self._episodes_mtime:
            return
        self._episodes = self._load_jsonl_file(EPISODES_PATH)
        self._episodes_mtime = mtime

    def _rebuild_people_index(self) -> None:
        lookup: dict[str, dict[str, Any]] = {}
        for person in self._people:
            values = [str(person.get("name", "")), *[str(alias) for alias in person.get("aliases", [])]]
            for value in values:
                key = normalize_text(value)
                if key and key not in lookup:
                    lookup[key] = person
        self._person_lookup = lookup

    def _load_json_file(self, path: Path, root_key: str) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get(root_key, [])
        if not isinstance(records, list):
            return []
        return [record for record in records if isinstance(record, dict)]

    def _load_jsonl_file(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
        return records

    def _get_mtime(self, path: Path) -> float | None:
        if not path.exists():
            return None
        return path.stat().st_mtime

    def _bounded_limit(self, limit: int) -> int:
        if limit < 1:
            return 1
        return min(limit, 10)
