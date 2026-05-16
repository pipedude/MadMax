import re
from dataclasses import dataclass
from typing import Any

from memory_engine.memory_config import (
    LONG_MEMORY_FACT_CONTAINS_SIMILARITY_FLOOR,
    LONG_MEMORY_GOAL_CONTAINS_SIMILARITY_FLOOR,
    LONG_MEMORY_MIN_TOKEN_LENGTH,
    LONG_MEMORY_EXACT_MATCH_THRESHOLD,
)


@dataclass(frozen=True)
class MatchDecision:
    matched: dict[str, Any] | None
    status: str  # 'exact', 'conflict', 'none'
    score: float
    reason: str


def normalize_text(value: str) -> str:
    return " ".join(value.strip().split()).casefold()


def clean_text(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("Expected string value")
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        raise ValueError("Expected non-empty string value")
    return cleaned


def clean_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Expected optional text value to be a string")
    cleaned = " ".join(value.strip().split())
    return cleaned or None


def unique_strings(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = " ".join(value.strip().split())
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def normalized_tokens(value: str) -> set[str]:
    normalized = normalize_text(value)
    if not normalized:
        return set()
    # Remove deterministic stop words, keep only clean tokenization
    tokens = re.findall(r"[a-z0-9-]+", normalized)
    return {token for token in tokens if len(token) >= LONG_MEMORY_MIN_TOKEN_LENGTH}


def token_similarity(left: str, right: str) -> float:
    left_tokens = normalized_tokens(left)
    right_tokens = normalized_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = left_tokens & right_tokens
    union = left_tokens | right_tokens
    return len(intersection) / len(union)


def contains_normalized_text(left: str, right: str) -> bool:
    left_normalized = normalize_text(left)
    right_normalized = normalize_text(right)
    return bool(left_normalized and right_normalized and (left_normalized in right_normalized or right_normalized in left_normalized))


class LongMemoryNormalizer:
    def find_person_match(self, people: list[dict[str, Any]], candidate_name: str) -> MatchDecision:
        candidate_key = normalize_text(candidate_name)
        for person in people:
            if normalize_text(str(person.get("name", ""))) == candidate_key:
                return MatchDecision(matched=person, status="exact", score=1.0, reason="exact_name")
            for alias in person.get("aliases", []):
                if normalize_text(str(alias)) == candidate_key:
                    return MatchDecision(matched=person, status="exact", score=1.0, reason="exact_alias")
        return MatchDecision(matched=None, status="none", score=0.0, reason="no_exact_match")

    def find_place_match(self, places: list[dict[str, Any]], candidate_name: str) -> MatchDecision:
        candidate_key = normalize_text(candidate_name)
        for place in places:
            if normalize_text(str(place.get("name", ""))) == candidate_key:
                return MatchDecision(matched=place, status="exact", score=1.0, reason="exact_name")
            for alias in place.get("aliases", []):
                if normalize_text(str(alias)) == candidate_key:
                    return MatchDecision(matched=place, status="exact", score=1.0, reason="exact_alias")
        return MatchDecision(matched=None, status="none", score=0.0, reason="no_exact_match")

    def find_fact_match(
        self,
        facts: list[dict[str, Any]],
        subject: str,
        fact_type: str,
        description: str,
        person_id: str | None = None,
    ) -> MatchDecision:
        subject_key = normalize_text(subject)
        best_match: dict[str, Any] | None = None
        best_score = 0.0

        for fact in facts:
            # 1. Check by person_id or subject name
            existing_person_id = clean_optional_text(fact.get("person_id"))
            if person_id and existing_person_id:
                if person_id != existing_person_id:
                    continue
            elif normalize_text(str(fact.get("subject", ""))) != subject_key:
                continue
            
            # We NO LONGER block search by fact_type.
            # If the subject matched, we compare descriptions even if types differ.
            
            existing_description = str(fact.get("description", ""))
            
            # 100% text match
            if normalize_text(existing_description) == normalize_text(description):
                return MatchDecision(matched=fact, status="exact", score=1.0, reason="exact_description")
            
            similarity = token_similarity(existing_description, description)
            if contains_normalized_text(existing_description, description):
                similarity = max(similarity, LONG_MEMORY_FACT_CONTAINS_SIMILARITY_FLOOR)
            
            if similarity > best_score:
                best_score = similarity
                best_match = fact

        if best_match is not None:
            if best_score >= LONG_MEMORY_EXACT_MATCH_THRESHOLD:
                return MatchDecision(matched=best_match, status="exact", score=best_score, reason="near_exact_description")
            
            # Conflict threshold lowered to 15% (0.15).
            # Everything above goes to the Surgeon.
            if best_score >= 0.15:
                return MatchDecision(matched=best_match, status="conflict", score=best_score, reason="gray_zone_description")
        
        return MatchDecision(matched=None, status="none", score=best_score, reason="no_fact_match")

    def find_goal_match(
        self,
        goals: list[dict[str, Any]],
        description: str,
        person_id: str | None,
        status: str,
    ) -> MatchDecision:
        best_match: dict[str, Any] | None = None
        best_score = 0.0
        for goal in goals:
            existing_person_id = clean_optional_text(goal.get("person_id"))
            if person_id and existing_person_id and existing_person_id != person_id:
                continue
                
            existing_description = str(goal.get("description", ""))
            
            if normalize_text(existing_description) == normalize_text(description):
                return MatchDecision(matched=goal, status="exact", score=1.0, reason="exact_description")
            
            similarity = token_similarity(existing_description, description)
            if contains_normalized_text(existing_description, description):
                similarity = max(similarity, LONG_MEMORY_GOAL_CONTAINS_SIMILARITY_FLOOR)
            
            if similarity > best_score:
                best_score = similarity
                best_match = goal

        if best_match is not None:
            if best_score >= LONG_MEMORY_EXACT_MATCH_THRESHOLD:
                return MatchDecision(matched=best_match, status="exact", score=best_score, reason="near_exact_goal")
            if best_score >= 0.15:
                return MatchDecision(matched=best_match, status="conflict", score=best_score, reason="gray_zone_goal")
        
        return MatchDecision(matched=None, status="none", score=best_score, reason="no_goal_match")

    def find_experience_match(
        self,
        experience: list[dict[str, Any]],
        action: str,
        obj: str,
        place_id: str | None,
    ) -> MatchDecision:
        action_key = normalize_text(action)
        object_key = normalize_text(obj)
        best_match = None
        best_score = 0.0

        for item in experience:
            existing_action = str(item.get("action", ""))
            existing_object = str(item.get("object", ""))

            # Exact match: action + object + place
            if (normalize_text(existing_action) == action_key and
                normalize_text(existing_object) == object_key):
                existing_place_id = clean_optional_text(item.get("place_id"))
                if existing_place_id == place_id:
                    return MatchDecision(matched=item, status="exact", score=1.0, reason="same_action_place_and_object")

            # Fuzzy matching by action + object
            action_sim = token_similarity(existing_action, action)
            object_sim = token_similarity(existing_object, obj)

            if contains_normalized_text(existing_action, action):
                action_sim = max(action_sim, LONG_MEMORY_FACT_CONTAINS_SIMILARITY_FLOOR)
            if contains_normalized_text(existing_object, obj):
                object_sim = max(object_sim, LONG_MEMORY_FACT_CONTAINS_SIMILARITY_FLOOR)

            score = (action_sim + object_sim) / 2.0

            if score > best_score:
                best_score = score
                best_match = item

        if best_match is not None:
            if best_score >= LONG_MEMORY_EXACT_MATCH_THRESHOLD:
                return MatchDecision(matched=best_match, status="exact", score=best_score, reason="near_exact_experience")
            if best_score >= 0.15:
                return MatchDecision(matched=best_match, status="conflict", score=best_score, reason="gray_zone_experience")

        return MatchDecision(matched=None, status="none", score=best_score, reason="no_experience_match")
