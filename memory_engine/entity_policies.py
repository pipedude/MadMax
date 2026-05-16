import re

from memory_engine.long_memory_normalize import normalize_text

GENERIC_PERSON_REFERENCE_TERMS = {
    "user",
    "interlocutor",
    "human",
    "person",
    "guest",
    "guests",
    "man",
    "woman",
    "child",
    "son",
    "daughter",
    "mom",
    "dad",
    "husband",
    "wife",
    "someone",
}


def get_generic_person_reference_terms_prompt_text() -> str:
    """Returns a formatted list of generic references for insertion into LLM prompts."""
    return ", ".join(f'"{t}"' for t in sorted(GENERIC_PERSON_REFERENCE_TERMS))


def is_placeholder_person_reference(value: str) -> bool:
    normalized_value = normalize_text(value)
    if not normalized_value:
        return True
    if normalized_value == "user's family":
        return False
    if normalized_value in GENERIC_PERSON_REFERENCE_TERMS:
        return True
    tokens = [token for token in re.split(r"[\s_-]+", normalized_value) if token]
    return bool(tokens) and all(token in GENERIC_PERSON_REFERENCE_TERMS for token in tokens)
