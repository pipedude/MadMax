import json
from dataclasses import dataclass, field
from typing import Any

ALLOWED_FACT_TYPES = {
    "preference",
    "relation",
    "profile",
    "household",
    "behavior",
    "other",
}
ALLOWED_GOAL_STATUSES = {"active", "inactive"}
ALLOWED_EXPERIENCE_STATUSES = {"active", "inactive"}

OPERATION_SCHEMAS: dict[str, dict[str, tuple[str, ...]]] = {
    "upsert_person_candidate": {
        "required": ("person_name",),
        "optional": ("aliases", "role", "last_seen_at", "source_session_id"),
    },
    "upsert_place_candidate": {
        "required": ("place_name",),
        "optional": ("aliases", "place_type", "parent_place_name", "last_confirmed_at", "source_session_id"),
    },
    "upsert_fact_candidate": {
        "required": ("subject_name", "fact_type", "description"),
        "optional": ("confidence", "source_session_id"),
    },
    "add_goal_candidate": {
        "required": ("description",),
        "optional": ("person_name", "due_at", "source_session_id", "status"),
    },
    "upsert_experience_candidate": {
        "required": ("action", "object"),
        "optional": ("place_name", "reason", "effect", "status", "confidence", "source_session_id"),
    },
    "add_episode_candidate": {
        "required": ("summary",),
        "optional": ("participants", "place_name", "time", "source_session_id"),
    },
    "upsert_persona_candidate": {
        "required": (),
        "optional": ("trait", "self_perception", "source_session_id"),
    },
}

CONFLICT_CATEGORIES = ["fact", "goal", "experience"]

@dataclass(frozen=True)
class Operation:
    op: str
    data: dict[str, Any]

    def validate(self) -> None:
        if self.op not in OPERATION_SCHEMAS:
            raise ValueError(f"Unsupported operation: {self.op}")

        if not isinstance(self.data, dict):
            raise ValueError("Operation data must be a dictionary")

        schema = OPERATION_SCHEMAS[self.op]
        required_fields = set(schema["required"])
        optional_fields = set(schema["optional"])
        allowed_fields = required_fields | optional_fields

        missing_fields = [field_name for field_name in schema["required"] if field_name not in self.data]
        if missing_fields:
            raise ValueError(f"Missing required fields for {self.op}: {', '.join(missing_fields)}")

        unknown_fields = [field_name for field_name in self.data if field_name not in allowed_fields]
        if unknown_fields:
            raise ValueError(f"Unknown fields for {self.op}: {', '.join(unknown_fields)}")

        for field_name in required_fields:
            value = self.data[field_name]
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Field {field_name} for {self.op} must be a non-empty string")

        for field_name in optional_fields:
            if field_name not in self.data or self.data[field_name] is None:
                continue

            value = self.data[field_name]
            if field_name in {"aliases", "participants"}:
                if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
                    raise ValueError(f"Field {field_name} for {self.op} must be a list of non-empty strings")
                continue

            if field_name == "confidence":
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Field {field_name} for {self.op} must be a number")
                if not (0.0 <= value <= 1.0):
                    raise ValueError(f"Field {field_name} for {self.op} must be between 0.0 and 1.0")
                continue

            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Field {field_name} for {self.op} must be a non-empty string when provided")

        if self.op == "upsert_fact_candidate":
            fact_type = self.data["fact_type"]
            if fact_type not in ALLOWED_FACT_TYPES:
                raise ValueError(f"Unsupported fact_type for {self.op}: {fact_type}")

        if self.op == "add_goal_candidate" and "status" in self.data:
            status = self.data["status"]
            if status not in ALLOWED_GOAL_STATUSES:
                raise ValueError(f"Unsupported status for {self.op}: {status}")

        if self.op == "upsert_experience_candidate" and "status" in self.data:
            status = self.data["status"]
            if status not in ALLOWED_EXPERIENCE_STATUSES:
                raise ValueError(f"Unsupported status for {self.op}: {status}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "op": self.op,
            "data": self.data,
        }


@dataclass(frozen=True)
class ExtractionResult:
    operations: list[Operation] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        return {"operations": [operation.to_dict() for operation in self.operations]}


def normalize_operation_item(item: dict[str, Any]) -> dict[str, Any]:
    if "op" in item:
        op = str(item.get("op", "")).strip()
        data = item.get("data")
        if data is None:
            data = {
                key: value
                for key, value in item.items()
                if key not in {"op", "data"}
            }
        return {
            "op": op,
            "data": data,
        }

    if "operation" in item:
        op = str(item.get("operation", "")).strip()
        data = item.get("data")
        if data is None:
            data = {
                key: value
                for key, value in item.items()
                if key not in {"operation", "data"}
            }
        return {
            "op": op,
            "data": data,
        }

    matching_keys = [key for key in item if key in OPERATION_SCHEMAS]
    if len(matching_keys) == 1 and isinstance(item[matching_keys[0]], dict):
        return {
            "op": matching_keys[0],
            "data": item[matching_keys[0]],
        }

    return item


def validate_operations_payload(payload: dict[str, Any] | list[Any]) -> dict[str, list[dict[str, Any]]]:
    if isinstance(payload, list):
        operations = payload
    else:
        operations = payload.get("operations")
    if not isinstance(operations, list):
        raise ValueError("Payload must contain an operations list")

    validated_operations: list[Operation] = []
    for index, item in enumerate(operations):
        if not isinstance(item, dict):
            raise ValueError(f"Each operation must be a dictionary, got {type(item).__name__} at index {index}")
        normalized_item = normalize_operation_item(item)
        op = str(normalized_item.get("op", "")).strip()
        data = normalized_item.get("data", {})
        if not op:
            raise ValueError(
                f"Operation at index {index} does not contain a supported non-empty op field: {json.dumps(item, ensure_ascii=False)}"
            )
        validated_operations.append(
            Operation(
                op=op,
                data=data,
            )
        )

    return ExtractionResult(operations=validated_operations).to_dict()
