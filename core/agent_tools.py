from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from inspect import isawaitable
from typing import Any

from google.genai import types

from core.errors import ToolExecutionError
from memory_engine.long_memory_query_service import LongMemoryQueryService


ToolHandler = Callable[[dict[str, Any]], dict[str, Any] | Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class RegisteredTool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def to_function_declaration(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class AgentToolExecutor:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register_tool(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: ToolHandler,
    ) -> None:
        if not name:
            raise ValueError("Tool name cannot be empty.")
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")
        self._tools[name] = RegisteredTool(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
        )

    def has_tools(self) -> bool:
        return bool(self._tools)

    def get_tool_definitions(self) -> list[types.Tool]:
        if not self._tools:
            return []
        declarations = [tool.to_function_declaration() for tool in self._tools.values()]
        return [types.Tool(function_declarations=declarations)]

    async def execute(self, tool_name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
        tool = self._tools.get(tool_name)
        if tool is None:
            raise ToolExecutionError(f"Unknown tool: {tool_name}")

        payload = arguments or {}
        if not isinstance(payload, dict):
            raise ToolExecutionError(f"Tool {tool_name} arguments must be an object.")

        try:
            result = tool.handler(payload)
            if isawaitable(result):
                result = await result
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"Tool {tool_name} execution error: {exc}") from exc

        if not isinstance(result, dict):
            raise ToolExecutionError(
                f"Tool {tool_name} returned an invalid result: expected an object."
            )
        return result


def _require_string(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str):
        raise ToolExecutionError(f"Argument {key} must be a string.")
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        raise ToolExecutionError(f"Argument {key} cannot be empty.")
    return cleaned


def _optional_string(arguments: dict[str, Any], key: str) -> str | None:
    value = arguments.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ToolExecutionError(f"Argument {key} must be a string.")
    cleaned = " ".join(value.strip().split())
    return cleaned or None


def _optional_bool(arguments: dict[str, Any], key: str, default: bool) -> bool:
    value = arguments.get(key)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ToolExecutionError(f"Argument {key} must be a boolean.")
    return value


def _optional_limit(arguments: dict[str, Any], key: str, default: int) -> int:
    value = arguments.get(key)
    if value is None:
        return default
    if not isinstance(value, int):
        raise ToolExecutionError(f"Argument {key} must be an integer.")
    if value < 1 or value > 10:
        raise ToolExecutionError(f"Argument {key} must be between 1 and 10.")
    return value


def _optional_goal_status(arguments: dict[str, Any], key: str, default: str) -> str:
    value = arguments.get(key)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ToolExecutionError(f"Argument {key} must be a string.")
    cleaned = value.strip().lower()
    if cleaned not in {"active", "done", "any"}:
        raise ToolExecutionError(f"Argument {key} must be one of: active, done, any.")
    return cleaned


def _optional_experience_status(arguments: dict[str, Any], key: str, default: str) -> str:
    value = arguments.get(key)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ToolExecutionError(f"Argument {key} must be a string.")
    cleaned = value.strip().lower()
    if cleaned not in {"active", "inactive", "any"}:
        raise ToolExecutionError(f"Argument {key} must be one of: active, inactive, any.")
    return cleaned


def build_default_agent_tool_executor() -> AgentToolExecutor:
    query_service = LongMemoryQueryService()
    executor = AgentToolExecutor()

    async def memory_lookup_person(arguments: dict[str, Any]) -> dict[str, Any]:
        return await query_service.lookup_person(
            query=_require_string(arguments, "query"),
            include_related=_optional_bool(arguments, "include_related", True),
            facts_limit=_optional_limit(arguments, "facts_limit", 5),
            goals_limit=_optional_limit(arguments, "goals_limit", 5),
            episodes_limit=_optional_limit(arguments, "episodes_limit", 5),
        )

    executor.register_tool(
        name="memory_lookup_person",
        description="Searches for a person in long-term memory and, if needed, returns related facts, goals and recent events.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Person name or alias to search in long-term memory.",
                },
                "include_related": {
                    "type": "boolean",
                    "description": "If true, return related facts, goals and episodes for the found person.",
                },
                "facts_limit": {
                    "type": "integer",
                    "description": "Maximum facts to return. From 1 to 10.",
                },
                "goals_limit": {
                    "type": "integer",
                    "description": "Maximum goals to return. From 1 to 10.",
                },
                "episodes_limit": {
                    "type": "integer",
                    "description": "Maximum recent episodes to return. From 1 to 10.",
                },
            },
            "required": ["query"],
        },
        handler=memory_lookup_person,
    )

    async def memory_lookup_goal(arguments: dict[str, Any]) -> dict[str, Any]:
        return await query_service.lookup_goals(
            query=_optional_string(arguments, "query"),
            person_name=_optional_string(arguments, "person_name"),
            status=_optional_goal_status(arguments, "status", "active"),
            limit=_optional_limit(arguments, "limit", 5),
        )

    executor.register_tool(
        name="memory_lookup_goal",
        description="Searches for goals and plans in long-term memory by topic, person and status.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords to search by goal description.",
                },
                "person_name": {
                    "type": "string",
                    "description": "Name of the person whose goals to find.",
                },
                "status": {
                    "type": "string",
                    "enum": ["active", "done", "any"],
                    "description": "Goal status for filtering.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum goals to return. From 1 to 10.",
                },
            },
            "required": [],
        },
        handler=memory_lookup_goal,
    )

    async def memory_lookup_experience(arguments: dict[str, Any]) -> dict[str, Any]:
        return await query_service.lookup_experience(
            query=_optional_string(arguments, "query"),
            place_name=_optional_string(arguments, "place_name"),
            status=_optional_experience_status(arguments, "status", "active"),
            limit=_optional_limit(arguments, "limit", 5),
        )

    executor.register_tool(
        name="memory_lookup_experience",
        description="Searches long-term memory records of type experience: practical observations, contextual behavior rules and interaction patterns. Use when you need to recall what previously worked or did not work in a similar situation, with a specific object or in a specific place.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords to search by action, object, reason and effect.",
                },
                "place_name": {
                    "type": "string",
                    "description": "Place name to filter experience if the situation is tied to a specific location.",
                },
                "status": {
                    "type": "string",
                    "enum": ["active", "inactive", "any"],
                    "description": "Experience record status for filtering.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum experience records to return. From 1 to 10.",
                },
            },
            "required": [],
        },
        handler=memory_lookup_experience,
    )

    async def memory_recent_episodes(arguments: dict[str, Any]) -> dict[str, Any]:
        return await query_service.recent_episodes(
            query=_optional_string(arguments, "query"),
            person_name=_optional_string(arguments, "person_name"),
            limit=_optional_limit(arguments, "limit", 5),
        )

    executor.register_tool(
        name="memory_recent_episodes",
        description="Returns recent episodes from long-term memory by topic or person.",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords to search by episode summary.",
                },
                "person_name": {
                    "type": "string",
                    "description": "Name of the person whose events to find.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum episodes to return. From 1 to 10.",
                },
            },
            "required": [],
        },
        handler=memory_recent_episodes,
    )

    return executor
