from dataclasses import dataclass
from enum import Enum


class AgentMode(str, Enum):
    WAKEWORD = "wakeword"
    API = "api"


@dataclass
class AgentRuntimeState:
    mode: AgentMode = AgentMode.WAKEWORD
    is_receiving_response: bool = False
    is_playing_audio: bool = False

    def reset(self) -> None:
        self.is_receiving_response = False
        self.is_playing_audio = False
