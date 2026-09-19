from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from compact import CompactState


@dataclass
class LoopState:
    """Hold the mutable conversation state consumed by the agent loop."""

    messages: List[Dict[str, Any]]
    turn_count: int
    transition_reason: Optional[str]


@dataclass
class SessionContext:
    """Group the runtime state that is persisted for one main-agent session."""

    loop_state: LoopState
    compact_state: CompactState
    pending_visible_response: bool = False
