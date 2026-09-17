from typing import Any, Dict, List

from background import RuntimeTaskRecord
from compact import CompactState
from persistence import CHECKPOINT_VERSION


def subagent_id_for(session_id: str, tool_call_id: str) -> str:
    return f"subagent:{session_id}:{tool_call_id}"


def build_main_checkpoint(
    messages: List[Dict[str, Any]],
    turn_count: int,
    transition_reason: str | None,
    compact_state: CompactState,
    background_tasks: List[RuntimeTaskRecord],
    pending_visible_response: bool,
) -> Dict[str, Any]:
    """生成MainAgent的checkpoint"""
    payload: Dict[str, Any] = {
        "version": CHECKPOINT_VERSION,
        "loop_state": {
            "messages": messages,
            "turn_count": turn_count,
            "transition_reason": transition_reason,
        },
        "compact_state": compact_state.model_dump(),
        "background_tasks": [task.model_dump() for task in background_tasks],
        "pending_visible_response": pending_visible_response,
    }
    return payload


def build_subagent_checkpoint(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    max_turns: int,
    turn_count: int,
    compact_state: CompactState,
) -> Dict[str, Any]:
    """生成Subagent的checkpoint"""
    payload: Dict[str, Any] = {
        "version": CHECKPOINT_VERSION,
        "context": {
            "messages": messages,
            "tools": tools,
            "max_turns": max_turns,
            "turn_count": turn_count,
        },
        "compact_state": compact_state.model_dump(),
    }
    return payload


def pending_tool_calls(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """收集所有AI决定调用但没执行的tool call"""
    assistant_index: int = -1
    for index in range(len(messages) - 1, -1, -1):
        message: Dict[str, Any] = messages[index]
        if message.get("role") == "assistant" and message.get("tool_calls"):
            # 向前搜素到第一个含Tool Call的AI消息
            assistant_index = index
            break
        if message.get("role") == "assistant":
            return []
    if assistant_index < 0:
        return []

    resolved_ids: set[str] = set()
    for message in messages[assistant_index + 1:]:
        if message.get("role") != "tool":
            continue
        content: str = str(message.get("content", ""))
        if content.startswith("[Hook message]:"):
            continue
        tool_call_id: Any = message.get("tool_call_id")
        if isinstance(tool_call_id, str):
            resolved_ids.add(tool_call_id)

    tool_calls: Any = messages[assistant_index].get("tool_calls", [])
    if not isinstance(tool_calls, list):
        raise ValueError("Checkpoint assistant tool_calls must be a list")
    pending: List[Dict[str, Any]] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            raise ValueError("Checkpoint tool call must be an object")
        tool_call_id: Any = tool_call.get("id")
        if not isinstance(tool_call_id, str):
            raise ValueError("Checkpoint tool call is missing an id")
        if tool_call_id not in resolved_ids:
            pending.append(tool_call)
    return pending
