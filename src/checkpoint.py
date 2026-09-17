from typing import Any, Dict, List
import json

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


def resolved_foreground_subagent_call_ids(
    messages: List[Dict[str, Any]],
) -> List[str]:
    """Find completed foreground subagent calls whose child checkpoints are safe to delete.

    A subagent result is durable only when the parent conversation contains a terminal
    tool message with the same ``tool_call_id``. Hook messages do not count as terminal
    results. Background subagent calls are intentionally excluded because their parent
    tool message is only a scheduling acknowledgement; their child checkpoint is deleted
    by the background completion callback after the real result is persisted.
    """
    # Collect tool calls for which the parent conversation already has a final result.
    terminal_ids: set[str] = set() # 已经有结果的tool，包含所有的tool结果
    for message in messages:
        if message.get("role") != "tool":
            continue
        content: str = str(message.get("content", ""))
        tool_call_id: Any = message.get("tool_call_id")
        if not content.startswith("[Hook message]:") and isinstance(tool_call_id, str):
            terminal_ids.add(tool_call_id)

    resolved_ids: List[str] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        tool_calls: Any = message.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            raise ValueError("Checkpoint assistant tool_calls must be a list")
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                raise ValueError("Checkpoint tool call must be an object")
            tool_call_id: Any = tool_call.get("id")
            function: Any = tool_call.get("function")
            if not isinstance(tool_call_id, str) or not isinstance(function, dict):
                raise ValueError("Checkpoint tool call is incomplete")
            if function.get("name") != "subagent" or tool_call_id not in terminal_ids:
                continue # 不是subagent或者subagent还没有结果的都排除
            arguments: Any = function.get("arguments", "{}")
            if not isinstance(arguments, str):
                raise ValueError("Checkpoint tool call arguments must be JSON text")
            tool_input: Any = json.loads(arguments)
            if not isinstance(tool_input, dict):
                raise ValueError("Checkpoint tool call arguments must be an object")
            # Only foreground results prove that the subagent invocation itself finished.
            if tool_input.get("run_in_background") is not True:
                resolved_ids.append(tool_call_id)
    # 只有已经解决的subagent才会删掉checkpoint
    return resolved_ids
