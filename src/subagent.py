from typing import Any, Callable, Dict, List, Optional
import asyncio
import logging

from PyQt5.QtCore.QProcess import finished
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

from ai_config import client
from background import BACKGROUND_MANAGER
from checkpoint import build_subagent_checkpoint, pending_tool_calls, subagent_id_for
from compact import CompactState, agent_compact_states, try_compact
import config
from hook import HookEvent, HookPayload, HookResponse, hook_manager
from persistence import CheckpointRecord, PersistenceStore, get_persistence_store
from prompt.system_prompt import system_prompt_builder
from recovery import CONTINUE_MESSAGE, RecoveryType, backoff_delay, choose_recovery
from tool_execution import execute_tool_call, recover_tool_call
from tools import TOOLS, TOOL_HANDLERS


logger = logging.getLogger(__name__)

SUBAGENT_DEFAULT_MAX_TURNS: int = 20
CHILDREN_TOOLS: List[Dict[str, Any]] = TOOLS


class SubagentContext:
    messages: List[Dict[str, Any]]
    tools: List[Dict[str, Any]]
    handlers: Dict[str, Callable]
    max_turns: int
    turn_count: int

    def __init__(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        handlers: Dict[str, Callable],
        max_turns: int,
        turn_count: int = 0,
    ):
        self.messages = messages
        self.tools = tools
        self.handlers = handlers
        self.max_turns = max_turns
        self.turn_count = turn_count


def _summary(messages: List[Dict[str, Any]]) -> str:
    assistant_contents: List[str] = [
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "assistant" and message.get("content")
    ]
    return "".join(assistant_contents) or "No summary"


def _restore_context(
    record: CheckpointRecord,
    handlers: Dict[str, Callable],
) -> tuple[SubagentContext, CompactState]:
    if record.agent_type != "subagent":
        raise ValueError(f"Checkpoint {record.agent_id} is not a subagent checkpoint")
    context_payload: Any = record.payload.get("context")
    compact_payload: Any = record.payload.get("compact_state")
    if not isinstance(context_payload, dict) or not isinstance(compact_payload, dict):
        raise ValueError(f"Subagent checkpoint {record.agent_id} is incomplete")
    messages: Any = context_payload.get("messages")
    tools: Any = context_payload.get("tools")
    if not isinstance(messages, list) or not isinstance(tools, list):
        raise ValueError(f"Subagent checkpoint {record.agent_id} has invalid messages or tools")
    context: SubagentContext = SubagentContext(
        messages=messages,
        tools=tools,
        handlers=handlers,
        max_turns=int(context_payload.get("max_turns", SUBAGENT_DEFAULT_MAX_TURNS)),
        turn_count=int(context_payload.get("turn_count", 0)),
    )
    compact_state: CompactState = CompactState.model_validate(compact_payload)
    return context, compact_state


async def run_subagent_tool(
    prompt: str,
    agent_id: str,
    tool_call_id: str,
    tools: List[Dict[str, Any]] = CHILDREN_TOOLS,
    handlers: Dict[str, Callable] = TOOL_HANDLERS,
    max_turns: int = SUBAGENT_DEFAULT_MAX_TURNS,
) -> str:
    global stop_reason
    store: PersistenceStore = get_persistence_store()
    session_id: str = agent_id
    subagent_id: str = subagent_id_for(session_id, tool_call_id)
    checkpoint_record: Optional[CheckpointRecord] = store.load_checkpoint(
        session_id,
        subagent_id,
    )
    subagent: SubagentContext
    subagent_compact_state: CompactState

    if checkpoint_record is not None:
        subagent, subagent_compact_state = _restore_context(
            checkpoint_record,
            handlers,
        )
    else:
        init_messages: List[Dict[str, Any]] = [
            {"role": "system", "content": await system_prompt_builder.build()},
            {"role": "user", "content": prompt},
        ]
        subagent = SubagentContext(
            messages=init_messages,
            tools=tools,
            handlers=handlers,
            max_turns=max_turns,
        )
        subagent_compact_state = CompactState()
        start_response: HookResponse = await hook_manager.run_hooks(
            HookEvent(name="SessionStart", payload=HookPayload())
        )
        for hook_message in start_response.messages:
            subagent.messages.append(
                {
                    "role": "system",
                    "content": f"[Hook message]: {hook_message}",
                }
            )

    agent_compact_states[subagent_id] = subagent_compact_state

    def save_checkpoint() -> None:
        payload: Dict[str, Any] = build_subagent_checkpoint(
            subagent.messages,
            subagent.tools,
            subagent.max_turns,
            subagent.turn_count,
            subagent_compact_state,
        )
        store.save_checkpoint(session_id, subagent_id, "subagent", payload)

    save_checkpoint()
    pending_calls: List[Dict[str, Any]] = pending_tool_calls(subagent.messages)
    for pending_call in pending_calls:
        await recover_tool_call(
            subagent.messages,
            pending_call,
            subagent.handlers,
            subagent_id,
            save_checkpoint,
        )
    if pending_calls:
        background_notifications: str = BACKGROUND_MANAGER.get_background_task_notification(
            subagent_id
        )
        if background_notifications.strip():
            subagent.messages.append(
                {"role": "user", "content": background_notifications}
            )
        save_checkpoint()

    last_message: Optional[Dict[str, Any]] = (
        subagent.messages[-1] if subagent.messages else None
    )
    if last_message is not None and last_message.get("role") == "assistant":
        agent_compact_states.pop(subagent_id, None)
        return _summary(subagent.messages)

    retry_count: int = 0
    while subagent.turn_count < subagent.max_turns:
        subagent.messages = await try_compact(
            subagent.messages,
            subagent_compact_state,
        )
        save_checkpoint()
        try:
            response: ChatCompletion = await client.chat.completions.create(
                model=config.MODEL_ID,
                messages=subagent.messages,  # type: ignore
                tools=subagent.tools,  # type: ignore
                max_tokens=config.MAX_TOKENS,
            )
            stop_reason: Optional[str] = response.choices[0].finish_reason
            decision: RecoveryType = choose_recovery(stop_reason, None)
        except Exception as exc:
            logger.error("Error during subagent chat completion: %s", str(exc))
            error_text: str = str(exc)
            decision = choose_recovery(None, error_text)

        match decision.type:
            case "continue":
                retry_count = 0
                subagent.messages.append(
                    {"role": "user", "content": CONTINUE_MESSAGE}
                )
                save_checkpoint()
                continue
            case "compact":
                retry_count = 0
                subagent.messages = await try_compact(
                    subagent.messages,
                    subagent_compact_state,
                )
                save_checkpoint()
                continue
            case "backoff":
                await asyncio.sleep(backoff_delay(retry_count))
                retry_count += 1
                continue
            case "fail":
                retry_count = 0
                save_checkpoint()
                break

        retry_count = 0
        assistant_message: ChatCompletionMessage = response.choices[0].message
        valid_tool_calls: List[Dict[str, Any]] = []
        if assistant_message.tool_calls:
            for index, tool_call in enumerate(assistant_message.tool_calls):
                if not isinstance(tool_call, ChatCompletionMessageToolCall):
                    logger.warning(
                        "Skipping unsupported custom subagent tool call: %s",
                        tool_call,
                    )
                    continue
                tool_call_id_value: str = (
                    tool_call.id or f"call_{subagent.turn_count + 1}_{index + 1}"
                )
                valid_tool_calls.append(
                    {
                        "id": tool_call_id_value,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                )

        assistant_payload: Dict[str, Any] = {
            "role": "assistant",
            "content": assistant_message.content or "",
        }
        if valid_tool_calls:
            assistant_payload["tool_calls"] = valid_tool_calls
        subagent.messages.append(assistant_payload)
        subagent.turn_count += 1
        save_checkpoint()

        if not valid_tool_calls:
            break
        for tool_call_payload in valid_tool_calls:
            await execute_tool_call(
                subagent.messages,
                tool_call_payload,
                subagent.handlers,
                subagent_id,
                save_checkpoint,
            )

        # subagent don't run background task
        # key: if stop reason == "stop" means agent finished all tasks, or tool_call means need tool call
        if stop_reason == "stop":
            break

    agent_compact_states.pop(subagent_id, None)
    return _summary(subagent.messages)
