from typing import Any, Callable, Dict, List, Optional
import json
import logging

from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

from background import BACKGROUND_MANAGER, RuntimeTaskRecord
from hook import HookEvent, HookPayload, HookResponse, hook_manager
from tools import run_tool


logger = logging.getLogger(__name__)

CheckpointCallback = Callable[[], None]
ToolCompletionCallback = Callable[[str, Dict[str, Any]], None]


async def execute_tool_call(
    messages: List[Dict[str, Any]],
    tool_call_payload: Dict[str, Any],
    handlers: Dict[str, Callable],
    agent_id: str,
    checkpoint_callback: Optional[CheckpointCallback] = None,
    completion_callback: Optional[ToolCompletionCallback] = None, # 目前只有一种，就算清除完成的subagent的checkpoint
) -> str:
    """ Execute tool call, and save checkpoint"""
    tool_call: ChatCompletionMessageToolCall = (
        ChatCompletionMessageToolCall.model_validate(tool_call_payload)
    )
    tool_name: str = tool_call.function.name
    tool_input: Dict[str, Any] = json.loads(tool_call.function.arguments)
    completion_input: Dict[str, Any] = {
        **tool_input,
        "tool_call_id": tool_call.id,
    }

    pre_response: HookResponse = await hook_manager.run_hooks(
        HookEvent(
            name="PreToolCall",
            payload=HookPayload(tool_name=tool_name, tool_input=tool_input),
        )
    )
    for hook_message in pre_response.messages:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": f"[Hook message]: {hook_message}",
            }
        )
    if pre_response.blocked:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": f"[Tool call blocked by hook]: {pre_response.blocked_reason}",
            }
        )
        # 就算被阻止也持久化工具阻止的结果
        if checkpoint_callback is not None:
            checkpoint_callback()
        if completion_callback is not None:
            completion_callback(tool_name, completion_input)
        return tool_name

    logger.debug("Executing tool call: %s", tool_name)
    output: str = await run_tool(tool_call, handlers, agent_id=agent_id)
    post_response: HookResponse = await hook_manager.run_hooks(
        HookEvent(
            name="PostToolCall",
            payload=HookPayload(tool_name=tool_name, tool_input=tool_input),
        )
    )
    for hook_message in post_response.messages:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": f"[Hook message]: {hook_message}",
            }
        )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": output,
        }
    )
    # 工具正常执行后就持久化，避免重复调用
    if checkpoint_callback is not None:
        checkpoint_callback()
    if completion_callback is not None:
        completion_callback(tool_name, completion_input)
    return tool_name


async def recover_tool_call(
    messages: List[Dict[str, Any]],
    tool_call_payload: Dict[str, Any],
    handlers: Dict[str, Callable],
    agent_id: str,
    checkpoint_callback: Optional[CheckpointCallback] = None,
    completion_callback: Optional[ToolCompletionCallback] = None,
) -> str:
    """
    Reexecute unfinished tool calls. No check to tool call is unfinished.

    Args:
        messages: The state messages, tool result will append to it
        tool_call_payload: The unfinished tool call payload
        handlers: all tool handlers
        agent_id: The agent id, who call this tool
        checkpoint_callback: The function which save checkpoint
        completion_callback: The function which save completion, now only clean subagent call
    """
    tool_call: ChatCompletionMessageToolCall = (
        ChatCompletionMessageToolCall.model_validate(tool_call_payload)
    )
    tool_input: Dict[str, Any] = json.loads(tool_call.function.arguments)
    if tool_input.get("run_in_background") is True:
        existing_task: Optional[RuntimeTaskRecord] = next(
            (
                task
                for task in BACKGROUND_MANAGER.snapshot()
                if task.agent_id == agent_id
                and task.tool_input.get("tool_call_id") == tool_call.id
            ),
            None,
        )
        if existing_task is not None:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": (
                        f"[Background task {existing_task.id} restored]\n"
                        f"Command: {tool_input.get('command', '')}.\n"
                        "Result will be available in the <task_notifications> section."
                    ),
                }
            )
            if checkpoint_callback is not None:
                checkpoint_callback()
            return tool_call.function.name
    return await execute_tool_call(
        messages,
        tool_call_payload,
        handlers,
        agent_id,
        checkpoint_callback,
        completion_callback,
    )
