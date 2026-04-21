from typing import List, Dict, Optional, Any
import logging
import json

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

import config as config
from tools import run_tool, TOOLS, TOOL_HANDLERS, SUBAGENT_TOOLS
from subagent import run_subagent
from todo import todo_manager
from ai_config import client
from prompt.system_prompt import system_prompt_builder
from compact import CompactState, try_compact
from hook import HookEvent, HookResponse, HookPayload, hook_manager

logger = logging.getLogger(__name__)

MAX_TURNS: int = 20  # Maximum number of turns in the agent loop before stopping

PARENT_TOOL_HANDLERS = TOOL_HANDLERS.copy()
PARENT_TOOL_HANDLERS["subagent"] = lambda **kw: run_subagent(kw["prompt"])
PARENT_TOOLS = TOOLS + SUBAGENT_TOOLS

class LoopState:
    messages:           List[Dict[str, Any]]    # The list of messages in the conversation history
    turn_count:         int                     # The number of turns taken in the loop
    transition_reason:  Optional[str]           # The reason for transitioning to the next turn, if applicable

    def __init__(self, messages: List[Dict[str, str]], turn_count: int, transition_reason: Optional[str]):
        self.messages = messages
        self.turn_count = turn_count
        self.transition_reason = transition_reason

async def agent_loop(state: LoopState, compact_state: CompactState, agent_id: str) -> None:
    """
    Run the agent loop until completion.

    The loop will continue until the agent decides to stop by returning False from run_one_loop.
    """

    logger.debug("Starting agent loop")
    state.messages = await try_compact(state.messages, compact_state)
    while await run_one_loop(state, agent_id) and state.turn_count < MAX_TURNS:
        state.messages = await try_compact(state.messages, compact_state)
    
    logger.debug("Agent loop finished after %d turns", state.turn_count)

async def run_one_loop(state: LoopState, agent_id: str) -> bool:
    """
    Run one loop of the agent's reasoning and acting process.

    Returns True if the loop should continue, or False if it should stop.
    """
    logger.debug("Running loop turn %d", state.turn_count + 1)
    response: ChatCompletion = await client.chat.completions.create(
        model=config.MODEL_ID,
        messages=[{"role": "system", "content": await system_prompt_builder.build()}] + state.messages, # type: ignore
        tools=PARENT_TOOLS, # type: ignore
        max_tokens=config.MAX_TOKENS,
    )
    assistant_message: ChatCompletionMessage = response.choices[0].message
    valid_tool_calls: List[Dict[str, Any]] = []
    # if a assistant message includes tool calls
    # It's content is empty
    if assistant_message.tool_calls:
        for i, tool_call in enumerate(assistant_message.tool_calls):
            if not isinstance(tool_call, ChatCompletionMessageToolCall):
                logger.warning("Received tool call that is not of type ChatCompletionMessageToolCall, now custom tool is not supported ,skipping: %s", tool_call)
                continue
            tool_name = getattr(getattr(tool_call, "function", None), "name", "unknown")
            tool_args = getattr(getattr(tool_call, "function", None), "arguments", "{}")
            tool_call_id = tool_call.id if tool_call.id else f"call_{state.turn_count + 1}_{i + 1}"
            valid_tool_calls.append({
                "tool_call": tool_call,
                "name": tool_name,
                "id": tool_call_id,
                "arguments": tool_args,
            })

    assistant_payload: Dict[str, Any] = {
        "role": "assistant",
        "content": assistant_message.content if assistant_message.content else "",
    }
    # But it's tool_calls is not empty, and must be saved in the message for later tool execution
    if valid_tool_calls:
        assistant_payload["tool_calls"] = [
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": call["arguments"],
                },
            }
            for call in valid_tool_calls
        ]
    state.messages.append(assistant_payload)

    if len(valid_tool_calls) == 0:
        logger.debug("No tool calls requested by model, stopping loop")
        state.transition_reason = None
        return False
    
    used_todo: bool = False
    for call in valid_tool_calls:
        tool_call = call["tool_call"]
        # Skip custom tool
        if not isinstance(tool_call, ChatCompletionMessageToolCall):
            logger.warning("Received tool call that is not of type ChatCompletionMessageToolCall, skipping: %s", tool_call)
            continue
        tool_name = call["name"]
        if tool_name == "todo":
            used_todo = True
        
        # add PreToolCall hook here
        pre_response: HookResponse = await hook_manager.run_hooks(HookEvent(name="PreToolCall", payload=HookPayload(tool_name=tool_name, tool_input=json.loads(tool_call.function.arguments))))
        for msg in pre_response.messages:
            state.messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": f"[Hook message]: {msg}",
            })
        if pre_response.blocked:
            state.messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": f"[Tool call blocked by hook]: {pre_response.blocked_reason}",
            })
            continue

        logger.debug("Executing tool call: %s", tool_name)
        output: str = await run_tool(tool_call, PARENT_TOOL_HANDLERS, agent_id=agent_id)

        # add PostToolCall hook here
        post_response: HookResponse = await hook_manager.run_hooks(HookEvent(name="PostToolCall", payload=HookPayload(tool_name=tool_name, tool_input=json.loads(tool_call.function.arguments))))
        for msg in post_response.messages:
            state.messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": f"[Hook message]: {msg}",
            })
        
        # Real tool output should after post hook messages
        state.messages.append({
            "role": "tool",
            "tool_call_id": call["id"],
            "content": output,
        })


    # Todo reminder should in the back of tool message
    if not used_todo:
        # If the tool call didn't include an update to the todo list, increment the reminder counter
        todo_manager.note_round_without_update()
        remainder: Optional[str] = todo_manager.reminder()
        if remainder:
            state.messages.append({
                "role": "user",
                "content": remainder,
            })

    state.turn_count += 1
    state.transition_reason = "tool_call"
    logger.debug("Turn %d finished with transition_reason=%s", state.turn_count, state.transition_reason)
    return True
