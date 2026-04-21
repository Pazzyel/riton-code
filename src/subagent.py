from typing import List, Dict, Any, Callable
import logging
import uuid
import json

from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

from tools import TOOLS, TOOL_HANDLERS, run_tool
from ai_config import client
import config
from compact import CompactState, try_compact, agent_compact_states
from hook import HookEvent, HookPayload, HookResponse, hook_manager
from prompt.system_prompt import system_prompt_builder

logger = logging.getLogger(__name__)

SUBAGENT_DEFAULT_MAX_TURNS: int = 20  # Default maximum number of turns for a subagent loop

CHILDREN_TOOLS = TOOLS

class SubagentContext:
    """Context object passed to subagents, containing conversation history, available tools, and other relevant information."""
    messages: List[Dict[str, Any]]    # The list of messages in the conversation history
    tools: List[Dict[str, Any]]       # The list of tools available to the subagent
    handlers: Dict[str, Callable]     # A mapping of tool names to their handler functions
    max_turns: int                    # The number of maximum turns in the loop

    def __init__(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], handlers: Dict[str, Callable], max_turns: int):
        self.messages = messages
        self.tools = tools
        self.handlers = handlers
        self.max_turns = max_turns

async def run_subagent(prompt: str, 
                 tools: List[Dict[str, Any]] = CHILDREN_TOOLS, 
                 handlers: Dict[str, Callable] = TOOL_HANDLERS, 
                 max_turns: int = SUBAGENT_DEFAULT_MAX_TURNS) -> str:
    init_message: List[Dict[str, Any]] = [
        {"role": "system", "content": await system_prompt_builder.build()},
        {"role": "user", "content": prompt}
    ]
    subagent: SubagentContext = SubagentContext(
        messages=init_message,
        tools=tools,
        handlers=handlers,
        max_turns=max_turns,
    )
    subagent_compact_state: CompactState = CompactState()
    subagent_id: str = f"subagent_{uuid.uuid4().hex}"
    agent_compact_states[subagent_id] = subagent_compact_state

    start_response: HookResponse = await hook_manager.run_hooks(HookEvent(name="SessionStart", payload=HookPayload()))
    for msg in start_response.messages:
        subagent.messages.append({
            "role": "system",
            "content": f"[Hook message]: {msg}",
        })

    for _ in range(max_turns):
        # Compact the subagent's conversation history first
        subagent.messages = await try_compact(subagent.messages, subagent_compact_state)
        
        response = await client.chat.completions.create(
            model=config.MODEL_ID,
            messages=subagent.messages, # type: ignore
            tools=subagent.tools, # type: ignore
            max_tokens=config.MAX_TOKENS,
        )
        assistant_message: ChatCompletionMessage = response.choices[0].message
        valid_tool_calls: List[Dict[str, Any]] = []
        if assistant_message.tool_calls:
            for i, tool_call in enumerate(assistant_message.tool_calls):
                if not isinstance(tool_call, ChatCompletionMessageToolCall):
                    logger.warning("Received tool call that is not of type ChatCompletionMessageToolCall, now custom tool is not supported ,skipping: %s", tool_call)
                    continue
                tool_name = getattr(getattr(tool_call, "function", None), "name", "unknown")
                tool_args = getattr(getattr(tool_call, "function", None), "arguments", "{}")
                tool_call_id = tool_call.id if tool_call.id else f"call_{i + 1}"
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
        subagent.messages.append(assistant_payload)

        if len(valid_tool_calls) == 0:
            break

        for call in valid_tool_calls:
            tool_call = call["tool_call"]
            tool_name = call["name"]

                    
            # add PreToolCall hook here
            pre_response: HookResponse = await hook_manager.run_hooks(HookEvent(name="PreToolCall", payload=HookPayload(tool_name=tool_name, tool_input=json.loads(tool_call.function.arguments))))
            for msg in pre_response.messages:
                subagent.messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": f"[Hook message]: {msg}",
                })
            if pre_response.blocked:
                subagent.messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": f"[Tool call blocked by hook]: {pre_response.blocked_reason}",
                })
                continue

            logger.debug("SubAgent Executing tool call: %s", tool_name)
            output: str = await run_tool(tool_call, subagent.handlers, subagent_id)

            # add PostToolCall hook here
            post_response: HookResponse = await hook_manager.run_hooks(HookEvent(name="PostToolCall", payload=HookPayload(tool_name=tool_name, tool_input=json.loads(tool_call.function.arguments))))
            for msg in post_response.messages:
                subagent.messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": f"[Hook message]: {msg}",
                })

            # Real tool output should after post hook messages
            subagent.messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": output,
            })

    # TODO Now the subagent summarizes is only using the contact of the assistant messages
    # May need use summary model to summarize the conversation
    agent_compact_states.pop(subagent_id, None)  # Clean up compact state for this subagent
    return "".join([msg["content"] for msg in subagent.messages if msg["role"] == "assistant"]) or "No summary"