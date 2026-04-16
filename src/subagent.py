from typing import List, Dict, Any, Callable
import json
import logging
import uuid

from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

from tools import TOOLS, TOOL_HANDLERS, run_tool
from ai_config import client
import config
from compact import CompactState, try_compact, agent_compact_states

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

def run_subagent(prompt: str, 
                 tools: List[Dict[str, Any]] = CHILDREN_TOOLS, 
                 handlers: Dict[str, Callable] = TOOL_HANDLERS, 
                 max_turns: int = SUBAGENT_DEFAULT_MAX_TURNS) -> str:
    init_message: Dict[str, Any] = {"role": "user", "content": prompt}
    subagent: SubagentContext = SubagentContext(
        messages=[init_message],
        tools=tools,
        handlers=handlers,
        max_turns=max_turns,
    )
    subagent_compact_state: CompactState = CompactState()
    subagent_id: str = f"subagent_{uuid.uuid4().hex}"
    agent_compact_states[subagent_id] = subagent_compact_state
    for _ in range(max_turns):
        # Compact the subagent's conversation history first
        subagent.messages = try_compact(subagent.messages, subagent_compact_state)
        
        response = client.chat.completions.create(
            model=config.MODEL_ID,
            messages=subagent.messages, # type: ignore
            tools=subagent.tools, # type: ignore
            max_tokens=config.MAX_TOKENS,
        )
        assistant_message: ChatCompletionMessage = response.choices[0].message
        subagent.messages.append({"role": "assistant", "content": assistant_message.content})
        if (not assistant_message.tool_calls or len(assistant_message.tool_calls) == 0):
            break
    
        results: List[Dict[str, str]] = []

        for tool_call in assistant_message.tool_calls:
            if not isinstance(tool_call, ChatCompletionMessageToolCall):
                logger.warning("Received tool call that is not of type ChatCompletionMessageToolCall, now custom tool is not supported ,skipping: %s", tool_call)
                continue
            tool_name = getattr(getattr(tool_call, "function", None), "name", "unknown")
            logger.debug("SubAgent Executing tool call: %s", tool_name)
            output: str = run_tool(tool_call, subagent.handlers, subagent_id)
            results.append({
                "type": tool_call.type,
                "tool_call_id": tool_call.id,
                "content": output,
            }) 

        subagent.messages.append({
            "role": "tool",
            "name": tool_name,
            "content": json.dumps(results),
        })

    # TODO Now the subagent summarizes is only using the contact of the assistant messages
    # May need use summary model to summarize the conversation
    agent_compact_states.pop(subagent_id, None)  # Clean up compact state for this subagent
    return "".join([msg["content"] for msg in subagent.messages if msg["role"] == "assistant"]) or "No summary"