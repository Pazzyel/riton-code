from typing import List, Dict, Optional, Any
import json
import argparse
import logging

from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage

import config as config
from tools import run_tool, TOOLS, TOOL_HANDLERS, SUBAGENT_TOOLS
from subagent import run_subagent
from todo import todo_manager
from ai_config import client
from skill import SKILL_REGISTRY
from directory import WORKDIR

logger = logging.getLogger(__name__)

MAX_TURNS: int = 20  # Maximum number of turns in the agent loop before stopping


PARENT_TOOL_HANDLERS = TOOL_HANDLERS.copy()
PARENT_TOOL_HANDLERS["subagent"] = lambda **kw: run_subagent(kw["prompt"])
PARENT_TOOLS = TOOLS + SUBAGENT_TOOLS

SYSTEM = (
    f"You are a coding agent at {str(WORKDIR)}. "
    "Use bash to inspect and change the workspace. Act first, then report clearly."
    f"""
    <available_skills>
    {SKILL_REGISTRY.describe_available()}
    </available_skills>
    """
)

class LoopState:
    messages:           List[Dict[str, Any]]    # The list of messages in the conversation history
    turn_count:         int                     # The number of turns taken in the loop
    transition_reason:  Optional[str]           # The reason for transitioning to the next turn, if applicable

    def __init__(self, messages: List[Dict[str, str]], turn_count: int, transition_reason: Optional[str]):
        self.messages = messages
        self.turn_count = turn_count
        self.transition_reason = transition_reason

def agent_loop(state: LoopState) -> None:
    """
    Run the agent loop until completion.

    The loop will continue until the agent decides to stop by returning False from run_one_loop.
    """

    logger.debug("Starting agent loop")
    while run_one_loop(state) and state.turn_count < MAX_TURNS:
        pass
    logger.debug("Agent loop finished after %d turns", state.turn_count)

def run_one_loop(state: LoopState) -> bool:
    """
    Run one loop of the agent's reasoning and acting process.

    Returns True if the loop should continue, or False if it should stop.
    """
    logger.debug("Running loop turn %d", state.turn_count + 1)
    response: ChatCompletion = client.chat.completions.create(
        model=config.MODEL_ID,
        messages=[{"role": "system", "content": SYSTEM}] + state.messages, # type: ignore
        tools=PARENT_TOOLS, # type: ignore
        max_tokens=1000,
    )
    assistant_message: ChatCompletionMessage = response.choices[0].message
    state.messages.append({
        "role": "assistant",
        "content": assistant_message.content if assistant_message.content else "",
        
    })

    if (not assistant_message.tool_calls or len(assistant_message.tool_calls) == 0):
        logger.debug("No tool calls requested by model, stopping loop")
        state.transition_reason = None
        return False
    
    results: List[Dict[str, str]] = []
    used_todo: bool = False
    for tool_call in assistant_message.tool_calls:
        tool_name = getattr(getattr(tool_call, "function", None), "name", "unknown")
        if tool_name == "todo":
            used_todo = True
        logger.debug("Executing tool call: %s", tool_name)
        output: str = run_tool(tool_call, PARENT_TOOL_HANDLERS)
        results.append({
            "type": tool_call.type,
            "tool_call_id": tool_call.id,
            "content": output,
        }) 

    state.messages.append({
        "role": "tool",
        "name": tool_name,
        "content": json.dumps(results),
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the coding agent loop.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging output")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        force=True,
    )

    logger.debug("Debug logging enabled")
    state: LoopState = LoopState(
        messages=[],
        turn_count=0,
        transition_reason=None,
    )
    while True:
        try:
            query: str = input(">> ")
        except (EOFError, KeyboardInterrupt):
            break

        if query.strip().lower() == "exit":
            break

        state.messages.append({
            "role": "user",
            "content": query,
        })
        agent_loop(state)
        print(state.messages[-1]["content"])