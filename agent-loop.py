from typing import List, Dict, Optional
import config
import os
import subprocess
import json
from subprocess import CompletedProcess, CalledProcessError

from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage

client: OpenAI = OpenAI(
    base_url=config.BASE_URL,
    api_key=config.API_KEY,
)

SYSTEM = (
    f"You are a coding agent at {os.getcwd()}. "
    "Use bash to inspect and change the workspace. Act first, then report clearly."
)

class LoopState:
    messages:           List[Dict[str, str]]    # The list of messages in the conversation history
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

    while run_one_loop(state):
        pass

def run_one_loop(state: LoopState) -> bool:
    """
    Run one loop of the agent's reasoning and acting process.

    Returns True if the loop should continue, or False if it should stop.
    """
    response: ChatCompletion = client.chat.completions.create(
        model=config.MODEL_ID,
        messages=[{"role": "system", "content": SYSTEM}] + state.messages, # type: ignore
        tools=TOOLS, # type: ignore
        max_tokens=1000,
    )
    assistant_message: ChatCompletionMessage = response.choices[0].message
    state.messages.append({
        "role": "assistant",
        "content": assistant_message.content if assistant_message.content else "",
    })
    
    if (not assistant_message.tool_calls or len(assistant_message.tool_calls) == 0):
        state.transition_reason = None
        return False
    
    results: List[Dict[str, str]] = []
    for tool_call in assistant_message.tool_calls:
        output: str = run_tool(tool_call)
        results.append({
            "type": tool_call.type,
            "tool_call_id": tool_call.id,
            "content": output,
        }) 

    state.messages.append({
        "role": "tool",
        "content": str(results),
    })
    state.turn_count += 1
    state.transition_reason = "tool_call"
    return True

def run_tool(tool_call) -> str:
    # TODO: Add support for more tools'
    arguments: Dict[str, str] = json.loads(tool_call.function.arguments)
    if tool_call.function.name == "bash":
        return run_bash(arguments["command"])
    return "Error: Unknown tool call."

TOOLS = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command in the current workspace.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The shell command to run."}},
            "required": ["command"],
        },
    },
}]

def run_bash(command: str) -> str:
    """
    Run a bash command and return its output.

    The function is for the bash tool
    """
    dangerous_commands: List[str] = ["rm -rf/", "dd", "mkfs", "shutdown", "reboot"]
    if any(dc in command for dc in dangerous_commands):
        return "Error: Rejected command, it is too dangerous to run."
    try:
        result: CompletedProcess = subprocess.run(
            command, 
            shell=True, 
            cwd=os.getcwd(), 
            check=True, 
            text=True,
            timeout=30,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
    except CalledProcessError as e:
        return f"Error: Command failed with exit code {e.returncode}: {e.stderr.decode()}"
    except TimeoutError as e:
        return f"Error: Command timed out: {str(e)}"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"
    
    output: str = (result.stdout + result.stderr).strip()
    # Limit output to 50,000 characters to prevent overwhelming the agent
    return output[:50000] if output else "Command executed successfully with no output."

if __name__ == "__main__":
    initial_state: LoopState = LoopState(
        messages=[],
        turn_count=0,
        transition_reason=None,
    )
    initial_state.messages.append({
        "role": "user",
        "content": "What time is it now?\n",
    })
    agent_loop(initial_state)
    print(initial_state.messages[-1]["content"])
    print(len(initial_state.messages))