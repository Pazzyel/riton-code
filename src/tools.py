from pathlib import Path
from typing import Optional, Dict, Callable, List, Any
import aiofiles
import asyncio
from asyncio.subprocess import Process
import json
import os
import logging
import shlex
import inspect

from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

from todo import todo_manager
from skill import load_skill, get_skill_dir
from directory import WORKDIR
from compact import track_recent_files, agent_compact_states
from permission.permission import permission_manager, PermissionResult

logger = logging.getLogger(__name__)

TOOL_HANDLERS: Dict[str, Callable] = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(path=kw["path"], limit=kw.get("limit"), agent_id=kw["agent_id"]),
    "write_file": lambda **kw: run_write(path=kw["path"], content=kw["content"], agent_id=kw["agent_id"]),
    "edit_file":  lambda **kw: run_edit(path=kw["path"], old_text=kw["old_text"], new_text=kw["new_text"], agent_id=kw["agent_id"]),
    "todo":       lambda **kw: todo_manager.update(kw["todos"]),
    "load_skill": lambda **kw: load_skill(kw["name"]),
    "get_skill_dir": lambda **kw: get_skill_dir(kw["name"])
}

TOOLS = [
    {
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
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file in the current workspace. Provide the relative path to the file. Optionally provide a line limit to avoid reading very large files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The relative path to the file to read."},
                    "limit": {"type": "integer", "description": "Optional line limit for reading the file."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write to a file in the current workspace. Provide the relative path and content to write.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The relative path to the file to write."},
                    "content": {"type": "string", "description": "The content to write to the file."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Edit a file in the current workspace. Provide the relative path, the exact text to find, and the new text to replace it with. "
                "This is for making targeted edits to files. If you want to replace all content, use write_file instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The relative path to the file to edit."},
                    "old_text": {"type": "string", "description": "The exact text in the file to find and replace."},
                    "new_text": {"type": "string", "description": "The new text to replace the old text with."},
                },
                # All fields are required for edit_file
                # since we need to know what text to replace and what to replace it with
                # even if new_text is just an empty string for deletion
                # or old_text is just an empty string for prepending/appending
                #
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "todo",
            "description": (
                "Manage a todo list. Provide a list of todo items with their content and status (pending, in_progress, completed). "
                "The agent can use this to keep track of its plan and progress. The tool will return a rendered string representation of the todo list that the agent can include in its messages."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "description": "The list of todo items to update. Each item should have content and status.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string", "description": "The content of the todo item."},
                                "status": {
                                    "type": "string",
                                    "description": "The status of the todo item. Should be one of 'pending', 'in_progress', or 'completed'.",
                                },
                                "activeForm": {
                                    "type": "string",
                                    "description": "Optional present-continuous label.",
                                },
                            },
                            "required": ["content", "status"]
                        }
                    }
                },
                "required": ["todos"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "load_skill",
            "description": (
                "Load a skill by name from the skills directory. This will read the corresponding SKILL.md file and return its content. "
                "The agent can then include this content in its messages to use the information about the skill, such as its description and usage instructions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The name of the skill to load."},
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_skill_dir",
            "description": "Get the posix path to a specific skill's directory. This can be used to inspect all files in the skill",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The name of the skill for which to get the directory path."},
                },
                "required": ["name"]
            }
        }
    }
]

SUBAGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "subagent",
            "description": "Run a subagent in a clean context and return a summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"}
                },
                "required": ["prompt"]
            }
        }
    }
]

async def run_tool(tool_call: ChatCompletionMessageToolCall, tool_handlers: Dict[str, Callable], agent_id: str) -> str:
    """Run a tool call using the provided handlers and return the output."""
    # TODO: Add support for more tools'
    arguments: Dict[str, str] = json.loads(tool_call.function.arguments)
    logger.debug("Dispatching tool '%s' with args: %s", tool_call.function.name, arguments)

    # Check permissions before running the tool
    permission_result: PermissionResult = permission_manager.check(tool_call.function.name, arguments)
    if permission_result.behavior == "deny":
        logger.info(f"Permission denied for tool call '{tool_call.function.name}' with args {arguments}: {permission_result.reason}")
        return f"Permission denied: {permission_result.reason}"
    if permission_result.behavior == "ask":
        if not permission_manager.ask_user(tool_call.function.name, arguments):
            logger.info(f"User denied permission for tool call '{tool_call.function.name}' with args {arguments}")
            return "Permission denied by user."

    handler: Optional[Callable] = tool_handlers.get(tool_call.function.name)
    # For subagent tool calls, we want to track recent files accessed by the subagent for better compaction in the main agent
    all_args: Dict[str, Any] = {**arguments, "agent_id": agent_id}
    if handler:
        try:
            # Handlers may be sync wrappers (e.g. lambdas) that return a coroutine.
            # Always inspect the call result and await when needed.
            result = handler(**all_args)
            # You couldn'd use inspect.iscoroutine here
            # Because handlers use lambda warppers, and the lambda is not async
            # But it still returns a coroutine when it calls the async function inside
            if inspect.isawaitable(result):
                return await result
            return result
        except Exception as e:
            logger.error(f"Error while running tool {tool_call.function.name}: {str(e)}")
            return f"Error: Exception while running tool: {str(e)}"
    return "Error: Unknown tool call."


async def run_bash(command: str) -> str:
    """
    Run a bash command and return its output.

    The function is for the bash tool
    """
    logger.debug(f"Running bash command: {command}")
    dangerous_commands: List[str] = ["rm -rf/", "dd", "mkfs", "shutdown", "reboot"]
    tokens = shlex.split(command)
    if any(dc in tokens for dc in dangerous_commands):
        return "Error: Rejected command, it is too dangerous to run."
    if "cat" in command:
        return "Error: 'cat' command is not allowed. Use the read_file tool instead to read file contents."
    try:
        result: Process = await asyncio.create_subprocess_shell(
            command, 
            shell=True, 
            cwd=os.getcwd(), 
            stdout=asyncio.subprocess.PIPE, 
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=30)  # Set a timeout for command execution
    except asyncio.TimeoutError as e:
        logger.error(f"Error while running bash command '{command}': {str(e)}")
        return f"Error: Command timed out: {str(e)}"
    except (FileNotFoundError, OSError) as e:
        logger.error(f"Error while running bash command '{command}': {str(e)}")
        return f"Error: {e}"
    
    
    output: str = (stdout.decode() + stderr.decode()).strip()
    # Limit output to 50,000 characters to prevent overwhelming the agent
    return output[:50000] if output else "Command executed successfully with no output."


def safe_path(p: str, agent_id: str) -> Path:
    """Resolve a path and ensure it is within the workspace."""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    # Using file operation tool will trigger the tracking of recent files for compaction purposes
    if agent_id not in agent_compact_states:
        raise ValueError(f"Agent ID {agent_id} not found in compact states.")
    track_recent_files(agent_compact_states[agent_id], p)
    return path

async def run_read(path: str, limit: Optional[int] = None, agent_id: str = "") -> str:
    """Read a file safely, ensuring it is within the workspace and optionally limiting the number of lines."""
    logger.debug(f"Reading file at path: {path} with limit: {limit}")
    file_path: Path = safe_path(path, agent_id)
    async with aiofiles.open(file_path, mode='r') as f:
        text: str = await f.read()
    lines = text.splitlines()
    if limit and limit < len(lines):
        lines = lines[:limit]
    return "\n".join(lines)[:50000]

async def run_write(path: str, content: str, agent_id: str = "") -> str:
    """Write to a file safely, ensuring it is within the workspace."""
    logger.debug(f"Writing to file at path: {path}")
    file_path: Path = safe_path(path, agent_id)
    async with aiofiles.open(file_path, mode='w') as f:
        await f.write(content)
    return "File written successfully."

async def run_edit(path: str, old_text: str, new_text: str, agent_id: str = "") -> str:
    """Edit a file safely, ensuring it is within the workspace."""
    logger.debug(f"Editing file at path: {path}")
    file_path: Path = safe_path(path, agent_id)
    async with aiofiles.open(file_path, mode='r') as f:
        text: str = await f.read()
    if old_text not in text:
        return "Error: Old text not found in file."
    updated_text = text.replace(old_text, new_text)
    async with aiofiles.open(file_path, mode='w') as f:
        await f.write(updated_text)
    return "File edited successfully."