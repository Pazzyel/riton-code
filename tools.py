from pathlib import Path
from typing import Optional, Dict, Callable, List
import subprocess
from subprocess import CompletedProcess, CalledProcessError, TimeoutExpired
import json
import os
import logging
import shlex

from todo import todo_manager

logger = logging.getLogger(__name__)

WORKDIR = Path.cwd()

TOOL_HANDLERS: Dict[str, Callable] = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"], kw.get("limit")),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"],
                                        kw["new_text"]),
    "todo":       lambda **kw: todo_manager.update(kw["todos"]),
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

    }
]

def run_tool(tool_call) -> str:
    # TODO: Add support for more tools'
    arguments: Dict[str, str] = json.loads(tool_call.function.arguments)
    logger.debug("Dispatching tool '%s' with args: %s", tool_call.function.name, arguments)
    handler: Optional[Callable] = TOOL_HANDLERS.get(tool_call.function.name)
    if handler:
        try:
            return handler(**arguments)
        except Exception as e:
            logger.error(f"Error while running tool {tool_call.function.name}: {str(e)}")
            return f"Error: Exception while running tool: {str(e)}"
    return "Error: Unknown tool call."


def run_bash(command: str) -> str:
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
        logger.error(f"Error while running bash command '{command}': {str(e)}")
        return f"Error: Command failed with exit code {e.returncode}: {e.stderr}"
    except TimeoutExpired as e:
        logger.error(f"Error while running bash command '{command}': {str(e)}")
        return f"Error: Command timed out: {str(e)}"
    except (FileNotFoundError, OSError) as e:
        logger.error(f"Error while running bash command '{command}': {str(e)}")
        return f"Error: {e}"
    
    output: str = (result.stdout + result.stderr).strip()
    # Limit output to 50,000 characters to prevent overwhelming the agent
    return output[:50000] if output else "Command executed successfully with no output."


def safe_path(p: str) -> Path:
    """Resolve a path and ensure it is within the workspace."""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_read(path: str, limit: Optional[int] = None) -> str:
    """Read a file safely, ensuring it is within the workspace and optionally limiting the number of lines."""
    logger.debug(f"Reading file at path: {path} with limit: {limit}")
    text = safe_path(path).read_text()
    lines = text.splitlines()
    if limit and limit < len(lines):
        lines = lines[:limit]
    return "\n".join(lines)[:50000]

def run_write(path: str, content: str) -> str:
    """Write to a file safely, ensuring it is within the workspace."""
    logger.debug(f"Writing to file at path: {path}")
    safe_path(path).write_text(content)
    return "File written successfully."

def run_edit(path: str, old_text: str, new_text: str) -> str:
    """Edit a file safely, ensuring it is within the workspace."""
    logger.debug(f"Editing file at path: {path}")
    file_path = safe_path(path)
    text = file_path.read_text()
    if old_text not in text:
        return "Error: Old text not found in file."
    updated_text = text.replace(old_text, new_text)
    file_path.write_text(updated_text)
    return "File edited successfully."