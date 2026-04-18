from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from pathlib import Path
import time
import json
import logging
import asyncio
import aiofiles

from openai.types.chat.chat_completion import ChatCompletion
import tiktoken

from directory import TRANSCRIPT_DIR, TOOL_RESULTS_DIR, WORKDIR
from ai_config import client, MODEL_CONTEXT_LIMIT
import config

logger = logging.getLogger(__name__)

KEEP_RECENT_TOOL_RESULTS: int = 3
KEEP_RECENT_MESSAGES: int = 5
KEEP_TOOL_RESULT_LENGTH: int = 120 # if a tool result is less than this number of characters, we will keep it in the messages even if it's not in the most recent ones
PERSIST_THRESHOLD: int = 30000
PREVIEW_CHARS: int = 2000



class CompactState(BaseModel):
    """State representation for the compact agent."""
    has_compacted: bool = False
    last_summary: str = ""
    recent_files: List[str] = []

def track_recent_files(state: CompactState, file_path: str) -> None:
    """Track recently accessed files in the compact state, keeping only the most recent 5 files."""
    # Why remove ? Put the new edited file at the end of the list 
    if file_path in state.recent_files:
        state.recent_files.remove(file_path)

    state.recent_files.append(file_path)
    if len(state.recent_files) > 5:
        state.recent_files.pop(0)

async def persist_large_tool_output(tool_call_id: str, output: str) -> str:
    """Persist large tool output to disk and return the file path."""
    if len(output) <= PERSIST_THRESHOLD:
        return output  # No need to persist if output is below threshold
    
    TOOL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    file_path: Path = TOOL_RESULTS_DIR / f"{tool_call_id}.txt"
    if not file_path.exists():
        async with aiofiles.open(file_path, "w") as f:
            await f.write(output)

    preview: str = output[:PREVIEW_CHARS] + "... [truncated]"
    realtive_path: Path = file_path.relative_to(WORKDIR)
    return f"""
        <persisted_output>
        Full output for tool call {tool_call_id} has been persisted to {str(realtive_path)} 
        due to its length exceeding {PERSIST_THRESHOLD} characters.\n
        Preview of the output:\n
        {preview}\n
        </persisted_output>
    """

def extract_tool_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract only the tool messages from the conversation history."""
    return [msg for msg in messages if msg["role"] == "tool"]

def micro_compact(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Perform a micro compaction by removing tool call details and keeping only the most recent ones.
    
    It will replace tool messages to ""[Earlier tool result compacted. Re-run the tool if you need full detail.]" 
    
    except for the most recent KEEP_RECENT_TOOL_RESULTS tool messages.
    """
    compacted_messages: List[Dict[str, Any]] = []
    tool_messages: List[Dict[str, Any]] = extract_tool_messages(messages)

    if len(tool_messages) <= KEEP_RECENT_TOOL_RESULTS:
        return messages  # No need to compact if tool messages are within the limit
    
    for tool_msg in tool_messages[:-KEEP_RECENT_TOOL_RESULTS]:
        if not isinstance(tool_msg.get("content", ""), str) or len(tool_msg["content"]) <= KEEP_TOOL_RESULT_LENGTH:
            continue  # Skip compaction for non-string content or short content
        compacted_messages.append({
            "role": "tool",
            "name": tool_msg.get("name", "unknown"),
            "content": "[Earlier tool result compacted. Re-run the tool if you need full detail.]"
        })
    return compacted_messages

async def summary_messages(messages: List[Dict[str, Any]]) -> str:
    """Summarize the conversation history and return a new list of messages with the summary."""
    summary_prompt: str = f"""
        Summarize this coding-agent conversation so work can continue.\n
        Preserve:\n
        1. The current goal\n
        2. Important findings and decisions\n
        3. Files read or changed\n
        4. Remaining work\n
        5. User constraints and preferences\n
        Be compact but concrete.\n\n
        {messages}
    """
    response: ChatCompletion = await client.chat.completions.create(
        model=config.MODEL_ID,
        messages=[{"role": "user", "content": summary_prompt}],
        max_tokens=MODEL_CONTEXT_LIMIT,
    )

    return response.choices[0].message.content.strip() if response.choices and response.choices[0].message and response.choices[0].message.content else "Summary failed."

async def write_history_to_transcript(messages: List[Dict[str, Any]]) -> Path:
    """Write the conversation history to disk for later reference."""
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    file_path: Path = TRANSCRIPT_DIR / f"history_{time.time()}.jsonl"
    async with aiofiles.open(file_path, "w") as f:
        for msg in messages:
            await f.write(f"{json.dumps(msg, default=str)}\n")
    return file_path

async def compact_history(messages: List[Dict[str, Any]], state: CompactState, focus: Optional[str] = None) -> List[Dict[str, Any]]:
    """Compact the conversation history if it exceeds the context limit."""
    transcript_path: Path = await write_history_to_transcript(messages)
    logger.info(f"[Conversation history written to: {transcript_path}]")

    summary: str = await summary_messages(messages)
    if focus:
        summary += f"\n\nFocus for next steps: {focus}"
    if state.recent_files and len(state.recent_files) > 0:
        recent_files_lines: str = "\n".join(f"- {file}" for file in state.recent_files)
        summary += f"\n\nRecent files to reopen if needed: {recent_files_lines}"

    state.last_summary = summary
    state.has_compacted = True
    
    return [
        {
            "role": "user", 
            "content": f"""
                [Conversation history has been compacted due to length exceeding context limit.]\n\n
                {summary}
            """
        }
    ]

def estimate_message_tokens(messages: List[Dict[str, Any]]) -> int:
    """Estimate the number of tokens in the messages."""
    encoder = tiktoken.get_encoding("cl100k_base")
    return sum(len(encoder.encode(msg.get("content", ""))) for msg in messages)

async def try_compact(messages: List[Dict[str, Any]], state: CompactState, focus: Optional[str] = None) -> List[Dict[str, Any]]:
    """Try to compact the messages if they exceed the context limit."""
    if len(messages) <= KEEP_RECENT_MESSAGES:
        return messages  # No need to compact if messages are within the limit
    cuted_messages: List[Dict[str, Any]] = messages[:-KEEP_RECENT_MESSAGES]
    compacted_messages: List[Dict[str, Any]] = micro_compact(cuted_messages)
    
    if estimate_message_tokens(compacted_messages) <= MODEL_CONTEXT_LIMIT:
        return messages  # No need to compact
    
    logger.info("Context length exceeds limit, performing compaction...")
    compacted_messages = await compact_history(messages, state, focus)
    
    return compacted_messages + messages[-KEEP_RECENT_MESSAGES:]

agent_compact_states: Dict[str, CompactState] = {}