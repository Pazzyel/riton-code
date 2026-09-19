from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from pathlib import Path
import time
import json
import logging
import re
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
        Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.
        This summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context.
        \n\n
        Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points. In your analysis process:
        \n\n
        1. Chronologically analyze each message and section of the conversation. For each section thoroughly identify:
           - The user's explicit requests and intents
           - Your approach to addressing the user's requests
           - Key decisions, technical concepts and code patterns
           - Specific details like:
             - file names
             - full code snippets
             - function signatures
             - file edits
           - Errors that you ran into and how you fixed them
           - Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
        2. Double-check for technical accuracy and completeness, addressing each required element thoroughly.`
        \n\n
        Finally, wrap your summary in <summary> tags. Your summary should include the following sections:
        \n\n
        1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
        2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
        3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable and include a summary of why this file read or edit is important.
        4. Errors and fixes: List all errors that you ran into, and how you fixed them. Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
        5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
        6. All user messages: List ALL user messages that are not tool results. These are critical for understanding the users' feedback and changing intent.
        7. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
        8. Current Work: Describe in detail precisely what was being worked on immediately before this summary request, paying special attention to the most recent messages from both user and assistant. Include file names and code snippets where applicable.
        9. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's most recent explicit requests, and the task you were working on immediately before this summary request. If your last task was concluded, then only list next steps if they are explicitly in line with the users request. Do not start on tangential requests or really old requests that were already completed without confirming with the user first.
           If there is a next step, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off. This should be verbatim to ensure there's no drift in task interpretation.`
        \n\n
        The text you need to output liked this.
        \n\n
        <analysis>
        [Your thought process, ensuring all points are covered thoroughly and accurately]
        </analysis>
        <summary>
        [The final summary result]
        </summary>
        \n\n
        Be compact but concrete. The following messages warped by <messages> is the origin messages you need to sumarry.
        \n\n
        <messages>
        {messages}
        </messages>
    """
    response: ChatCompletion = await client.chat.completions.create(
        model=config.MODEL_ID,
        messages=[{"role": "user", "content": summary_prompt}],
        max_tokens=MODEL_CONTEXT_LIMIT,
    )

    content: str = (
        response.choices[0].message.content.strip()
        if response.choices
        and response.choices[0].message
        and response.choices[0].message.content
        else "Summary failed."
    )
    summary_match: Optional[re.Match[str]] = re.search(
        r"<summary\b[^>]*>(.*?)</summary\s*>",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return summary_match.group(1).strip() if summary_match else content


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
