from typing import List, Dict, Optional, Any
import logging
import asyncio

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall

import config as config
from tools import TOOLS, TOOL_HANDLERS, SUBAGENT_TOOLS
from subagent import run_subagent_tool
from todo import todo_manager
from ai_config import client
from prompt.system_prompt import system_prompt_builder
from compact import CompactState, try_compact
from recovery import choose_recovery, RecoveryType, CONTINUE_MESSAGE, backoff_delay
from background import BACKGROUND_MANAGER
from cron import CRON_TRIGGER_INTERVAL, CronJob, cron_lock, cron_queue
from checkpoint import pending_tool_calls
from tool_execution import (
    CheckpointCallback,
    ToolCompletionCallback,
    execute_tool_call,
    recover_tool_call,
)

logger = logging.getLogger(__name__)

MAX_TURNS: int = 20  # Maximum number of turns in the agent loop before stopping

PARENT_TOOL_HANDLERS = TOOL_HANDLERS.copy()
PARENT_TOOL_HANDLERS["subagent"] = lambda **kw: run_subagent_tool(
    prompt=kw["prompt"],
    agent_id=kw["agent_id"],
    tool_call_id=kw["tool_call_id"],
)
PARENT_TOOLS = TOOLS + SUBAGENT_TOOLS

class LoopState:
    messages:           List[Dict[str, Any]]    # The list of messages in the conversation history
    turn_count:         int                     # The number of turns taken in the loop
    transition_reason:  Optional[str]           # The reason for transitioning to the next turn, if applicable

    def __init__(self, messages: List[Dict[str, Any]], turn_count: int, transition_reason: Optional[str]):
        self.messages = messages
        self.turn_count = turn_count
        self.transition_reason = transition_reason

async def agent_loop(
    state: LoopState,
    compact_state: CompactState,
    agent_id: str,
    checkpoint_callback: Optional[CheckpointCallback] = None,
    completion_callback: Optional[ToolCompletionCallback] = None,
) -> None:
    """
    Run the agent loop until completion.

    The loop will continue until the agent decides to stop by returning False from run_one_loop.
    """

    logger.debug("Starting agent loop")
    # 开启loop先删持久化一次，保险
    state.messages = await try_compact(state.messages, compact_state)
    if checkpoint_callback is not None:
        checkpoint_callback()
    loop_count: int = 0
    while loop_count < MAX_TURNS:
        should_continue: bool = await run_one_loop(
            state,
            agent_id,
            compact_state,
            checkpoint_callback,
            completion_callback,
        )
        loop_count += 1
        if checkpoint_callback is not None:
            checkpoint_callback()
        if not should_continue:
            break
        state.messages = await try_compact(state.messages, compact_state)
        if checkpoint_callback is not None:
            checkpoint_callback()
    
    logger.debug("Agent loop finished after %d turns", state.turn_count)

retry_count = 0  # Initialize retry count for backoff strategy
async def run_one_loop(
    state: LoopState,
    agent_id: str,
    compact_state: CompactState,
    checkpoint_callback: Optional[CheckpointCallback] = None,
    completion_callback: Optional[ToolCompletionCallback] = None,
) -> bool:
    """
    Run one loop of the agent's reasoning and acting process.

    Returns True if the loop should continue, or False if it should stop.
    """
    logger.debug("Running loop turn %d", state.turn_count + 1)
    try:
        response: ChatCompletion = await client.chat.completions.create(
            model=config.MODEL_ID,
            messages=[{"role": "system", "content": await system_prompt_builder.build()}] + state.messages, # type: ignore
            tools=PARENT_TOOLS, # type: ignore
            max_tokens=config.MAX_TOKENS,
        )
        stop_reason: Optional[str] = response.choices[0].finish_reason
        decision: RecoveryType = choose_recovery(stop_reason, None)
    except Exception as e:
        logger.error("Error during chat completion: %s", str(e))
        error_text = str(e)
        decision: RecoveryType = choose_recovery(None, error_text)

    global retry_count
    match decision.type:
        case "continue":
            retry_count = 0  # Reset retry count on success
            state.messages.append({
                "role": "user",
                "content": CONTINUE_MESSAGE,
            })
            return True  # Continue the loop
        case "compact":
            retry_count = 0  # Reset retry count on success
            logger.info("Compacting conversation history due to context length")
            state.messages = await try_compact(state.messages, compact_state)
            return True  # Continue the loop after compacting
        case "backoff":
            logger.warning("Encountered transient error, backing off before retrying")
            await asyncio.sleep(backoff_delay(retry_count))  # Simple backoff strategy, can be improved with exponential backoff
            retry_count += 1
            return True
        case "fail":
            retry_count = 0  # Reset retry count on unrecoverable error
            logger.error("Unrecoverable error encountered, stopping agent loop")
            return False  # Stop the loop on unrecoverable error

    retry_count = 0  # Reset retry count on success or unrecoverable error
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
    # 模型决定调用工具后持久化
    state.messages.append(assistant_payload)
    if checkpoint_callback is not None:
        checkpoint_callback()

    if len(valid_tool_calls) == 0:
        logger.debug("No tool calls requested by model, stopping loop")
        state.transition_reason = None
        return False
    
    used_todo: bool = False
    for call in assistant_payload["tool_calls"]:
        tool_name: str = call["function"]["name"]
        if tool_name == "todo":
            used_todo = True
        await execute_tool_call(
            state.messages,
            call,
            PARENT_TOOL_HANDLERS,
            agent_id,
            checkpoint_callback,
            completion_callback,
        )


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

    # Background task notifications should be in the back of tool message
    background_notifications: str = BACKGROUND_MANAGER.get_background_task_notification(agent_id)
    if background_notifications is not None and background_notifications.strip() != "":
        state.messages.append({
            "role": "user",
            "content": background_notifications,
        })

    state.turn_count += 1
    state.transition_reason = "tool_call"
    logger.debug("Turn %d finished with transition_reason=%s", state.turn_count, state.transition_reason)
    return True


async def recover_pending_tool_calls(
    state: LoopState,
    agent_id: str,
    checkpoint_callback: Optional[CheckpointCallback] = None,
    completion_callback: Optional[ToolCompletionCallback] = None,
) -> bool:
    """恢复已经保存的工具调用但未执行成功的结果"""
    pending_calls: List[Dict[str, Any]] = pending_tool_calls(state.messages)
    if not pending_calls:
        return False
    for tool_call in pending_calls:
        await recover_tool_call(
            state.messages,
            tool_call,
            PARENT_TOOL_HANDLERS,
            agent_id,
            checkpoint_callback,
            completion_callback,
        )
    background_notifications: str = BACKGROUND_MANAGER.get_background_task_notification(agent_id)
    if background_notifications.strip():
        state.messages.append(
            {
                "role": "user",
                "content": background_notifications,
            }
        )
    state.turn_count += 1
    state.transition_reason = "tool_call"
    if checkpoint_callback is not None:
        checkpoint_callback()
    return True





# 把CRON任务Agent执行的部分移动到这里，防止循环依赖

# 为防死锁，固定先加agent_lock，再加cron_lock
agent_lock: asyncio.Lock = asyncio.Lock()  # User input and cron job may race


async def queue_processor_loop(
    state: LoopState,
    compact_state: CompactState,
    agent_id: str,
    checkpoint_callback: Optional[CheckpointCallback] = None,
    completion_callback: Optional[ToolCompletionCallback] = None,
) -> None:
    while True:
        await asyncio.sleep(CRON_TRIGGER_INTERVAL)
        if not await has_cron_queue():
            continue
        async with agent_lock:
            if not await has_cron_queue():
                continue
            logger.info(f"[cron] Processing {len(cron_queue)} queued jobs")
            triggered_jobs: List[CronJob] = await consume_cron_queue()
            for job in triggered_jobs:
                logger.info(f"[cron] Injecting job {job.id} into agent loop")
                state.messages.append({
                    "role": "user",
                    "content": f"[cron job {job.id}] {job.prompt}",
                })
            if checkpoint_callback is not None:
                checkpoint_callback()
            await agent_loop(
                state,
                compact_state,
                agent_id,
                checkpoint_callback,
                completion_callback,
            )

async def has_cron_queue() -> bool:
    """Check if there are any jobs in the cron_queue."""
    async with cron_lock:
        return len(cron_queue) > 0

async def consume_cron_queue() -> List[CronJob]:
    """Consume and return all jobs in the cron_queue."""
    async with cron_lock:
        jobs: List[CronJob] = list(cron_queue)
        cron_queue.clear()
        return jobs
