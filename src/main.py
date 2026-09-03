import argparse
import logging
import asyncio

from agent_loop import LoopState, agent_loop, queue_processor_loop, agent_lock
from compact import CompactState, agent_compact_states
from hook import HookEvent, HookPayload, HookResponse, hook_manager
from cron import cron_schedule_loop

MAIN_AGENT_ID: str = "agent_main"

async def input_loop(state: LoopState, compact_state: CompactState, agent_id: str):
    """Asynchronous input loop to read user input and feed it into the agent loop."""
    while True:
        query: str = await asyncio.get_event_loop().run_in_executor(None, input, ">> ")
        if query.strip().lower() == "exit":
            break

        async with agent_lock:
            state.messages.append({
                "role": "user",
                "content": query,
            })
            await agent_loop(state, compact_state, agent_id)
            print(state.messages[-1]["content"])

async def loops(state: LoopState, compact_state: CompactState, agent_id: str):
    # 包含输入线程，定时器检查线程，和cron队列处理线程
    await asyncio.gather(
        cron_schedule_loop(),
        queue_processor_loop(state, compact_state, agent_id),
        input_loop(state, compact_state, agent_id)
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the coding agent loop.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging output")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, 
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", 
        force=True
    )

    logger = logging.getLogger(__name__)

    logger.debug("Debug logging enabled")

    state: LoopState = LoopState(
        messages=[],
        turn_count=0,
        transition_reason=None,
    )
    compact_state: CompactState = CompactState()
    agent_compact_states[MAIN_AGENT_ID] = compact_state

    # Hook result was ignored
    start_response: HookResponse = asyncio.run(hook_manager.run_hooks(HookEvent(name="SessionStart", payload=HookPayload())))
    for msg in start_response.messages:
        state.messages.append({
            "role": "system",
            "content": f"[Hook message]: {msg}",
        })

    asyncio.run(loops(state, compact_state, MAIN_AGENT_ID))

