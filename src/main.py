import argparse
import logging
import asyncio

from agent_loop import LoopState, agent_loop
from compact import CompactState, agent_compact_states
from hook import HookEvent, HookPayload, HookResponse, hook_manager

MAIN_AGENT_ID: str = "agent_main"

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
        asyncio.run(agent_loop(state, compact_state, MAIN_AGENT_ID))
        print(state.messages[-1]["content"])