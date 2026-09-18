from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from threading import RLock
from typing import Any, Callable, Dict, List, Optional
import uuid

from persistence import (
    CheckpointRecord,
    MessageRecord,
    PersistenceStore,
    SessionRecord,
    get_persistence_store,
    one_line_message,
    set_persistence_store,
)


logger = logging.getLogger(__name__)

# 目前传入的change_callback，checkpoint_callback的唯一行为是更新当前主agent的checkpoint
# 目前传入的completion_callback的唯一行为是更新当前主agent的checkpoint并删除子agent的

def build_parser() -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run the coding agent loop."
    )
    session_group: argparse._MutuallyExclusiveGroup = parser.add_mutually_exclusive_group()
    session_group.add_argument(
        "-s",
        "--session",
        dest="session_id",
        help="Continue an existing session.",
    )
    session_group.add_argument(
        "-l",
        "--list-sessions",
        action="store_true",
        help="List sessions ordered by their last visible conversation time.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging output.",
    )
    return parser


def print_sessions(store: PersistenceStore) -> None:
    sessions: List[SessionRecord] = store.list_sessions()
    for session in sessions:
        print(f"{session.session_id} {session.updated_at}")


def print_history(store: PersistenceStore, session_id: str) -> None:
    messages: List[MessageRecord] = store.list_messages(session_id)
    for message in messages:
        print(f"{message.role}: {one_line_message(message.content)}")


async def run_session(
    store: PersistenceStore,
    session_id: str,
    resumed: bool,
) -> None:
    # Runtime imports intentionally stay below the -l fast path so listing sessions
    # does not initialize the model client or require an API key.
    from agent_loop import (
        LoopState,
        PARENT_TOOL_HANDLERS,
        agent_lock,
        agent_loop,
        final_answer,
        notification_processor_loop,
        recover_pending_tool_calls,
    )
    from background import (
        BACKGROUND_MANAGER,
        RuntimeTaskRecord,
        format_background_task_notification,
    )
    from checkpoint import build_main_checkpoint, subagent_id_for
    from compact import CompactState, agent_compact_states
    from cron import cron_schedule_loop
    from hook import HookEvent, HookPayload, HookResponse, hook_manager
    from notification import NOTIFICATION_QUEUE, Notification
    from tools import TOOL_HANDLERS

    checkpoint_record: Optional[CheckpointRecord] = store.load_checkpoint(
        session_id,
        session_id,
    )
    if resumed:
        # 继续上次的对话
        if checkpoint_record is None:
            raise ValueError(f"Session {session_id} has no checkpoint")
        if checkpoint_record.agent_type != "main":
            raise ValueError(f"Session {session_id} checkpoint is not a main checkpoint")
        loop_payload: Any = checkpoint_record.payload.get("loop_state")
        compact_payload: Any = checkpoint_record.payload.get("compact_state")
        background_payload: Any = checkpoint_record.payload.get("background_tasks")
        notification_payload: Any = checkpoint_record.payload.get(
            "pending_notifications",
            [],
        )
        if (
            not isinstance(loop_payload, dict)
            or not isinstance(compact_payload, dict)
            or not isinstance(background_payload, list)
            or not isinstance(notification_payload, list)
        ):
            raise ValueError(f"Session {session_id} checkpoint is incomplete")
        messages: Any = loop_payload.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"Session {session_id} checkpoint messages are invalid")
        state: LoopState = LoopState(
            messages=messages,
            turn_count=int(loop_payload.get("turn_count", 0)),
            transition_reason=loop_payload.get("transition_reason"),
        )
        compact_state: CompactState = CompactState.model_validate(compact_payload)
        pending_visible_response: bool = bool(
            checkpoint_record.payload.get("pending_visible_response", False)
        )
        background_tasks: List[RuntimeTaskRecord] = [
            RuntimeTaskRecord.model_validate(task) for task in background_payload
        ]
        BACKGROUND_MANAGER.restore(background_tasks)
        NOTIFICATION_QUEUE.restore(
            Notification.model_validate(item) for item in notification_payload
        )
    else:
        # 新对话
        state = LoopState(messages=[], turn_count=0, transition_reason=None)
        compact_state = CompactState()
        pending_visible_response = False
        BACKGROUND_MANAGER.restore([])
        NOTIFICATION_QUEUE.restore([])
        start_response: HookResponse = await hook_manager.run_hooks(
            HookEvent(name="SessionStart", payload=HookPayload())
        )
        for hook_message in start_response.messages:
            state.messages.append(
                {
                    "role": "system",
                    "content": f"[Hook message]: {hook_message}",
                }
            )

    agent_compact_states[session_id] = compact_state
    checkpoint_lock: RLock = RLock()
    event_loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()

    def current_payload() -> Dict[str, Any]:
        payload: Dict[str, Any] = build_main_checkpoint(
            state.messages,
            state.turn_count,
            state.transition_reason,
            compact_state,
            BACKGROUND_MANAGER.snapshot(),
            pending_visible_response,
            NOTIFICATION_QUEUE.snapshot(),
        )
        return payload

    def save_checkpoint() -> None:
        with checkpoint_lock:
            payload: Dict[str, Any] = current_payload()
            store.save_checkpoint(session_id, session_id, "main", payload)

    def save_parent_and_delete_subagent(parent_tool_call_id: Any) -> bool:
        if not isinstance(parent_tool_call_id, str):
            return False
        with checkpoint_lock:
            payload: Dict[str, Any] = current_payload()
            store.save_parent_checkpoint_and_delete_child(
                session_id,
                session_id,
                "main",
                payload,
                subagent_id_for(session_id, parent_tool_call_id),
            )
        return True

    def save_completed_foreground_subagent(
        tool_name: str,
        tool_input: Dict[str, Any],
    ) -> bool:
        if tool_name != "subagent" or tool_input.get("run_in_background") is True:
            return False
        parent_tool_call_id: Any = tool_input.get("tool_call_id")
        return save_parent_and_delete_subagent(parent_tool_call_id)

    def publish_notification(notification: Notification) -> bool:
        published: bool = NOTIFICATION_QUEUE.publish(notification)
        # Persist even on a duplicate retry so a previous failed save can recover.
        save_checkpoint()
        return published

    def notification_for_background_task(task: RuntimeTaskRecord) -> Notification:
        return Notification(
            id=f"background:{task.id}",
            content=format_background_task_notification([task]),
        )

    def commit_background_completion(task: RuntimeTaskRecord) -> None:
        """Commit a worker result on the main event loop."""
        if task.agent_id != session_id:
            # Subagent-owned tasks retain their existing polling behavior.
            save_checkpoint()
            return
        removed_task: Optional[RuntimeTaskRecord] = BACKGROUND_MANAGER.remove_task(task.id)
        if removed_task is None:
            return
        NOTIFICATION_QUEUE.publish(notification_for_background_task(removed_task))
        if removed_task.tool_name == "subagent":
            parent_tool_call_id: Any = removed_task.tool_input.get("tool_call_id")
            if save_parent_and_delete_subagent(parent_tool_call_id):
                return
        save_checkpoint()

    def background_completed(task: RuntimeTaskRecord) -> bool:
        """Move completion handling from a worker thread to the main event loop."""
        try:
            event_loop.call_soon_threadsafe(commit_background_completion, task)
        except RuntimeError:
            return False
        return True

    def resolve_background_handler(
        tool_name: str,
        owner_agent_id: str,
    ) -> Optional[Callable]:
        del owner_agent_id
        if tool_name == "subagent":
            return PARENT_TOOL_HANDLERS.get(tool_name)
        return TOOL_HANDLERS.get(tool_name)

    BACKGROUND_MANAGER.set_callbacks(save_checkpoint, background_completed)
    save_checkpoint()

    if resumed:
        # 启动所有未完成的后台任务
        BACKGROUND_MANAGER.resume_running_tasks(resolve_background_handler)
        replayed_tools: bool = await recover_pending_tool_calls(
            state,
            session_id,
            save_checkpoint,
            save_completed_foreground_subagent,
        )
        completed_tasks: List[RuntimeTaskRecord] = BACKGROUND_MANAGER.pop_completed_tasks(
            session_id
        )
        completed_subagent_calls: List[Any] = []
        for task in completed_tasks:
            NOTIFICATION_QUEUE.publish(notification_for_background_task(task))
            if task.tool_name == "subagent":
                completed_subagent_calls.append(task.tool_input.get("tool_call_id"))
        if completed_tasks:
            saved_with_child_cleanup: bool = False
            for parent_tool_call_id in completed_subagent_calls:
                saved_with_child_cleanup = (
                    save_parent_and_delete_subagent(parent_tool_call_id)
                    or saved_with_child_cleanup
                )
            if not saved_with_child_cleanup:
                save_checkpoint()
        last_role: Optional[str] = (
            str(state.messages[-1].get("role")) if state.messages else None
        )
        if replayed_tools or last_role in {"user", "tool"}:
            await agent_loop(
                state,
                compact_state,
                session_id,
                save_checkpoint,
                save_completed_foreground_subagent,
            )
        recovered_answer: Optional[str] = final_answer(state.messages)
        if recovered_answer is not None and pending_visible_response:
            pending_visible_response = False
            store.save_visible_message_and_checkpoint(
                session_id,
                "assistant",
                recovered_answer,
                session_id,
                "main",
                current_payload(),
                only_if_last_user=True,
            )

    print_history(store, session_id)

    def save_notification_batch(notifications: List[Notification]) -> None:
        nonlocal pending_visible_response
        pending_visible_response = True
        store.save_visible_messages_and_checkpoint(
            session_id,
            [("user", notification.content) for notification in notifications],
            session_id,
            "main",
            current_payload(),
        )

    def save_notification_answer(answer: str) -> None:
        nonlocal pending_visible_response
        pending_visible_response = False
        saved: bool = store.save_visible_message_and_checkpoint(
            session_id,
            "assistant",
            answer,
            session_id,
            "main",
            current_payload(),
            only_if_last_user=True,
        )
        if saved:
            print(answer)

    async def input_loop() -> None:
        nonlocal pending_visible_response
        while True:
            query: str = await asyncio.get_running_loop().run_in_executor(
                None,
                input,
                ">> ",
            )
            if query.strip().lower() == "exit":
                break
            async with agent_lock:
                state.messages.append({"role": "user", "content": query})
                pending_visible_response = True
                store.save_visible_message_and_checkpoint(
                    session_id,
                    "user",
                    query,
                    session_id,
                    "main",
                    current_payload(),
                )
                await agent_loop(
                    state,
                    compact_state,
                    session_id,
                    save_checkpoint,
                    save_completed_foreground_subagent,
                )
                answer: Optional[str] = final_answer(state.messages)
                if answer is None:
                    logger.warning("Agent loop ended without a final assistant answer")
                    continue
                pending_visible_response = False
                store.save_visible_message_and_checkpoint(
                    session_id,
                    "assistant",
                    answer,
                    session_id,
                    "main",
                    current_payload(),
                    only_if_last_user=True,
                )
                print(answer)

    cron_task: asyncio.Task[None] = asyncio.create_task(
        cron_schedule_loop(publish_notification)
    )
    queue_task: asyncio.Task[None] = asyncio.create_task(
        notification_processor_loop(
            state,
            compact_state,
            session_id,
            NOTIFICATION_QUEUE,
            save_notification_batch,
            save_notification_answer,
            save_checkpoint,
            save_completed_foreground_subagent,
        )
    )
    try:
        await input_loop()
    finally:
        cron_task.cancel()
        queue_task.cancel()
        await asyncio.gather(cron_task, queue_task, return_exceptions=True)
        save_checkpoint()
        BACKGROUND_MANAGER.set_callbacks(None)


def main(argv: Optional[List[str]] = None) -> int:
    parser: argparse.ArgumentParser = build_parser()
    args: argparse.Namespace = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        force=True,
    )
    store: PersistenceStore = get_persistence_store()
    set_persistence_store(store)

    if args.list_sessions:
        print_sessions(store)
        return 0

    resumed: bool = args.session_id is not None
    if resumed:
        session_id: str = str(args.session_id)
        if not store.session_exists(session_id):
            print(f"Session not found: {session_id}", file=sys.stderr)
            return 2
    else:
        session_id = f"session_{uuid.uuid4().hex}"
        store.create_session(session_id)
        print(f"session_id: {session_id}")

    try:
        asyncio.run(run_session(store, session_id, resumed))
    except (ValueError, TypeError) as exc:
        logger.error("Unable to start session: %s", str(exc))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
