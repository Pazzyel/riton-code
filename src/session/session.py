from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from agent_loop import (
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
from compact import CompactState, agent_compact_states
from cron import cron_schedule_loop
from hook import HookEvent, HookPayload, HookResponse, hook_manager
from notification import NOTIFICATION_QUEUE, Notification
from persistence import CheckpointRecord, MessageRecord, PersistenceStore, one_line_message
from session_persistence import SessionPersistenceCoordinator
from session_state import LoopState, SessionContext
from tools import TOOL_HANDLERS


logger = logging.getLogger(__name__)


class MainSession:
    """Coordinate the complete lifecycle of one interactive main-agent session."""

    def __init__(
        self,
        store: PersistenceStore,
        session_id: str,
        resumed: bool,
        context: SessionContext,
    ) -> None:
        """Initialize a session around an already created or restored context."""
        self.store = store
        self.session_id = session_id
        self.resumed = resumed
        self.context = context
        self.persistence = SessionPersistenceCoordinator(
            store,
            session_id,
            context,
            BACKGROUND_MANAGER,
            NOTIFICATION_QUEUE,
        )
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

    @classmethod
    async def create(
        cls,
        store: PersistenceStore,
        session_id: str,
        resumed: bool,
    ) -> MainSession:
        """Create a new context or restore one from its latest checkpoint."""
        checkpoint_record: Optional[CheckpointRecord] = store.load_checkpoint(
            session_id,
            session_id,
        )
        if resumed:
            context = cls._restore_context(session_id, checkpoint_record)
        else:
            context = await cls._create_context()
        agent_compact_states[session_id] = context.compact_state
        return cls(store, session_id, resumed, context)

    @staticmethod
    def _restore_context(
        session_id: str,
        checkpoint_record: Optional[CheckpointRecord],
    ) -> SessionContext:
        """Validate and deserialize a main-session checkpoint into runtime state."""
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
        context = SessionContext(
            loop_state=LoopState(
                messages=messages,
                turn_count=int(loop_payload.get("turn_count", 0)),
                transition_reason=loop_payload.get("transition_reason"),
            ),
            compact_state=CompactState.model_validate(compact_payload),
            pending_visible_response=bool(
                checkpoint_record.payload.get("pending_visible_response", False)
            ),
        )
        BACKGROUND_MANAGER.restore(
            [RuntimeTaskRecord.model_validate(task) for task in background_payload]
        )
        NOTIFICATION_QUEUE.restore(
            Notification.model_validate(item) for item in notification_payload
        )
        return context

    @staticmethod
    async def _create_context() -> SessionContext:
        """Create an empty context and apply SessionStart hook messages."""
        BACKGROUND_MANAGER.restore([])
        NOTIFICATION_QUEUE.restore([])
        context = SessionContext(
            loop_state=LoopState([], 0, None),
            compact_state=CompactState(),
        )
        start_response: HookResponse = await hook_manager.run_hooks(
            HookEvent(name="SessionStart", payload=HookPayload())
        )
        for hook_message in start_response.messages:
            context.loop_state.messages.append(
                {
                    "role": "system",
                    "content": f"[Hook message]: {hook_message}",
                }
            )
        return context

    async def run(self) -> None:
        """
        Run recovery, interactive input, scheduled work, and orderly cleanup.

        在应用启动时恢复checkpoint，并拉起输入接受任务，cron检查任务和notification消费任务
        """
        self._event_loop = asyncio.get_running_loop()
        BACKGROUND_MANAGER.set_callbacks(
            self.persistence.save_checkpoint,
            self._background_completed,
        )
        self.persistence.save_checkpoint()
        if self.resumed:
            await self._recover()
        self._print_history()
        cron_task: asyncio.Task[None] = asyncio.create_task(
            cron_schedule_loop(self._publish_notification)
        )
        notification_task: asyncio.Task[None] = asyncio.create_task(
            notification_processor_loop(
                self.context.loop_state,
                self.context.compact_state,
                self.session_id,
                NOTIFICATION_QUEUE,
                self.persistence.save_notification_batch,
                self._save_notification_answer,
                self.persistence.save_checkpoint,
                self.persistence.save_completed_foreground_subagent,
            )
        )
        try:
            await self._input_loop()
        finally:
            cron_task.cancel()
            notification_task.cancel()
            await asyncio.gather(
                cron_task,
                notification_task,
                return_exceptions=True,
            )
            self.persistence.save_checkpoint()
            BACKGROUND_MANAGER.set_callbacks(None)

    async def _recover(self) -> None:
        """Resume persisted work in the ordering required for at-least-once safety."""
        BACKGROUND_MANAGER.resume_running_tasks(self._resolve_background_handler)
        replayed_tools: bool = await recover_pending_tool_calls(
            self.context.loop_state,
            self.session_id,
            self.persistence.save_checkpoint,
            self.persistence.save_completed_foreground_subagent,
        )
        self._migrate_completed_background_tasks()
        messages = self.context.loop_state.messages
        last_role: Optional[str] = (
            str(messages[-1].get("role")) if messages else None
        )
        if replayed_tools or last_role in {"user", "tool"}:
            await self._run_agent_loop()
        recovered_answer: Optional[str] = final_answer(messages)
        if recovered_answer is not None and self.context.pending_visible_response:
            self.persistence.save_assistant_answer(recovered_answer)

    def _migrate_completed_background_tasks(self) -> None:
        """Move restored terminal main-agent tasks into the notification queue."""
        completed_tasks: List[RuntimeTaskRecord] = BACKGROUND_MANAGER.pop_completed_tasks(
            self.session_id
        )
        completed_subagent_calls: List[Any] = []
        for task in completed_tasks:
            NOTIFICATION_QUEUE.publish(self._notification_for_background_task(task))
            if task.tool_name == "subagent":
                completed_subagent_calls.append(task.tool_input.get("tool_call_id"))
        if not completed_tasks:
            return
        saved_with_child_cleanup: bool = False
        for parent_tool_call_id in completed_subagent_calls:
            saved_with_child_cleanup = (
                self.persistence.save_parent_and_delete_subagent(parent_tool_call_id)
                or saved_with_child_cleanup
            )
        if not saved_with_child_cleanup:
            self.persistence.save_checkpoint()

    def _publish_notification(self, notification: Notification) -> bool:
        """Publish a notification and persist the queue even for duplicate retries."""
        published: bool = NOTIFICATION_QUEUE.publish(notification)
        self.persistence.save_checkpoint()
        return published

    @staticmethod
    def _notification_for_background_task(task: RuntimeTaskRecord) -> Notification:
        """Convert a terminal background task into the standard notification form."""
        return Notification(
            id=f"background:{task.id}",
            content=format_background_task_notification([task]),
        )

    def _commit_background_completion(self, task: RuntimeTaskRecord) -> None:
        """Commit a worker result after control returns to the main event loop."""
        if task.agent_id != self.session_id:
            self.persistence.save_checkpoint()
            return
        removed_task: Optional[RuntimeTaskRecord] = BACKGROUND_MANAGER.remove_task(task.id)
        if removed_task is None:
            return
        NOTIFICATION_QUEUE.publish(self._notification_for_background_task(removed_task))
        if removed_task.tool_name == "subagent":
            parent_tool_call_id: Any = removed_task.tool_input.get("tool_call_id")
            if self.persistence.save_parent_and_delete_subagent(parent_tool_call_id):
                return
        self.persistence.save_checkpoint()

    def _background_completed(self, task: RuntimeTaskRecord) -> bool:
        """Schedule background completion handling from a worker thread safely."""
        if self._event_loop is None:
            return False
        try:
            self._event_loop.call_soon_threadsafe(
                self._commit_background_completion,
                task,
            )
        except RuntimeError:
            return False
        return True

    @staticmethod
    def _resolve_background_handler(
        tool_name: str,
        owner_agent_id: str,
    ) -> Optional[Callable]:
        """Resolve the executable handler used when a persisted task is resumed."""
        del owner_agent_id
        if tool_name == "subagent":
            return PARENT_TOOL_HANDLERS.get(tool_name)
        return TOOL_HANDLERS.get(tool_name)

    async def _input_loop(self) -> None:
        """Read terminal input until the user requests session exit."""
        while True:
            query: str = await asyncio.get_running_loop().run_in_executor(
                None,
                input,
                ">> ",
            )
            if query.strip().lower() == "exit":
                return
            await self._handle_query(query)

    async def _handle_query(self, query: str) -> None:
        """Persist and execute one visible user query under the shared agent lock."""
        async with agent_lock:
            self.context.loop_state.messages.append(
                {"role": "user", "content": query}
            )
            self.persistence.save_user_message(query)
            await self._run_agent_loop()
            answer: Optional[str] = final_answer(self.context.loop_state.messages)
            if answer is None:
                logger.warning("Agent loop ended without a final assistant answer")
                return
            if self.persistence.save_assistant_answer(answer):
                print(answer)

    async def _run_agent_loop(self) -> None:
        """Run the main agent with this session's persistence callbacks."""
        await agent_loop(
            self.context.loop_state,
            self.context.compact_state,
            self.session_id,
            self.persistence.save_checkpoint,
            self.persistence.save_completed_foreground_subagent,
        )

    def _save_notification_answer(self, answer: str) -> None:
        """Persist and print the final answer produced for queued notifications."""
        if self.persistence.save_assistant_answer(answer):
            print(answer)

    def _print_history(self) -> None:
        """Render persisted visible history before accepting new input."""
        messages: List[MessageRecord] = self.store.list_messages(self.session_id)
        for message in messages:
            print(f"{message.role}: {one_line_message(message.content)}")
