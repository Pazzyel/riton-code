import asyncio
from pathlib import Path
import sys
import tempfile
from typing import Any
import unittest
from unittest.mock import AsyncMock, patch


SRC_DIR: Path = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from background import BACKGROUND_MANAGER, BackgroundManager, RuntimeTaskRecord  # noqa: E402
from compact import CompactState, agent_compact_states  # noqa: E402
from hook import HookResponse  # noqa: E402
from notification import NOTIFICATION_QUEUE, Notification, NotificationQueue  # noqa: E402
from persistence import CHECKPOINT_VERSION, PersistenceStore  # noqa: E402
from session import MainSession  # noqa: E402
from session_persistence import SessionPersistenceCoordinator  # noqa: E402
from session_state import LoopState, SessionContext  # noqa: E402


class SessionPersistenceCoordinatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self.store: PersistenceStore = PersistenceStore(
            Path(self.temp_dir.name) / "agent.db"
        )
        self.session_id: str = "session_test"
        self.store.create_session(self.session_id)
        self.background_manager: BackgroundManager = BackgroundManager()
        self.notification_queue: NotificationQueue = NotificationQueue()
        self.context: SessionContext = SessionContext(
            LoopState([], 0, None),
            CompactState(),
        )
        self.coordinator: SessionPersistenceCoordinator = (
            SessionPersistenceCoordinator(
                self.store,
                self.session_id,
                self.context,
                self.background_manager,
                self.notification_queue,
            )
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_checkpoint_snapshots_runtime_state(self) -> None:
        task: RuntimeTaskRecord = RuntimeTaskRecord(
            id="bg_task_1",
            agent_id=self.session_id,
            tool_name="read_file",
            tool_input={},
            command="",
        )
        self.background_manager.restore([task])
        self.notification_queue.publish(Notification(id="notice_1", content="ready"))

        self.coordinator.save_checkpoint()

        record = self.store.load_checkpoint(self.session_id, self.session_id)
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual("bg_task_1", record.payload["background_tasks"][0]["id"])
        self.assertEqual("notice_1", record.payload["pending_notifications"][0]["id"])

    def test_visible_messages_update_pending_state_and_checkpoint(self) -> None:
        notifications = [
            Notification(id="one", content="first"),
            Notification(id="two", content="second"),
        ]

        self.coordinator.save_notification_batch(notifications)
        self.assertTrue(self.context.pending_visible_response)
        self.assertTrue(self.coordinator.save_assistant_answer("done"))
        self.assertFalse(self.context.pending_visible_response)

        visible = self.store.list_messages(self.session_id)
        self.assertEqual(
            ["user", "user", "assistant"],
            [message.role for message in visible],
        )
        record = self.store.load_checkpoint(self.session_id, self.session_id)
        self.assertIsNotNone(record)
        assert record is not None
        self.assertFalse(record.payload["pending_visible_response"])

    def test_subagent_completion_deletes_child_checkpoint(self) -> None:
        child_id: str = f"subagent:{self.session_id}:call_child"
        self.store.save_checkpoint(
            self.session_id,
            child_id,
            "subagent",
            {
                "version": CHECKPOINT_VERSION,
                "context": {"messages": [], "tools": [], "max_turns": 1, "turn_count": 0},
                "compact_state": CompactState().model_dump(),
            },
        )

        handled: bool = self.coordinator.save_completed_foreground_subagent(
            "subagent",
            {"tool_call_id": "call_child"},
        )

        self.assertTrue(handled)
        self.assertIsNone(self.store.load_checkpoint(self.session_id, child_id))


class MainSessionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self.store: PersistenceStore = PersistenceStore(
            Path(self.temp_dir.name) / "agent.db"
        )
        self.session_id: str = "session_test"
        self.store.create_session(self.session_id)
        BACKGROUND_MANAGER.restore([])
        NOTIFICATION_QUEUE.restore([])

    def tearDown(self) -> None:
        BACKGROUND_MANAGER.restore([])
        BACKGROUND_MANAGER.set_callbacks(None)
        NOTIFICATION_QUEUE.restore([])
        agent_compact_states.pop(self.session_id, None)
        self.temp_dir.cleanup()

    async def test_create_new_session_applies_start_hook(self) -> None:
        with patch(
            "session.hook_manager.run_hooks",
            AsyncMock(return_value=HookResponse(messages=["startup context"])),
        ):
            session: MainSession = await MainSession.create(
                self.store,
                self.session_id,
                False,
            )

        self.assertEqual(
            "[Hook message]: startup context",
            session.context.loop_state.messages[-1]["content"],
        )
        self.assertIs(
            session.context.compact_state,
            agent_compact_states[self.session_id],
        )

    async def test_recovery_migrates_completed_task_after_tool_recovery(self) -> None:
        task: RuntimeTaskRecord = RuntimeTaskRecord(
            id="bg_task_7",
            agent_id=self.session_id,
            tool_name="read_file",
            tool_input={"tool_call_id": "call_7"},
            command="",
            status="completed",
            result="done",
            result_preview="done",
        )
        BACKGROUND_MANAGER.restore([task])
        context = SessionContext(
            LoopState([{"role": "assistant", "content": "complete"}], 1, None),
            CompactState(),
        )
        session = MainSession(self.store, self.session_id, True, context)

        async def verify_task_still_exists(*args: Any, **kwargs: Any) -> bool:
            del args, kwargs
            self.assertEqual(1, len(BACKGROUND_MANAGER.snapshot()))
            return False

        with patch(
            "session.recover_pending_tool_calls",
            AsyncMock(side_effect=verify_task_still_exists),
        ):
            await session._recover()

        self.assertEqual([], BACKGROUND_MANAGER.snapshot())
        pending = NOTIFICATION_QUEUE.snapshot()
        self.assertEqual(["background:bg_task_7"], [item.id for item in pending])

    async def test_background_completion_is_committed_on_event_loop(self) -> None:
        task: RuntimeTaskRecord = RuntimeTaskRecord(
            id="bg_task_8",
            agent_id=self.session_id,
            tool_name="read_file",
            tool_input={},
            command="",
            status="completed",
            result="done",
            result_preview="done",
        )
        BACKGROUND_MANAGER.restore([task])
        context = SessionContext(LoopState([], 0, None), CompactState())
        session = MainSession(self.store, self.session_id, False, context)
        session._event_loop = asyncio.get_running_loop()

        self.assertTrue(session._background_completed(task))
        await asyncio.sleep(0)

        self.assertEqual([], BACKGROUND_MANAGER.snapshot())
        self.assertEqual(
            ["background:bg_task_8"],
            [item.id for item in NOTIFICATION_QUEUE.snapshot()],
        )


if __name__ == "__main__":
    unittest.main()
