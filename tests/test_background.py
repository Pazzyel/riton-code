from pathlib import Path
import sys
from threading import Event
import time
from typing import Callable, Optional
import unittest


SRC_DIR: Path = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from background import (  # noqa: E402
    BACKGROUND_MANAGER,
    BackgroundManager,
    RuntimeTaskRecord,
)
from tool_execution import recover_tool_call  # noqa: E402


class BackgroundRecoveryTests(unittest.TestCase):
    def test_completion_callback_can_commit_without_duplicate_checkpoint(self) -> None:
        manager: BackgroundManager = BackgroundManager()
        completed: Event = Event()
        checkpoint_count: list[int] = [0]

        def checkpoint_changed() -> None:
            checkpoint_count[0] += 1

        def completion_committed(task: RuntimeTaskRecord) -> bool:
            self.assertEqual("completed", task.status)
            completed.set()
            return True

        def handler(**kwargs: str) -> str:
            return kwargs["value"]

        manager.set_callbacks(checkpoint_changed, completion_committed)
        manager.start_background_task(
            handler,
            "subagent",
            {"value": "done"},
            "session_test",
        )

        self.assertTrue(completed.wait(timeout=2))
        time.sleep(0.05)
        self.assertEqual(1, checkpoint_count[0])

    def test_running_task_is_resumed_and_completed_task_is_not(self) -> None:
        manager: BackgroundManager = BackgroundManager()
        running_task: RuntimeTaskRecord = RuntimeTaskRecord(
            id="bg_task_10",
            agent_id="session_test",
            tool_name="read_file",
            tool_input={"value": "resumed"},
            command="",
            status="running",
        )
        completed_task: RuntimeTaskRecord = RuntimeTaskRecord(
            id="bg_task_11",
            agent_id="session_test",
            tool_name="read_file",
            tool_input={"value": "old"},
            command="",
            status="completed",
            result="old",
        )
        manager.restore([running_task, completed_task])
        finished: Event = Event()
        call_count: list[int] = [0]

        def handler(**kwargs: str) -> str:
            call_count[0] += 1
            finished.set()
            return kwargs["value"]

        def resolver(tool_name: str, agent_id: str) -> Optional[Callable]:
            self.assertEqual("read_file", tool_name)
            self.assertEqual("session_test", agent_id)
            return handler

        manager.resume_running_tasks(resolver)
        self.assertTrue(finished.wait(timeout=2))
        deadline: float = time.time() + 2
        while time.time() < deadline:
            snapshots: list[RuntimeTaskRecord] = manager.snapshot()
            restored_running: RuntimeTaskRecord = next(
                task for task in snapshots if task.id == "bg_task_10"
            )
            if restored_running.status == "completed":
                break
            time.sleep(0.01)
        self.assertEqual(1, call_count[0])
        self.assertEqual("completed", restored_running.status)
        old_task: RuntimeTaskRecord = next(
            task for task in snapshots if task.id == "bg_task_11"
        )
        self.assertEqual("old", old_task.result)

    def test_snapshot_contains_replay_metadata(self) -> None:
        manager: BackgroundManager = BackgroundManager()
        finished: Event = Event()

        def handler(**kwargs: str) -> str:
            finished.set()
            return kwargs["prompt"]

        task_id: str = manager.start_background_task(
            handler,
            "subagent",
            {
                "prompt": "work",
                "agent_id": "session_test",
                "tool_call_id": "call_1",
            },
            "session_test",
        )
        self.assertTrue(finished.wait(timeout=2))
        task: RuntimeTaskRecord = next(
            task for task in manager.snapshot() if task.id == task_id
        )
        self.assertEqual("subagent", task.tool_name)
        self.assertEqual("call_1", task.tool_input["tool_call_id"])


class BackgroundToolCallRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_existing_background_task_gets_result_injected_without_duplicate(self) -> None:
        task: RuntimeTaskRecord = RuntimeTaskRecord(
            id="bg_task_20",
            agent_id="session_test",
            tool_name="read_file",
            tool_input={
                "path": "README.md",
                "agent_id": "session_test",
                "tool_call_id": "call_20",
            },
            command="",
            status="running",
        )
        BACKGROUND_MANAGER.restore([task])
        messages: list[dict[str, object]] = []
        checkpoints: list[int] = []
        tool_call: dict[str, object] = {
            "id": "call_20",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": '{"path":"README.md","run_in_background":true}',
            },
        }
        try:
            await recover_tool_call(
                messages,
                tool_call,
                {},
                "session_test",
                lambda: checkpoints.append(1),
            )
            self.assertEqual(1, len(BACKGROUND_MANAGER.snapshot()))
            self.assertEqual("tool", messages[-1]["role"])
            self.assertIn("restored", str(messages[-1]["content"]))
            self.assertEqual([1], checkpoints)
        finally:
            BACKGROUND_MANAGER.restore([])


if __name__ == "__main__":
    unittest.main()
