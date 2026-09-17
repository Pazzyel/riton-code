from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, Dict
import unittest


SRC_DIR: Path = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from checkpoint import build_subagent_checkpoint, subagent_id_for  # noqa: E402
from compact import CompactState  # noqa: E402
from persistence import PersistenceStore, set_persistence_store  # noqa: E402
from subagent import run_subagent_tool  # noqa: E402


class SubagentRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self.store: PersistenceStore = PersistenceStore(
            Path(self.temp_dir.name) / "agent.db"
        )
        self.session_id: str = "session_test"
        self.tool_call_id: str = "call_subagent"
        self.store.create_session(self.session_id)
        set_persistence_store(self.store)

    def tearDown(self) -> None:
        set_persistence_store(None)
        self.temp_dir.cleanup()

    async def test_completed_subagent_checkpoint_returns_without_starting_over(self) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "work"},
            {"role": "assistant", "content": "completed result"},
        ]
        payload: Dict[str, Any] = build_subagent_checkpoint(
            messages,
            [],
            20,
            1,
            CompactState(),
        )
        subagent_id: str = subagent_id_for(self.session_id, self.tool_call_id)
        self.store.save_checkpoint(
            self.session_id,
            subagent_id,
            "subagent",
            payload,
        )
        handlers: Dict[str, Callable] = {}

        result: str = await run_subagent_tool(
            "work",
            self.session_id,
            self.tool_call_id,
            tools=[],
            handlers=handlers,
        )

        self.assertEqual("completed result", result)
        self.assertIsNotNone(
            self.store.load_checkpoint(self.session_id, subagent_id)
        )


if __name__ == "__main__":
    unittest.main()
