from pathlib import Path
import sys
from typing import Any
import unittest
from unittest.mock import AsyncMock, patch


SRC_DIR: Path = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from hook import HookResponse  # noqa: E402
from tool_execution import execute_tool_call  # noqa: E402


class ToolExecutionCheckpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_completion_callback_can_atomically_replace_checkpoint_save(
        self,
    ) -> None:
        messages: list[dict[str, Any]] = []
        checkpoint_count: list[int] = [0]
        completion_count: list[int] = [0]
        tool_call: dict[str, Any] = {
            "id": "call_child",
            "type": "function",
            "function": {
                "name": "subagent",
                "arguments": '{"prompt":"work"}',
            },
        }

        def checkpoint_saved() -> None:
            checkpoint_count[0] += 1

        def completion_saved(tool_name: str, tool_input: dict[str, Any]) -> bool:
            self.assertEqual("subagent", tool_name)
            self.assertEqual("call_child", tool_input["tool_call_id"])
            completion_count[0] += 1
            return True

        with (
            patch(
                "tool_execution.hook_manager.run_hooks",
                AsyncMock(return_value=HookResponse()),
            ),
            patch("tool_execution.run_tool", AsyncMock(return_value="done")),
        ):
            await execute_tool_call(
                messages,
                tool_call,
                {},
                "session_test",
                checkpoint_saved,
                completion_saved,
            )

        self.assertEqual([1], completion_count)
        self.assertEqual([0], checkpoint_count)
        self.assertEqual("tool", messages[-1]["role"])
        self.assertEqual("done", messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
