from pathlib import Path
import sys
from typing import Any
import unittest
from unittest.mock import AsyncMock, patch


SRC_DIR: Path = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from agent_loop import LoopState, recover_pending_tool_calls  # noqa: E402


class AgentToolRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_recovery_executes_only_missing_calls_in_original_order(self) -> None:
        state: LoopState = LoopState(
            messages=[
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        },
                        {
                            "id": "call_b",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        },
                        {
                            "id": "call_c",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{}"},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "call_a", "content": "done"},
            ],
            turn_count=3,
            transition_reason=None,
        )
        replayed_ids: list[str] = []

        async def fake_recover(
            messages: list[dict[str, Any]],
            tool_call: dict[str, Any],
            *args: Any,
            **kwargs: Any,
        ) -> str:
            replayed_ids.append(tool_call["id"])
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": "replayed",
                }
            )
            return str(tool_call["function"]["name"])

        checkpoint_count: list[int] = []
        recover_mock: AsyncMock = AsyncMock(side_effect=fake_recover)
        with patch("agent_loop.recover_tool_call", recover_mock):
            recovered: bool = await recover_pending_tool_calls(
                state,
                "session_test",
                lambda: checkpoint_count.append(1),
            )

        self.assertTrue(recovered)
        self.assertEqual(["call_b", "call_c"], replayed_ids)
        self.assertEqual(4, state.turn_count)
        self.assertEqual("tool_call", state.transition_reason)
        self.assertEqual([1], checkpoint_count)


if __name__ == "__main__":
    unittest.main()
