import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch


SRC_DIR: Path = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from compact import summary_messages  # noqa: E402


class SummaryMessagesTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_summary_tag_content(self) -> None:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            "<analysis>internal reasoning</analysis>\n"
                            "<summary>\nThe compacted conversation\n</summary>"
                        )
                    )
                )
            ]
        )

        with patch(
            "compact.client.chat.completions.create",
            new=AsyncMock(return_value=response),
        ):
            result = await summary_messages([])

        self.assertEqual("The compacted conversation", result)

    async def test_returns_original_content_when_summary_tag_is_missing(self) -> None:
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="plain summary"))]
        )

        with patch(
            "compact.client.chat.completions.create",
            new=AsyncMock(return_value=response),
        ):
            result = await summary_messages([])

        self.assertEqual("plain summary", result)


if __name__ == "__main__":
    unittest.main()
