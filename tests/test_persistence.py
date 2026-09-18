import contextlib
import io
from pathlib import Path
import sqlite3
import sys
import tempfile
from typing import Any, Dict
import unittest
from unittest.mock import patch


SRC_DIR: Path = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from checkpoint import pending_tool_calls  # noqa: E402
from main import main  # noqa: E402
from persistence import (  # noqa: E402
    CHECKPOINT_VERSION,
    CheckpointRecord,
    PersistenceStore,
    one_line_message,
    set_persistence_store,
)
from checkpoint import subagent_id_for  # noqa: E402
from compact import CompactState  # noqa: E402
from notification import Notification  # noqa: E402
from background import RuntimeTaskRecord  # noqa: E402
from checkpoint import build_main_checkpoint  # noqa: E402


def checkpoint_payload(messages: list[dict[str, Any]]) -> Dict[str, Any]:
    return {
        "version": CHECKPOINT_VERSION,
        "loop_state": {
            "messages": messages,
            "turn_count": 0,
            "transition_reason": None,
        },
        "compact_state": {
            "has_compacted": False,
            "last_summary": "",
            "recent_files": [],
        },
        "background_tasks": [],
    }


class PersistenceStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()
        self.store: PersistenceStore = PersistenceStore(
            Path(self.temp_dir.name) / "db" / "agent.db"
        )
        self.session_id: str = "session_test"
        self.store.create_session(self.session_id)

    def tearDown(self) -> None:
        set_persistence_store(None)
        self.temp_dir.cleanup()

    def test_visible_messages_are_ordered_and_checkpoint_is_latest(self) -> None:
        first_payload: Dict[str, Any] = checkpoint_payload(
            [{"role": "user", "content": "first"}]
        )
        second_payload: Dict[str, Any] = checkpoint_payload(
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "answer"},
            ]
        )
        self.store.save_visible_message_and_checkpoint(
            self.session_id,
            "user",
            "first",
            self.session_id,
            "main",
            first_payload,
        )
        self.store.save_visible_message_and_checkpoint(
            self.session_id,
            "assistant",
            "answer",
            self.session_id,
            "main",
            second_payload,
            only_if_last_user=True,
        )

        roles: list[str] = [
            message.role for message in self.store.list_messages(self.session_id)
        ]
        self.assertEqual(["user", "assistant"], roles)
        record = self.store.load_checkpoint(self.session_id, self.session_id)
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(second_payload, record.payload)

    def test_visible_message_batch_and_checkpoint_are_saved_together(self) -> None:
        payload: Dict[str, Any] = checkpoint_payload(
            [
                {"role": "user", "content": "notification one"},
                {"role": "user", "content": "notification two"},
            ]
        )

        self.store.save_visible_messages_and_checkpoint(
            self.session_id,
            [
                ("user", "notification one"),
                ("user", "notification two"),
            ],
            self.session_id,
            "main",
            payload,
        )

        visible = self.store.list_messages(self.session_id)
        self.assertEqual(
            ["notification one", "notification two"],
            [message.content for message in visible],
        )
        record = self.store.load_checkpoint(self.session_id, self.session_id)
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(payload, record.payload)

    def test_parent_checkpoint_update_and_child_deletion_are_committed_together(
        self,
    ) -> None:
        child_agent_id: str = subagent_id_for(self.session_id, "call_child")
        child_payload: Dict[str, Any] = {
            "version": CHECKPOINT_VERSION,
            "context": {"messages": []},
        }
        parent_payload: Dict[str, Any] = checkpoint_payload(
            [{"role": "tool", "tool_call_id": "call_child", "content": "done"}]
        )
        self.store.save_checkpoint(
            self.session_id,
            child_agent_id,
            "subagent",
            child_payload,
        )

        self.store.save_parent_checkpoint_and_delete_child(
            self.session_id,
            self.session_id,
            "main",
            parent_payload,
            child_agent_id,
        )

        parent_record: CheckpointRecord | None = self.store.load_checkpoint(
            self.session_id,
            self.session_id,
        )
        self.assertIsNotNone(parent_record)
        assert parent_record is not None
        self.assertEqual(parent_payload, parent_record.payload)
        self.assertIsNone(self.store.load_checkpoint(self.session_id, child_agent_id))

    def test_parent_and_child_checkpoint_transaction_rolls_back_together(self) -> None:
        child_agent_id: str = subagent_id_for(self.session_id, "call_child")
        old_parent_payload: Dict[str, Any] = checkpoint_payload(
            [{"role": "assistant", "content": "old"}]
        )
        child_payload: Dict[str, Any] = {
            "version": CHECKPOINT_VERSION,
            "context": {"messages": []},
        }
        new_parent_payload: Dict[str, Any] = checkpoint_payload(
            [{"role": "tool", "tool_call_id": "call_child", "content": "done"}]
        )
        self.store.save_checkpoint(
            self.session_id,
            self.session_id,
            "main",
            old_parent_payload,
        )
        self.store.save_checkpoint(
            self.session_id,
            child_agent_id,
            "subagent",
            child_payload,
        )
        connection: sqlite3.Connection = sqlite3.connect(self.store.db_path)
        with contextlib.closing(connection), connection:
            connection.execute(
                """
                CREATE TRIGGER reject_child_checkpoint_delete
                BEFORE DELETE ON checkpoint
                WHEN OLD.agent_id = 'subagent:session_test:call_child'
                BEGIN
                    SELECT RAISE(ABORT, 'forced rollback');
                END
                """
            )

        with self.assertRaises(sqlite3.IntegrityError):
            self.store.save_parent_checkpoint_and_delete_child(
                self.session_id,
                self.session_id,
                "main",
                new_parent_payload,
                child_agent_id,
            )

        parent_record: CheckpointRecord | None = self.store.load_checkpoint(
            self.session_id,
            self.session_id,
        )
        self.assertIsNotNone(parent_record)
        assert parent_record is not None
        self.assertEqual(old_parent_payload, parent_record.payload)
        self.assertIsNotNone(
            self.store.load_checkpoint(self.session_id, child_agent_id)
        )

    def test_final_answer_is_not_duplicated(self) -> None:
        payload: Dict[str, Any] = checkpoint_payload([])
        self.store.save_visible_message_and_checkpoint(
            self.session_id,
            "user",
            "question",
            self.session_id,
            "main",
            payload,
        )
        first_insert: bool = self.store.save_visible_message_and_checkpoint(
            self.session_id,
            "assistant",
            "answer",
            self.session_id,
            "main",
            payload,
            only_if_last_user=True,
        )
        second_insert: bool = self.store.save_visible_message_and_checkpoint(
            self.session_id,
            "assistant",
            "answer",
            self.session_id,
            "main",
            payload,
            only_if_last_user=True,
        )
        self.assertTrue(first_insert)
        self.assertFalse(second_insert)
        self.assertEqual(2, len(self.store.list_messages(self.session_id)))

    def test_sessions_are_sorted_by_last_visible_message(self) -> None:
        second_session_id: str = "session_second"
        self.store.create_session(second_session_id)
        payload: Dict[str, Any] = checkpoint_payload([])
        self.store.save_visible_message_and_checkpoint(
            self.session_id,
            "user",
            "newest",
            self.session_id,
            "main",
            payload,
        )
        session_ids: list[str] = [
            session.session_id for session in self.store.list_sessions()
        ]
        self.assertEqual(self.session_id, session_ids[0])

    def test_cli_list_does_not_load_model_runtime(self) -> None:
        set_persistence_store(self.store)
        output: io.StringIO = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code: int = main(["-l"])
        self.assertEqual(0, exit_code)
        self.assertIn(self.session_id, output.getvalue())

    def test_cli_rejects_unknown_session(self) -> None:
        set_persistence_store(self.store)
        error_output: io.StringIO = io.StringIO()
        with contextlib.redirect_stderr(error_output):
            exit_code: int = main(["-s", "missing"])
        self.assertEqual(2, exit_code)
        self.assertIn("Session not found", error_output.getvalue())

    def test_cli_creates_new_session_and_exits_cleanly(self) -> None:
        set_persistence_store(self.store)
        output: io.StringIO = io.StringIO()
        with patch("builtins.input", return_value="exit"), contextlib.redirect_stdout(output):
            exit_code: int = main([])
        self.assertEqual(0, exit_code)
        created_sessions: list[str] = [
            session.session_id
            for session in self.store.list_sessions()
            if session.session_id != self.session_id
        ]
        self.assertEqual(1, len(created_sessions))
        self.assertTrue(created_sessions[0].startswith("session_"))

    def test_cli_resumes_completed_session_without_duplicate_answer(self) -> None:
        payload: Dict[str, Any] = checkpoint_payload(
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
        )
        self.store.save_visible_message_and_checkpoint(
            self.session_id,
            "user",
            "question",
            self.session_id,
            "main",
            payload,
        )
        self.store.save_visible_message_and_checkpoint(
            self.session_id,
            "assistant",
            "answer",
            self.session_id,
            "main",
            payload,
            only_if_last_user=True,
        )
        set_persistence_store(self.store)
        output: io.StringIO = io.StringIO()
        with patch("builtins.input", return_value="exit"), contextlib.redirect_stdout(output):
            exit_code: int = main(["-s", self.session_id])
        self.assertEqual(0, exit_code)
        self.assertIn("user: question", output.getvalue())
        self.assertIn("assistant: answer", output.getvalue())
        self.assertEqual(2, len(self.store.list_messages(self.session_id)))

    def test_cli_recovers_unrecorded_final_answer(self) -> None:
        user_payload: Dict[str, Any] = checkpoint_payload(
            [{"role": "user", "content": "question"}]
        )
        user_payload["pending_visible_response"] = True
        self.store.save_visible_message_and_checkpoint(
            self.session_id,
            "user",
            "question",
            self.session_id,
            "main",
            user_payload,
        )
        final_payload: Dict[str, Any] = checkpoint_payload(
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "recovered answer"},
            ]
        )
        final_payload["pending_visible_response"] = True
        self.store.save_checkpoint(
            self.session_id,
            self.session_id,
            "main",
            final_payload,
        )
        set_persistence_store(self.store)
        output: io.StringIO = io.StringIO()
        with patch("builtins.input", return_value="exit"), contextlib.redirect_stdout(output):
            exit_code: int = main(["-s", self.session_id])
        self.assertEqual(0, exit_code)
        visible_messages = self.store.list_messages(self.session_id)
        self.assertEqual(["user", "assistant"], [message.role for message in visible_messages])
        self.assertEqual("recovered answer", visible_messages[-1].content)


class CheckpointHelpersTests(unittest.TestCase):
    def test_main_checkpoint_serializes_pending_notifications(self) -> None:
        payload: Dict[str, Any] = build_main_checkpoint(
            [],
            0,
            None,
            CompactState(),
            [
                RuntimeTaskRecord(
                    id="bg_task_1",
                    agent_id="session_test",
                    tool_name="read_file",
                    tool_input={},
                    command="",
                )
            ],
            False,
            [Notification(id="notification_1", content="ready")],
        )

        self.assertEqual(
            [{"id": "notification_1", "content": "ready"}],
            payload["pending_notifications"],
        )

    def test_only_unresolved_tool_calls_are_replayed_in_order(self) -> None:
        messages: list[dict[str, Any]] = [
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
            {
                "role": "tool",
                "tool_call_id": "call_b",
                "content": "[Hook message]: not a terminal result",
            },
        ]
        pending_ids: list[str] = [
            tool_call["id"] for tool_call in pending_tool_calls(messages)
        ]
        self.assertEqual(["call_b", "call_c"], pending_ids)

    def test_final_assistant_has_no_pending_tool_calls(self) -> None:
        messages: list[dict[str, Any]] = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_a", "content": "done"},
            {"role": "assistant", "content": "final"},
        ]
        self.assertEqual([], pending_tool_calls(messages))

    def test_subagent_id_is_stable(self) -> None:
        first_id: str = subagent_id_for("session_x", "call_y")
        second_id: str = subagent_id_for("session_x", "call_y")
        self.assertEqual(first_id, second_id)
        self.assertEqual("subagent:session_x:call_y", first_id)

    def test_multiline_visible_message_is_one_physical_line(self) -> None:
        rendered: str = one_line_message("one\ntwo\\three\r")
        self.assertEqual("one\\ntwo\\\\three\\r", rendered)


if __name__ == "__main__":
    unittest.main()
