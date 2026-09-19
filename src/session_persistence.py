from threading import RLock
from typing import Any, Dict, List

from background import BackgroundManager
from checkpoint import build_main_checkpoint, subagent_id_for
from notification import Notification, NotificationQueue
from persistence import PersistenceStore
from session_state import SessionContext


class SessionPersistenceCoordinator:
    """Own checkpoint construction and visible-message transactions for a session."""

    def __init__(
        self,
        store: PersistenceStore,
        session_id: str,
        context: SessionContext,
        background_manager: BackgroundManager,
        notification_queue: NotificationQueue,
    ) -> None:
        """Bind persistent storage to the mutable runtime state of one session."""
        self.store = store
        self.session_id = session_id
        self.context = context
        self.background_manager = background_manager
        self.notification_queue = notification_queue
        self._lock: RLock = RLock()

    def current_payload(self) -> Dict[str, Any]:
        """Build the latest main checkpoint from all recoverable runtime state."""
        loop_state = self.context.loop_state
        return build_main_checkpoint(
            loop_state.messages,
            loop_state.turn_count,
            loop_state.transition_reason,
            self.context.compact_state,
            self.background_manager.snapshot(),
            self.context.pending_visible_response,
            self.notification_queue.snapshot(),
        )

    def save_checkpoint(self) -> None:
        """Persist the current main checkpoint under the coordinator lock."""
        with self._lock:
            self.store.save_checkpoint(
                self.session_id,
                self.session_id,
                "main",
                self.current_payload(),
            )

    def save_parent_and_delete_subagent(self, parent_tool_call_id: Any) -> bool:
        """Atomically save the parent state and delete a completed child checkpoint."""
        if not isinstance(parent_tool_call_id, str):
            return False
        with self._lock:
            self.store.save_parent_checkpoint_and_delete_child(
                self.session_id,
                self.session_id,
                "main",
                self.current_payload(),
                subagent_id_for(self.session_id, parent_tool_call_id),
            )
        return True

    def save_completed_foreground_subagent(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
    ) -> bool:
        """Handle the tool-completion callback for a foreground subagent."""
        if tool_name != "subagent" or tool_input.get("run_in_background") is True:
            return False
        return self.save_parent_and_delete_subagent(tool_input.get("tool_call_id"))

    def save_user_message(self, content: str) -> None:
        """Append a visible user message and mark its response as pending."""
        with self._lock:
            self.context.pending_visible_response = True
            self.store.save_visible_message_and_checkpoint(
                self.session_id,
                "user",
                content,
                self.session_id,
                "main",
                self.current_payload(),
            )

    def save_notification_batch(self, notifications: List[Notification]) -> None:
        """Atomically expose a notification batch and mark its response as pending."""
        with self._lock:
            self.context.pending_visible_response = True
            self.store.save_visible_messages_and_checkpoint(
                self.session_id,
                [("user", notification.content) for notification in notifications],
                self.session_id,
                "main",
                self.current_payload(),
            )

    def save_assistant_answer(self, answer: str) -> bool:
        """Persist one visible answer and clear the pending-response marker."""
        with self._lock:
            self.context.pending_visible_response = False
            return self.store.save_visible_message_and_checkpoint(
                self.session_id,
                "assistant",
                answer,
                self.session_id,
                "main",
                self.current_payload(),
                only_if_last_user=True,
            )
