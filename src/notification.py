from __future__ import annotations

import asyncio
from typing import Dict, Iterable, List

from pydantic import BaseModel


class Notification(BaseModel):
    id: str
    content: str


class NotificationQueue:
    """A recoverable notification queue with explicit acknowledgement."""

    def __init__(self) -> None:
        self._pending: Dict[str, Notification] = {}
        self._ready: asyncio.Queue[str] = asyncio.Queue()

    def publish(self, notification: Notification) -> bool:
        """Publish a notification unless the same id is already pending."""
        if notification.id in self._pending:
            return False
        stored: Notification = notification.model_copy(deep=True)
        self._pending[stored.id] = stored
        self._ready.put_nowait(stored.id)
        return True

    async def wait_and_drain(self) -> List[Notification]:
        """Wait for one notification, then drain every item currently ready."""
        first_id: str = await self._ready.get()
        ready_ids: List[str] = [first_id]
        while True:
            try:
                ready_ids.append(self._ready.get_nowait())
            except asyncio.QueueEmpty:
                break
        return self._notifications_for(ready_ids)

    def drain_ready(self) -> List[Notification]:
        """Drain items published since the consumer last waited."""
        ready_ids: List[str] = []
        while True:
            try:
                ready_ids.append(self._ready.get_nowait())
            except asyncio.QueueEmpty:
                break
        return self._notifications_for(ready_ids)

    def acknowledge(self, notification_ids: Iterable[str]) -> None:
        for notification_id in notification_ids:
            self._pending.pop(notification_id, None)

    def snapshot(self) -> List[Notification]:
        return [item.model_copy(deep=True) for item in self._pending.values()]

    def restore(self, notifications: Iterable[Notification]) -> None:
        self._pending = {}
        self._ready = asyncio.Queue()
        for notification in notifications:
            self.publish(notification)

    def _notifications_for(self, notification_ids: Iterable[str]) -> List[Notification]:
        return [
            notification.model_copy(deep=True)
            for notification_id in notification_ids
            if (notification := self._pending.get(notification_id)) is not None
        ]


NOTIFICATION_QUEUE: NotificationQueue = NotificationQueue()
