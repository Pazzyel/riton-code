import asyncio
from pathlib import Path
import sys
from typing import List
import unittest
from unittest.mock import AsyncMock, patch


SRC_DIR: Path = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from agent_loop import LoopState, agent_lock, notification_processor_loop  # noqa: E402
from compact import CompactState  # noqa: E402
from cron import CronJob, build_cron_notification  # noqa: E402
from notification import Notification, NotificationQueue  # noqa: E402


class NotificationQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_queue_deduplicates_snapshots_restores_and_acknowledges(self) -> None:
        queue: NotificationQueue = NotificationQueue()
        first: Notification = Notification(id="one", content="first")
        second: Notification = Notification(id="two", content="second")

        self.assertTrue(queue.publish(first))
        self.assertFalse(queue.publish(first))
        self.assertTrue(queue.publish(second))
        self.assertEqual(["one", "two"], [item.id for item in queue.snapshot()])

        batch: List[Notification] = await queue.wait_and_drain()
        self.assertEqual(["one", "two"], [item.id for item in batch])
        self.assertEqual(["one", "two"], [item.id for item in queue.snapshot()])

        queue.acknowledge(["one"])
        restored: List[Notification] = queue.snapshot()
        queue.restore(restored)
        self.assertEqual(["two"], [item.id for item in await queue.wait_and_drain()])

    async def test_processor_batches_notifications_into_one_agent_loop(self) -> None:
        queue: NotificationQueue = NotificationQueue()
        queue.publish(Notification(id="one", content="first"))
        state: LoopState = LoopState([], 0, None)
        injected: List[List[str]] = []
        answers: List[str] = []
        completed: asyncio.Event = asyncio.Event()

        def batch_callback(items: List[Notification]) -> None:
            injected.append([item.id for item in items])
            self.assertEqual([], queue.snapshot())

        def answer_callback(answer: str) -> None:
            answers.append(answer)
            completed.set()

        async def fake_agent_loop(*args: object, **kwargs: object) -> None:
            del args, kwargs
            state.messages.append({"role": "assistant", "content": "done"})

        with patch("agent_loop.agent_loop", AsyncMock(side_effect=fake_agent_loop)) as mocked:
            await agent_lock.acquire()
            try:
                processor: asyncio.Task[None] = asyncio.create_task(
                    notification_processor_loop(
                        state,
                        CompactState(),
                        "session_test",
                        queue,
                        batch_callback,
                        answer_callback,
                    )
                )
                await asyncio.sleep(0)
                queue.publish(Notification(id="two", content="second"))
            finally:
                agent_lock.release()
            await asyncio.wait_for(completed.wait(), timeout=1)
            processor.cancel()
            await asyncio.gather(processor, return_exceptions=True)

        self.assertEqual(1, mocked.await_count)
        self.assertEqual([["one", "two"]], injected)
        self.assertEqual(["first", "second"], [
            str(message["content"])
            for message in state.messages
            if message["role"] == "user"
        ])
        self.assertEqual(["done"], answers)


class NotificationProducerTests(unittest.TestCase):
    def test_cron_notification_has_stable_id_and_prefix(self) -> None:
        job: CronJob = CronJob(
            id="cron_123",
            cron="* * * * *",
            prompt="inspect the build",
            recurring=True,
            durable=True,
        )

        notification: Notification = build_cron_notification(
            job,
            "2026-09-18 10:30",
        )

        self.assertEqual("cron:cron_123:2026-09-18 10:30", notification.id)
        self.assertEqual(
            "[cron job cron_123] inspect the build",
            notification.content,
        )


if __name__ == "__main__":
    unittest.main()
