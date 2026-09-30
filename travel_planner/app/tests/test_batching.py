"""Tests para el procesamiento por lotes."""

import asyncio

import pytest

from app.booking.batching import BatchProcessor, ReservationBatchProcessor
from app.booking.reservations import ReservationManager


def test_batch_processor_processes_when_batch_size_is_reached():
    async def scenario():
        processor = BatchProcessor(batch_size=2, timeout_seconds=30)

        first = processor.add_item_sync("item_1", {"value": 1})
        second = processor.add_item_sync("item_2", {"value": 2})

        results = await asyncio.wait_for(asyncio.gather(first, second), timeout=1)

        assert [result["item_id"] for result in results] == ["item_1", "item_2"]
        assert all(result["processed"] for result in results)
        assert processor.get_stats()["items_processed"] == 2
        assert processor.get_stats()["queue_size"] == 0

    asyncio.run(scenario())


def test_batch_processor_sets_exceptions_when_operation_fails():
    async def scenario():
        class FailingBatchProcessor(BatchProcessor):
            async def _batch_operation(self, batch):
                raise RuntimeError("provider unavailable")

        processor = FailingBatchProcessor(batch_size=1)
        future = processor.add_item_sync("item_1", {"value": 1})

        with pytest.raises(RuntimeError, match="provider unavailable"):
            await asyncio.wait_for(future, timeout=1)

        assert processor.get_stats()["items_failed"] == 1

    asyncio.run(scenario())


def test_reservation_batch_processor_uses_reservation_manager():
    async def scenario():
        manager = ReservationManager(max_concurrent=2)
        processor = ReservationBatchProcessor(
            batch_size=1,
            timeout_seconds=30,
            reservation_manager=manager,
        )

        future = processor.add_item_sync("user_1", {"total_cost": 200, "total_time": 3})
        result = await asyncio.wait_for(future, timeout=2)

        assert result["user_id"] == "user_1"
        assert result["status"] == "confirmed"
        assert result["total_cost"] == 200
        assert result["total_time"] == 3
        assert manager.get_stats()["total_reservations"] == 1

    asyncio.run(scenario())


def test_reservation_batch_processor_requires_manager():
    with pytest.raises(ValueError, match="ReservationManager"):
        ReservationBatchProcessor(reservation_manager=None)


def test_batch_processor_flushes_a_partial_batch_after_the_timeout():
    async def scenario():
        processor = BatchProcessor(batch_size=10, timeout_seconds=0.05)
        future = processor.add_item_sync("item_1", {"value": 1})
        await asyncio.sleep(0.1)

        await processor.trigger_processing()
        result = await asyncio.wait_for(future, timeout=1)

        assert result["item_id"] == "item_1"
        assert processor.get_stats()["queue_size"] == 0

    asyncio.run(scenario())


def test_a_full_reservation_batch_is_confirmed_quickly():
    async def scenario():
        batch_size = 20
        manager = ReservationManager(max_concurrent=batch_size)
        processor = ReservationBatchProcessor(
            batch_size=batch_size,
            timeout_seconds=30,
            reservation_manager=manager,
        )

        futures = [processor.add_item_sync("user_1", {"total_cost": 10}) for _ in range(batch_size)]
        results = await asyncio.wait_for(asyncio.gather(*futures), timeout=2)

        assert [result["status"] for result in results] == ["confirmed"] * batch_size
        assert processor.get_stats()["total_batches"] == 1

    asyncio.run(scenario())


def test_server_can_process_a_whole_batch_concurrently():
    from app.api import server

    assert server.BATCH_TIMEOUT_SECONDS <= 1
    assert server.BATCH_TICK_SECONDS <= 1
    assert server.reservation_manager.max_concurrent >= server.batch_processor.batch_size


def test_batch_processor_holds_its_background_tasks_until_they_finish():
    async def scenario():
        processor = BatchProcessor(batch_size=1, timeout_seconds=30)

        future = processor.add_item_sync("item_1", {})
        assert processor.background_tasks
        assert not any(task.done() for task in processor.background_tasks)

        await asyncio.wait_for(future, timeout=1)
        await asyncio.sleep(0.01)  # let finished tasks run their done callbacks
        assert processor.background_tasks == set()

    asyncio.run(scenario())


def test_batch_processor_records_size_and_latency_of_each_batch():
    async def scenario():
        processor = BatchProcessor(batch_size=2, timeout_seconds=30)

        futures = [processor.add_item_sync(f"item_{index}", {}) for index in range(2)]
        await asyncio.wait_for(asyncio.gather(*futures), timeout=1)

        [batch] = processor.get_stats()["recent_batches"]
        assert batch["size"] == 2
        assert 0 < batch["avg_latency_ms"] <= batch["max_latency_ms"]
        assert batch["timestamp"]

    asyncio.run(scenario())
