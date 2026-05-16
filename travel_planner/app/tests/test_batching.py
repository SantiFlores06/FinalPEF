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
