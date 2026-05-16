"""Tests para el gestor asincronico de reservas."""

import pytest
import asyncio

from app.booking.reservations import (
    ReservationManager,
    ReservationStatus,
)


def test_create_reservation_keeps_itinerary_totals():
    async def scenario():
        manager = ReservationManager(max_concurrent=2)

        reservation = await manager.create_reservation(
            user_id="user_1",
            itinerary={"origin": "Madrid", "total_cost": 123.5, "total_time": 4.25},
        )

        assert reservation.user_id == "user_1"
        assert reservation.status == ReservationStatus.PENDING
        assert reservation.total_cost == 123.5
        assert reservation.total_time == 4.25
        assert manager.get_reservation(reservation.reservation_id) is reservation

        serialized = reservation.to_dict()
        assert serialized["status"] == "pending"
        assert serialized["total_cost"] == 123.5
        assert serialized["total_time"] == 4.25

    asyncio.run(scenario())


def test_process_multiple_confirms_all_reservations():
    async def scenario():
        manager = ReservationManager(max_concurrent=3)
        reservations = [
            await manager.create_reservation(
                user_id=f"user_{idx % 2}",
                itinerary={"total_cost": 100 + idx},
            )
            for idx in range(3)
        ]

        processed = await manager.process_multiple(reservations)

        assert len(processed) == 3
        assert all(res.status == ReservationStatus.CONFIRMED for res in processed)
        assert len(manager.get_user_reservations("user_0")) == 2

        stats = manager.get_stats()
        assert stats["total_reservations"] == 3
        assert stats["by_status"]["confirmed"] == 3
        assert stats["max_concurrent"] == 3

    asyncio.run(scenario())


def test_cancel_reservation_is_irreversible():
    async def scenario():
        manager = ReservationManager()
        reservation = await manager.create_reservation(
            user_id="user_1",
            itinerary={"total_cost": 50},
        )

        assert await manager.cancel_reservation("missing-id") is False
        assert await manager.cancel_reservation(reservation.reservation_id) is True
        assert reservation.status == ReservationStatus.CANCELLED
        assert await manager.cancel_reservation(reservation.reservation_id) is False

    asyncio.run(scenario())
