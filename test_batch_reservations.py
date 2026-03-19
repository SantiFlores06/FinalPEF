#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test del sistema de reservas sin fallos aleatorios.
"""
import asyncio
from travel_planner.app.booking.reservations import ReservationManager, ReservationStatus

async def test_batch():
    manager = ReservationManager(max_concurrent=5)

    print("=" * 60)
    print("TEST: Crear y procesar 5 reservas sin fallos aleatorios")
    print("=" * 60)

    reservations = []
    for i in range(5):
        res = await manager.create_reservation(
            user_id=f'user_{i}',
            itinerary={'origin': 'Madrid', 'destination': 'Barcelona', 'total_cost': 100 + i*10}
        )
        reservations.append(res)

    print(f'[OK] Creadas {len(reservations)} reservas')

    # Procesar todas
    processed = await manager.process_multiple(reservations)
    print(f'[OK] Procesadas {len(processed)} reservas')

    # Verificar estados
    confirmed_count = sum(1 for r in processed if r.status == ReservationStatus.CONFIRMED)
    failed_count = sum(1 for r in processed if r.status == ReservationStatus.FAILED)

    print(f'[OK] CONFIRMED: {confirmed_count}')
    print(f'[ERROR] FAILED: {failed_count}')

    if failed_count == 0:
        print('\n[SUCCESS] Todas las reservas quedaron CONFIRMED')
    else:
        print(f'\n[FAILURE] {failed_count} reservas fallaron')

    # Probar cancelación
    print("\n" + "=" * 60)
    print("TEST: Cancelar una reserva")
    print("=" * 60)

    first_id = processed[0].reservation_id
    print(f'Cancelando {first_id[:8]}...')
    cancelled = await manager.cancel_reservation(first_id)

    if cancelled:
        res_check = manager.get_reservation(first_id)
        if res_check.status == ReservationStatus.CANCELLED:
            print(f'[OK] Reserva cancelada exitosamente')
        else:
            print(f'[ERROR] Cancelacion registrada pero estado incorrecto: {res_check.status.value}')
    else:
        print('[ERROR] Fallo al cancelar')

    # Probar cancelación doble (debe fallar)
    print("\n" + "=" * 60)
    print("TEST: Intentar cancelar una reserva ya cancelada (debe fallar)")
    print("=" * 60)

    double_cancel = await manager.cancel_reservation(first_id)
    if not double_cancel:
        print('[OK] Correctamente rechazada la cancelacion doble')
    else:
        print('[ERROR] Se permitio cancelacion doble (BUG)')

if __name__ == "__main__":
    asyncio.run(test_batch())
