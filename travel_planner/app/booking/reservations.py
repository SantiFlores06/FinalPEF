"""
reservations.py - Sistema de procesamiento asíncrono de reservas.
Maneja múltiples reservas de usuarios concurrentemente usando asyncio.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import logging
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReservationStatus(Enum):
    """Estados posibles de una reserva."""
    PENDING = "pending"
    PROCESSING = "processing"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Reservation:
    """Representa una reserva de viaje."""
    reservation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    itinerary: Dict[str, Any] = field(default_factory=dict)
    status: ReservationStatus = ReservationStatus.PENDING
    total_cost: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la reserva a diccionario."""
        return {
            'reservation_id': self.reservation_id,
            'user_id': self.user_id,
            'itinerary': self.itinerary,
            'status': self.status.value,
            'total_cost': self.total_cost,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'error_message': self.error_message
        }


class ReservationManager:
    """
    Gestor de reservas con procesamiento asíncrono y concurrente.
    
    Permite procesar múltiples reservas simultáneamente sin bloquear,
    ideal para sistemas con alta carga de usuarios.
    """
    
    def __init__(self, max_concurrent: int = 10) -> None:
        """
        Inicializa el gestor de reservas.
        
        Args:
            max_concurrent: Número máximo de reservas procesadas simultáneamente.
        """
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.reservations: Dict[str, Reservation] = {}
        self.active_tasks: List[asyncio.Task] = []
    
    async def create_reservation(
        self,
        user_id: str,
        itinerary: Dict[str, Any]
    ) -> Reservation:
        """
        Crea una nueva reserva.
        
        Args:
            user_id: ID del usuario.
            itinerary: Diccionario con detalles del itinerario.
        
        Returns:
            Objeto Reservation creado.
        """
        reservation = Reservation(
            user_id=user_id,
            itinerary=itinerary,
            total_cost=itinerary.get('total_cost', 0.0)
        )
        
        self.reservations[reservation.reservation_id] = reservation
        logger.info(f"Reserva creada: {reservation.reservation_id} para usuario {user_id}")
        
        return reservation
    
    async def process_reservation(self, reservation: Reservation) -> Reservation:
        """
        Procesa una reserva individual de forma asíncrona.
        
        Simula operaciones I/O como:
        - Verificar disponibilidad
        - Procesar pago
        - Confirmar con proveedores
        - Enviar notificaciones
        
        Args:
            reservation: Reserva a procesar.
        
        Returns:
            Reserva procesada con estado actualizado.
        """
        async with self.semaphore:
            try:
                reservation.status = ReservationStatus.PROCESSING
                reservation.updated_at = datetime.now()
                
                logger.info(f"Procesando reserva {reservation.reservation_id}")
                
                # Simular verificación de disponibilidad (I/O)
                await asyncio.sleep(0.5)
                available = await self._check_availability(reservation)
                
                if not available:
                    raise Exception("No hay disponibilidad")
                
                # Simular procesamiento de pago (I/O)
                await asyncio.sleep(0.3)
                payment_ok = await self._process_payment(reservation)
                
                if not payment_ok:
                    raise Exception("Error en el pago")
                
                # Simular confirmación con proveedores (I/O)
                await asyncio.sleep(0.4)
                await self._confirm_with_providers(reservation)
                
                # Simular envío de notificación (I/O)
                await asyncio.sleep(0.2)
                await self._send_notification(reservation)
                
                # Éxito
                reservation.status = ReservationStatus.CONFIRMED
                reservation.updated_at = datetime.now()
                logger.info(f"✓ Reserva {reservation.reservation_id} confirmada")
                
            except Exception as e:
                reservation.status = ReservationStatus.FAILED
                reservation.error_message = str(e)
                reservation.updated_at = datetime.now()
                logger.error(f"✗ Reserva {reservation.reservation_id} falló: {e}")
            
            return reservation
    
    async def _check_availability(self, reservation: Reservation) -> bool:
        """Simula verificación de disponibilidad."""
        await asyncio.sleep(0.1)
        import random
        return random.random() > 0.05
    
    async def _process_payment(self, reservation: Reservation) -> bool:
        """Simula procesamiento de pago."""
        await asyncio.sleep(0.1)
        import random
        return random.random() > 0.02
    
    async def _confirm_with_providers(self, reservation: Reservation) -> None:
        """Simula confirmación con proveedores de transporte/hoteles."""
        await asyncio.sleep(0.1)
        logger.debug(f"Confirmando con proveedores para {reservation.reservation_id}")
    
    async def _send_notification(self, reservation: Reservation) -> None:
        """Simula envío de notificación al usuario."""
        await asyncio.sleep(0.1)
        logger.debug(f"Notificación enviada para {reservation.reservation_id}")
    
    async def process_multiple(self, reservations: List[Reservation]) -> List[Reservation]:
        """
        Procesa múltiples reservas concurrentemente.
        
        Args:
            reservations: Lista de reservas a procesar.
        
        Returns:
            Lista de reservas procesadas.
        """
        logger.info(f"Procesando {len(reservations)} reservas concurrentemente")
        
        tasks = [
            self.process_reservation(reservation)
            for reservation in reservations
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error en procesamiento: {result}")
            else:
                processed.append(result)
        
        success_count = sum(1 for r in processed if r.status == ReservationStatus.CONFIRMED)
        logger.info(f"✓ {success_count}/{len(processed)} reservas confirmadas")
        
        return processed
    
    async def cancel_reservation(self, reservation_id: str) -> bool:
        """
        Cancela una reserva.
        
        Args:
            reservation_id: ID de la reserva a cancelar.
        
        Returns:
            True si se canceló exitosamente.
        """
        if reservation_id not in self.reservations:
            return False
        
        reservation = self.reservations[reservation_id]
        
        if reservation.status == ReservationStatus.CONFIRMED:
            await asyncio.sleep(0.2)
            reservation.status = ReservationStatus.CANCELLED
            reservation.updated_at = datetime.now()
            logger.info(f"Reserva {reservation_id} cancelada")
            return True
        
        return False
    
    def get_reservation(self, reservation_id: str) -> Optional[Reservation]:
        """Obtiene una reserva por ID."""
        return self.reservations.get(reservation_id)
    
    def get_user_reservations(self, user_id: str) -> List[Reservation]:
        """Obtiene todas las reservas de un usuario."""
        return [
            r for r in self.reservations.values()
            if r.user_id == user_id
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema de reservas."""
        total = len(self.reservations)
        by_status = {}
        
        for reservation in self.reservations.values():
            status = reservation.status.value
            by_status[status] = by_status.get(status, 0) + 1
        
        return {
            'total_reservations': total,
            'by_status': by_status,
            'max_concurrent': self.max_concurrent
        }


async def main():
    """Ejemplo de uso del sistema de reservas."""
    print("=" * 60)
    print("Sistema de Reservas Asíncrono")
    print("=" * 60)
    
    manager = ReservationManager(max_concurrent=5)
    
    reservations = []
    for i in range(10):
        reservation = await manager.create_reservation(
            user_id=f"user_{i % 3}",
            itinerary={
                'origin': 'Madrid',
                'destination': 'Barcelona',
                'route': ['Madrid', 'Valencia', 'Barcelona'],
                'total_cost': 150.0 + (i * 10)
            }
        )
        reservations.append(reservation)
    
    print(f"\n✓ Creadas {len(reservations)} reservas")
    
    import time
    start_time = time.time()
    
    processed = await manager.process_multiple(reservations)
    
    elapsed = time.time() - start_time
    print(f"\n Tiempo total: {elapsed:.2f}s")
    print(f"   Promedio por reserva: {elapsed/len(reservations):.2f}s")
    
    print("\nResultados:")
    for res in processed[:5]:
        print(f"  {res.reservation_id[:8]}... → {res.status.value}")
    
    stats = manager.get_stats()
    print(f"\nEstadísticas:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    if processed:
        first_id = processed[0].reservation_id
        cancelled = await manager.cancel_reservation(first_id)
        print(f"\n✓ Cancelación de {first_id[:8]}...: {cancelled}")


if __name__ == "__main__":
    asyncio.run(main())
