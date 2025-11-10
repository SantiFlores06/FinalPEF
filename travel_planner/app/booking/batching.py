"""
batching.py - Sistema de procesamiento por lotes de reservas.
Agrupa múltiples reservas para procesarlas eficientemente en batches.
"""
import asyncio
from typing import List, Dict, Any, Optional, Callable, Deque
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
import logging

# (Importar ReservationManager para type hinting)
from .reservations import ReservationManager 

# Configurar logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BatchItem:
    """Representa un item individual en un batch."""
    item_id: str
    data: Any
    future: asyncio.Future = field(default_factory=asyncio.Future)
    added_at: datetime = field(default_factory=datetime.now)

class BatchProcessor:
    """
    Procesador por lotes que agrupa items y los procesa eficientemente.
    """
    def __init__(
        self,
        batch_size: int = 10,
        timeout_seconds: float = 5.0,
        max_concurrent_batches: int = 5
    ):
        self.batch_size = batch_size
        self.timeout = timedelta(seconds=timeout_seconds)
        self.queue: Deque[BatchItem] = deque()
        self.last_processed_time = datetime.now()
        self.processing = False
        self.semaphore = asyncio.Semaphore(max_concurrent_batches)
        
        self.stats = {
            'total_items': 0,
            'total_batches': 0,
            'items_processed': 0,
            'items_failed': 0
        }

    def add_item_sync(self, item_id: str, data: Any) -> asyncio.Future:
        """
        Agrega un item al batch y retorna el future (SIN AWAIT).
        Esta función es síncrona y súper rápida.
        """
        batch_item = BatchItem(item_id=item_id, data=data)
        self.queue.append(batch_item)
        self.stats['total_items'] += 1
        
        logger.debug(f"Item {item_id} agregado a la cola ({len(self.queue)} items)")
        
        # Dispara el procesamiento, pero NO lo espera (create_task)
        # Esto permite que la función add_item termine inmediatamente.
        asyncio.create_task(self._trigger_processing())
        
        # Devuelve el future, NO lo espera
        return batch_item.future

    def _should_process(self) -> bool:
        """Verifica si se debe procesar un lote."""
        if not self.queue:
            return False
        
        queue_size = len(self.queue)
        time_since_last = datetime.now() - self.last_processed_time
        
        if queue_size >= self.batch_size:
            logger.info(f"Trigger: Lote lleno (Tamaño: {queue_size})")
            return True
        if queue_size > 0 and time_since_last >= self.timeout:
            logger.info(f"Trigger: Timeout (Cola: {queue_size}, Tiempo: {time_since_last.seconds}s)")
            return True
        return False

    async def _trigger_processing(self):
        """
        Método llamado por el servidor para verificar si procesar.
        Esta es la función que SÍ existe.
        """
        if self.processing or not self._should_process():
            return

        self.processing = True
        try:
            batch = self._extract_batch()
            if batch:
                # Inicia el procesamiento del lote sin esperar a que termine
                asyncio.create_task(self._process_batch(batch))
        finally:
            self.processing = False
            self.last_processed_time = datetime.now()

    def _extract_batch(self) -> List[BatchItem]:
        """Extrae un batch de items de la cola."""
        batch = []
        while self.queue and len(batch) < self.batch_size:
            batch.append(self.queue.popleft())
        return batch

    async def _process_batch(self, batch: List[BatchItem]) -> None:
        """Procesa un batch completo."""
        async with self.semaphore:
            self.stats['total_batches'] += 1
            batch_id = f"batch_{self.stats['total_batches']}"
            logger.info(f"Procesando {batch_id} con {len(batch)} items...")
            
            try:
                # Simular resultados
                results = await self._batch_operation(batch)
                
                # Repartir resultados a los futures
                for i, item in enumerate(batch):
                    item.future.set_result(results[i])
                    self.stats['items_processed'] += 1
                
                logger.info(f"✅ {batch_id} procesado exitosamente.")

            except Exception as e:
                logger.error(f"❌ Error procesando {batch_id}: {e}")
                for item in batch:
                    item.future.set_exception(e)
                    self.stats['items_failed'] += 1

    async def _batch_operation(self, batch: List[BatchItem]) -> List[Any]:
        """Operación real del batch (simulada por defecto)."""
        await asyncio.sleep(0.1) # Simulación de I/O
        results = []
        for item in batch:
            results.append({
                'item_id': item.item_id,
                'processed': True,
                'data': item.data,
                'timestamp': datetime.now().isoformat()
            })
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del procesador."""
        return {
            **self.stats,
            'queue_size': len(self.queue),
            'batch_size': self.batch_size,
            'processing': self.processing
        }

class ReservationBatchProcessor(BatchProcessor):
    """
    Procesador especializado para reservas de viajes.
    """
    
    def __init__(
        self,
        batch_size: int = 20,
        timeout_seconds: float = 3.0,
        max_concurrent_batches: int = 5,
        reservation_manager: 'ReservationManager' = None # <-- Aceptar el manager
    ) -> None:
        """Inicializa procesador de reservas."""
        super().__init__(batch_size, timeout_seconds, max_concurrent_batches)
        if reservation_manager is None:
            raise ValueError("ReservationBatchProcessor requiere un ReservationManager")
        self.reservation_manager = reservation_manager # <-- Guardar el manager
    
    async def _batch_operation(self, batch: List[BatchItem]) -> List[Any]:
        """
        Procesa un batch de reservas llamando al ReservationManager real.
        """
        logger.info(f"Procesando batch de {len(batch)} reservas REALES...")
        
        # Crear tareas para procesar cada reserva REAL
        tasks = []
        for item in batch:
            # Usamos el item.data (que es el itinerario) y el item.item_id (que es el user_id)
            tasks.append(
                self.process_single_reservation(item.item_id, item.data)
            )
        
        # Procesar todas las reservas del lote en paralelo
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

    async def process_single_reservation(self, user_id: str, itinerary: Dict) -> Dict:
        """
        Lógica para crear y procesar UNA reserva.
        Esto es lo que el lote ejecutará en paralelo.
        """
        try:
            # 1. Crear la reserva (guarda en memoria)
            reservation = await self.reservation_manager.create_reservation(
                user_id=user_id,
                itinerary=itinerary
            )
            # 2. Procesar la reserva (simula pago, etc.)
            await self.reservation_manager.process_reservation(reservation)
            
            # 3. Devolver el resultado confirmado
            return reservation.to_dict()
        except Exception as e:
            logger.error(f"Error en sub-proceso de batch: {e}")
            return {"status": "failed", "error": str(e)}