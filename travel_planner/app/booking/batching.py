"""
batching.py - Sistema de procesamiento por lotes de reservas.
Agrupa múltiples reservas para procesarlas eficientemente en batches.
"""
import asyncio
import time
from typing import List, Dict, Any, Optional, Deque, Coroutine, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
import logging

from .reservations import ReservationManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SIMULATED_BATCH_IO_SECONDS = 0.1
MAX_BATCH_HISTORY = 200
MS_PER_SECOND = 1000


@dataclass
class BatchItem:
    """Representa un item individual en un batch."""
    item_id: str
    data: Any
    future: asyncio.Future = field(default_factory=asyncio.Future)
    enqueued_at: float = field(default_factory=time.monotonic)


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
        self.batch_history: Deque[Dict[str, Any]] = deque(maxlen=MAX_BATCH_HISTORY)
        # The event loop only keeps weak references to tasks, so hold them until they finish
        self.background_tasks: Set[asyncio.Task] = set()

        self.stats = {
            'total_items': 0,
            'total_batches': 0,
            'items_processed': 0,
            'items_failed': 0
        }

    def add_item_sync(self, item_id: str, data: Any) -> asyncio.Future:
        """
        Agrega un item al batch y retorna su future sin esperarlo.
        Dispara el procesamiento en segundo plano para que la llamada termine enseguida.
        """
        batch_item = BatchItem(item_id=item_id, data=data)
        self.queue.append(batch_item)
        self.stats['total_items'] += 1

        logger.debug(f"Item {item_id} agregado a la cola ({len(self.queue)} items)")

        self._run_in_background(self.trigger_processing())
        return batch_item.future

    def _run_in_background(self, coroutine: Coroutine[Any, Any, None]) -> None:
        """Schedule the coroutine, keeping a reference so it is not garbage-collected mid-run."""
        task = asyncio.create_task(coroutine)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)

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

    async def trigger_processing(self) -> None:
        """Extrae y lanza un lote si está lleno o si venció el timeout."""
        if self.processing or not self._should_process():
            return

        self.processing = True
        try:
            batch = self._extract_batch()
            if batch:
                self._run_in_background(self._process_batch(batch))
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
                results = await self._batch_operation(batch)
                for item, result in zip(batch, results, strict=True):
                    item.future.set_result(result)
                    self.stats['items_processed'] += 1

                logger.info(f"✅ {batch_id} procesado exitosamente.")

            except Exception as e:
                logger.error(f"❌ Error procesando {batch_id}: {e}")
                for item in batch:
                    item.future.set_exception(e)
                    self.stats['items_failed'] += 1

            self._record_batch_latency(batch)

    def _record_batch_latency(self, batch: List[BatchItem]) -> None:
        """Record the batch size and how long its items waited since they were enqueued."""
        finished_at = time.monotonic()
        latencies_ms = [(finished_at - item.enqueued_at) * MS_PER_SECOND for item in batch]
        self.batch_history.append({
            'timestamp': datetime.now().isoformat(),
            'size': len(batch),
            'avg_latency_ms': sum(latencies_ms) / len(latencies_ms),
            'max_latency_ms': max(latencies_ms),
        })

    async def _batch_operation(self, batch: List[BatchItem]) -> List[Any]:
        """Operación real del batch (simulada por defecto)."""
        await asyncio.sleep(SIMULATED_BATCH_IO_SECONDS)
        return [
            {
                'item_id': item.item_id,
                'processed': True,
                'data': item.data,
                'timestamp': datetime.now().isoformat()
            }
            for item in batch
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del procesador."""
        return {
            **self.stats,
            'queue_size': len(self.queue),
            'batch_size': self.batch_size,
            'processing': self.processing,
            'recent_batches': list(self.batch_history),
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
        reservation_manager: Optional[ReservationManager] = None
    ) -> None:
        """Inicializa procesador de reservas."""
        super().__init__(batch_size, timeout_seconds, max_concurrent_batches)
        if reservation_manager is None:
            raise ValueError("ReservationBatchProcessor requiere un ReservationManager")
        self.reservation_manager = reservation_manager

    async def _batch_operation(self, batch: List[BatchItem]) -> List[Any]:
        """
        Procesa un batch de reservas llamando al ReservationManager real.
        """
        logger.info(f"Procesando batch de {len(batch)} reservas REALES...")
        tasks = [
            self.process_single_reservation(user_id=item.item_id, itinerary=item.data)
            for item in batch
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def process_single_reservation(self, user_id: str, itinerary: Dict) -> Dict:
        """Crea y procesa una reserva; los errores se devuelven como resultado fallido."""
        try:
            reservation = await self.reservation_manager.create_reservation(
                user_id=user_id,
                itinerary=itinerary
            )
            await self.reservation_manager.process_reservation(reservation)
            return reservation.to_dict()
        except Exception as e:
            logger.error(f"Error en sub-proceso de batch: {e}")
            return {"status": "failed", "error": str(e)}
