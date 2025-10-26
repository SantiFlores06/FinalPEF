"""
batching.py - Sistema de procesamiento por lotes de reservas.
Agrupa múltiples reservas para procesarlas eficientemente en batches.
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from collections import deque
import logging

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
    
    Útil para optimizar operaciones que son más eficientes en batch,
    como consultas a bases de datos, APIs externas, etc.
    
    Patrón: Acumula items durante un tiempo o hasta alcanzar un tamaño,
    luego procesa todo el batch de una vez.
    """
    
    def __init__(
        self,
        batch_size: int = 10,
        timeout_seconds: float = 2.0,
        max_concurrent_batches: int = 3
    ) -> None:
        """
        Inicializa el procesador por lotes.
        
        Args:
            batch_size: Tamaño máximo del batch antes de procesarlo.
            timeout_seconds: Tiempo máximo de espera para llenar un batch.
            max_concurrent_batches: Número máximo de batches procesándose simultáneamente.
        """
        self.batch_size = batch_size
        self.timeout = timeout_seconds
        self.max_concurrent_batches = max_concurrent_batches
        
        self.queue: deque = deque()
        self.processing = False
        self.semaphore = asyncio.Semaphore(max_concurrent_batches)
        
        self.stats = {
            'total_items': 0,
            'total_batches': 0,
            'items_processed': 0,
            'items_failed': 0
        }
    
    async def add_item(
        self,
        item_id: str,
        data: Any
    ) -> Any:
        """
        Agrega un item al batch y retorna el resultado cuando esté procesado.
        
        Args:
            item_id: Identificador único del item.
            data: Datos del item a procesar.
        
        Returns:
            Resultado del procesamiento del item.
        """
        # Crear item con future para esperar resultado
        batch_item = BatchItem(item_id=item_id, data=data)
        
        # Agregar a la cola
        self.queue.append(batch_item)
        self.stats['total_items'] += 1
        
        logger.debug(f"Item {item_id} agregado a la cola ({len(self.queue)} items)")
        
        # Iniciar procesamiento si no está corriendo
        if not self.processing:
            asyncio.create_task(self._process_batches_loop())
        
        # Esperar resultado
        return await batch_item.future
    
    async def _process_batches_loop(self) -> None:
        """Loop principal que procesa batches continuamente."""
        self.processing = True
        
        try:
            while self.queue or self.processing:
                # Esperar para acumular más items
                await asyncio.sleep(self.timeout)
                
                # Si no hay items, terminar
                if not self.queue:
                    break
                
                # Extraer batch de la cola
                batch = self._extract_batch()
                
                if batch:
                    # Procesar batch de forma asíncrona
                    asyncio.create_task(self._process_batch(batch))
        
        finally:
            self.processing = False
    
    def _extract_batch(self) -> List[BatchItem]:
        """
        Extrae un batch de items de la cola.
        
        Returns:
            Lista de items para el batch.
        """
        batch = []
        
        while self.queue and len(batch) < self.batch_size:
            batch.append(self.queue.popleft())
        
        return batch
    
    async def _process_batch(self, batch: List[BatchItem]) -> None:
        """
        Procesa un batch completo.
        
        Args:
            batch: Lista de items a procesar.
        """
        async with self.semaphore:
            self.stats['total_batches'] += 1
            batch_id = f"batch_{self.stats['total_batches']}"
            
            logger.info(f"Procesando {batch_id} con {len(batch)} items")
            
            try:
                # Simular procesamiento del batch
                results = await self._batch_operation(batch)
                
                # Asignar resultados a los futures
                for item, result in zip(batch, results):
                    if not item.future.done():
                        item.future.set_result(result)
                        self.stats['items_processed'] += 1
                
                logger.info(f"✓ {batch_id} completado exitosamente")
                
            except Exception as e:
                logger.error(f"✗ Error procesando {batch_id}: {e}")
                
                # Marcar todos los items como fallidos
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(e)
                        self.stats['items_failed'] += 1
    
    async def _batch_operation(self, batch: List[BatchItem]) -> List[Any]:
        """
        Operación real del batch (debe ser implementada/customizada).
        
        Args:
            batch: Items a procesar.
        
        Returns:
            Lista de resultados en el mismo orden.
        """
        # Simular operación costosa (ej: query a DB, API externa)
        await asyncio.sleep(0.5)
        
        # Procesar todos los items
        results = []
        for item in batch:
            # Simular procesamiento
            result = {
                'item_id': item.item_id,
                'processed': True,
                'data': item.data,
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas del procesador."""
        return {
            **self.stats,
            'queue_size': len(self.queue),
            'batch_size': self.batch_size,
            'timeout': self.timeout,
            'processing': self.processing
        }


class ReservationBatchProcessor(BatchProcessor):
    """
    Procesador especializado para reservas de viajes.
    
    Extiende BatchProcessor con lógica específica para reservas.
    """
    
    def __init__(
        self,
        batch_size: int = 20,
        timeout_seconds: float = 3.0
    ) -> None:
        """Inicializa procesador de reservas."""
        super().__init__(batch_size, timeout_seconds)
    
    async def _batch_operation(self, batch: List[BatchItem]) -> List[Any]:
        """
        Procesa un batch de reservas.
        
        Optimizaciones típicas en batch:
        - Una sola query a DB en lugar de N queries
        - Una sola llamada a API de pago
        - Notificaciones agrupadas
        """
        logger.info(f"Procesando batch de {len(batch)} reservas")
        
        # Simular validación en batch
        await self._validate_batch(batch)
        
        # Simular procesamiento de pagos en batch
        await self._process_payments_batch(batch)
        
        # Simular confirmación con proveedores en batch
        await self._confirm_providers_batch(batch)
        
        # Generar resultados
        results = []
        for item in batch:
            result = {
                'reservation_id': item.item_id,
                'status': 'confirmed',
                'itinerary': item.data,
                'confirmed_at': datetime.now().isoformat()
            }
            results.append(result)
        
        return results
    
    async def _validate_batch(self, batch: List[BatchItem]) -> None:
        """Valida todas las reservas del batch."""
        await asyncio.sleep(0.2)
        logger.debug(f"Validadas {len(batch)} reservas")
    
    async def _process_payments_batch(self, batch: List[BatchItem]) -> None:
        """Procesa pagos en batch (más eficiente)."""
        await asyncio.sleep(0.3)
        logger.debug(f"Procesados {len(batch)} pagos")
    
    async def _confirm_providers_batch(self, batch: List[BatchItem]) -> None:
        """Confirma con proveedores en batch."""
        await asyncio.sleep(0.2)
        logger.debug(f"Confirmados {len(batch)} proveedores")


class WindowedBatchProcessor:
    """
    Procesador con ventanas de tiempo fijas.
    
    Procesa batches en ventanas regulares (ej: cada 5 segundos),
    independientemente del tamaño del batch.
    """
    
    def __init__(
        self,
        window_seconds: float = 5.0,
        processor: Callable = None
    ) -> None:
        """
        Inicializa procesador con ventanas.
        
        Args:
            window_seconds: Duración de cada ventana.
            processor: Función async para procesar el batch.
        """
        self.window_seconds = window_seconds
        self.processor = processor or self._default_processor
        self.current_batch: List[Any] = []
        self.running = False
    
    async def start(self) -> None:
        """Inicia el procesador de ventanas."""
        self.running = True
        logger.info(f"Iniciando procesador con ventanas de {self.window_seconds}s")
        
        while self.running:
            await asyncio.sleep(self.window_seconds)
            
            if self.current_batch:
                batch = self.current_batch
                self.current_batch = []
                
                logger.info(f"Procesando ventana con {len(batch)} items")
                await self.processor(batch)
    
    def add(self, item: Any) -> None:
        """Agrega item a la ventana actual."""
        self.current_batch.append(item)
    
    def stop(self) -> None:
        """Detiene el procesador."""
        self.running = False
    
    async def _default_processor(self, batch: List[Any]) -> None:
        """Procesador por defecto."""
        await asyncio.sleep(0.1)
        logger.info(f"Procesados {len(batch)} items")


# Ejemplo de uso
async def main():
    """Ejemplo de uso del sistema de batching."""
    print("=" * 60)
    print("Sistema de Procesamiento por Lotes")
    print("=" * 60)
    
    # Crear procesador
    processor = ReservationBatchProcessor(
        batch_size=5,
        timeout_seconds=1.0
    )
    
    # Simular llegada de reservas
    print("\nEnviando 12 reservas...")
    tasks = []
    
    for i in range(12):
        task = processor.add_item(
            item_id=f"reservation_{i}",
            data={
                'user_id': f"user_{i % 3}",
                'route': ['Madrid', 'Barcelona'],
                'cost': 100 + (i * 10)
            }
        )
        tasks.append(task)
        
        # Simular llegada escalonada
        if i % 4 == 0:
            await asyncio.sleep(0.5)
    
    # Esperar todos los resultados
    print("Esperando resultados...")
    results = await asyncio.gather(*tasks)
    
    print(f"\n✓ Procesadas {len(results)} reservas")
    print("\nPrimeras 3 resultados:")
    for result in results[:3]:
        print(f"  - {result['reservation_id']}: {result['status']}")
    
    # Estadísticas
    stats = processor.get_stats()
    print(f"\nEstadísticas:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("Ventana de Tiempo Fija")
    print("=" * 60)
    
    # Procesador con ventanas
    windowed = WindowedBatchProcessor(window_seconds=2.0)
    
    # Iniciar en background
    asyncio.create_task(windowed.start())
    
    # Agregar items en diferentes momentos
    for i in range(8):
        windowed.add(f"item_{i}")
        await asyncio.sleep(0.7)  # Algunos en misma ventana, otros en diferentes
    
    await asyncio.sleep(2.5)  # Esperar última ventana
    windowed.stop()


if __name__ == "__main__":
    asyncio.run(main())
