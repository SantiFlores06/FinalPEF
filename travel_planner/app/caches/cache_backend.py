"""
cache_backend.py - Selección del backend de caché con fallback automático.

Elige entre Redis (caché distribuido) y LRU en memoria según la configuración
por variables de entorno. Si se pide Redis pero no está disponible (paquete
ausente o servidor caído), cae automáticamente al LRU local para que la
aplicación nunca se rompa por un problema de caché.

Interfaz pública común de ambos backends:
    get(key)        -> Optional[Any]
    put(key, value) -> None | bool
    get_stats()     -> dict
"""

import os
import logging

from app.caches.lru_cache import LRUCache

logger = logging.getLogger(__name__)


def get_cache_backend(capacity: int = 100):
    """
    Devuelve el backend de caché configurado.

    Variables de entorno relevantes:
        CACHE_BACKEND=redis  -> intenta usar Redis (cualquier otro valor usa LRU).
        REDIS_HOST           -> host de Redis (default: localhost).
        REDIS_PORT           -> puerto de Redis (default: 6379).
        REDIS_DB             -> base de datos (default: 0).
        REDIS_PASSWORD       -> contraseña opcional.

    Si CACHE_BACKEND != "redis", o si la conexión con Redis falla, devuelve un
    LRUCache(capacity). Nunca lanza excepción por indisponibilidad de Redis.

    Args:
        capacity: Capacidad del LRU cuando se usa como backend o fallback.

    Returns:
        Una instancia de RedisCache o LRUCache. Ambas exponen get/put/get_stats.
    """
    backend = os.getenv("CACHE_BACKEND", "lru").strip().lower()

    if backend == "redis":
        try:
            # Import diferido: redis_cache importa el paquete 'redis' al cargarse.
            # Hacerlo aquí adentro permite tratar "paquete ausente" y "servidor
            # caído" como un único camino de fallback.
            from app.caches.redis_cache import RedisCache

            cache = RedisCache(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", "6379")),
                db=int(os.getenv("REDIS_DB", "0")),
                password=os.getenv("REDIS_PASSWORD") or None,
            )
            logger.info("Backend de caché: Redis")
            return cache
        except Exception as e:
            logger.warning(
                "No se pudo inicializar Redis (%s). "
                "Usando LRU en memoria como fallback.",
                e,
            )

    logger.info("Backend de caché: LRU en memoria (capacidad=%d)", capacity)
    return LRUCache(capacity=capacity)
