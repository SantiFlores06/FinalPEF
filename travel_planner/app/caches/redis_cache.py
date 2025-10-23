"""
redis_cache.py - Implementación de cache distribuido usando Redis.
Cache persistente ideal para sistemas con múltiples instancias/servidores.
"""

import redis
import json
import pickle
from typing import Any, Optional, Dict, Callable, List
from functools import wraps
from datetime import timedelta
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RedisCache:
    """
    Cache distribuido usando Redis.
    
    Redis permite compartir cache entre múltiples instancias de la aplicación,
    ideal para sistemas escalables con concurrencia alta.
    
    Ventajas sobre LRU local:
        - Cache compartido entre servidores
        - Persistencia opcional
        - Operaciones atómicas
        - TTL automático
    """
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ttl: int = 3600,
        prefix: str = "travel_planner:",
        decode_responses: bool = True
    ) -> None:
        """
        Inicializa conexión con Redis.
        
        Args:
            host: Host de Redis.
            port: Puerto de Redis.
            db: Base de datos de Redis (0-15).
            password: Contraseña (si está configurada).
            ttl: Time-to-live por defecto en segundos.
            prefix: Prefijo para todas las claves.
            decode_responses: Si decodificar respuestas a strings.
        """
        self.ttl = ttl
        self.prefix = prefix
        
        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=decode_responses,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Verificar conexión
            self.client.ping()
            logger.info(f"Conectado a Redis en {host}:{port}")
        except redis.ConnectionError as e:
            logger.error(f"Error conectando a Redis: {e}")
            raise
    
    def _make_key(self, key: str) -> str:
        """Crea clave con prefijo."""
        return f"{self.prefix}{key}"
    
    def get(self, key: str) -> Optional[Any]:
        """
        Obtiene valor del cache.
        
        Args:
            key: Clave a buscar.
        
        Returns:
            Valor deserializado o None si no existe.
        """
        try:
            full_key = self._make_key(key)
            value = self.client.get(full_key)
            
            if value is None:
                return None
            
            # Deserializar JSON
            return json.loads(value)
        except json.JSONDecodeError:
            logger.warning(f"Error deserializando {key}, intentando pickle")
            # Fallback a pickle para objetos complejos
            value_bytes = self.client.get(self._make_key(key))
            return pickle.loads(value_bytes) if value_bytes else None
        except Exception as e:
            logger.error(f"Error obteniendo {key}: {e}")
            return None
    
    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        nx: bool = False
    ) -> bool:
        """
        Almacena valor en cache.
        
        Args:
            key: Clave para almacenar.
            value: Valor a almacenar.
            ttl: Time-to-live en segundos (None = usar default).
            nx: Si True, solo almacena si la clave NO existe (SET NX).
        
        Returns:
            True si se almacenó correctamente.
        """
        try:
            full_key = self._make_key(key)
            ttl_seconds = ttl if ttl is not None else self.ttl
            
            # Serializar a JSON
            try:
                serialized = json.dumps(value)
            except (TypeError, ValueError):
                # Fallback a pickle para objetos no JSON-serializables
                serialized = pickle.dumps(value)
            
            if nx:
                # Solo set si no existe
                return bool(self.client.set(full_key, serialized, ex=ttl_seconds, nx=True))
            else:
                return bool(self.client.setex(full_key, ttl_seconds, serialized))
        except Exception as e:
            logger.error(f"Error almacenando {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Elimina una clave del cache.
        
        Args:
            key: Clave a eliminar.
        
        Returns:
            True si se eliminó.
        """
        try:
            full_key = self._make_key(key)
            return bool(self.client.delete(full_key))
        except Exception as e:
            logger.error(f"Error eliminando {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Verifica si una clave existe."""
        try:
            full_key = self._make_key(key)
            return bool(self.client.exists(full_key))
        except Exception as e:
            logger.error(f"Error verificando {key}: {e}")
            return False
    
    def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """
        Incrementa un contador atómicamente.
        
        Útil para rate limiting o conteo de accesos.
        
        Args:
            key: Clave del contador.
            amount: Cantidad a incrementar.
        
        Returns:
            Nuevo valor del contador.
        """
        try:
            full_key = self._make_key(key)
            return self.client.incrby(full_key, amount)
        except Exception as e:
            logger.error(f"Error incrementando {key}: {e}")
            return None
    
    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """
        Obtiene múltiples valores en una sola operación.
        
        Args:
            keys: Lista de claves.
        
        Returns:
            Diccionario con valores encontrados.
        """
        try:
            full_keys = [self._make_key(k) for k in keys]
            values = self.client.mget(full_keys)
            
            result = {}
            for key, value in zip(keys, values):
                if value:
                    try:
                        result[key] = json.loads(value)
                    except json.JSONDecodeError:
                        pass
            
            return result
        except Exception as e:
            logger.error(f"Error obteniendo múltiples claves: {e}")
            return {}
    
    def set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Almacena múltiples valores usando pipeline.
        
        Args:
            mapping: Diccionario de clave-valor.
            ttl: TTL para todas las claves.
        
        Returns:
            True si tuvo éxito.
        """
        try:
            pipeline = self.client.pipeline()
            ttl_seconds = ttl if ttl is not None else self.ttl
            
            for key, value in mapping.items():
                full_key = self._make_key(key)
                serialized = json.dumps(value)
                pipeline.setex(full_key, ttl_seconds, serialized)
            
            pipeline.execute()
            return True
        except Exception as e:
            logger.error(f"Error almacenando múltiples valores: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """
        Elimina todas las claves que coinciden con un patrón.
        
        Args:
            pattern: Patrón de Redis (e.g., "route:*").
        
        Returns:
            Número de claves eliminadas.
        """
        try:
            full_pattern = self._make_key(pattern)
            keys = self.client.keys(full_pattern)
            
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Error limpiando patrón {pattern}: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del servidor Redis.
        
        Returns:
            Diccionario con info del servidor.
        """
        try:
            info = self.client.info()
            return {
                'used_memory_human': info.get('used_memory_human'),
                'connected_clients': info.get('connected_clients'),
                'total_commands_processed': info.get('total_commands_processed'),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'hit_rate': self._calculate_hit_rate(info)
            }
        except Exception as e:
            logger.error(f"Error obteniendo stats: {e}")
            return {}
    
    def _calculate_hit_rate(self, info: Dict) -> float:
        """Calcula hit rate desde stats de Redis."""
        hits = info.get('keyspace_hits', 0)
        misses = info.get('keyspace_misses', 0)
        total = hits + misses
        return hits / total if total > 0 else 0.0
    
    def ttl_remaining(self, key: str) -> Optional[int]:
        """
        Obtiene el TTL restante de una clave.
        
        Args:
            key: Clave a verificar.
        
        Returns:
            Segundos restantes o None si no existe/no tiene TTL.
        """
        try:
            full_key = self._make_key(key)
            ttl = self.client.ttl(full_key)
            return ttl if ttl > 0 else None
        except Exception as e:
            logger.error(f"Error obteniendo TTL de {key}: {e}")
            return None
    
    def close(self) -> None:
        """Cierra la conexión con Redis."""
        try:
            self.client.close()
            logger.info("Conexión con Redis cerrada")
        except Exception as e:
            logger.error(f"Error cerrando conexión: {e}")


def redis_cache_decorator(
    cache: RedisCache,
    ttl: Optional[int] = None,
    key_prefix: str = "func"
):
    """
    Decorador para cachear resultados de funciones en Redis.
    
    Args:
        cache: Instancia de RedisCache.
        ttl: TTL específico para este cache.
        key_prefix: Prefijo para las claves generadas.
    
    Example:
        @redis_cache_decorator(redis_cache, ttl=600)
        def buscar_ruta(origen, destino):
            # Cálculo costoso
            return resultado
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Crear clave única
            cache_key = f"{key_prefix}:{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            
            # Buscar en cache
            result = cache.get(cache_key)
            if result is not None:
                logger.debug(f"Cache HIT: {cache_key}")
                return result
            
            # Calcular y cachear
            logger.debug(f"Cache MISS: {cache_key}")
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl=ttl)
            return result
        
        return wrapper
    return decorator


# Ejemplo de uso
if __name__ == "__main__":
    print("=" * 60)
    print("Prueba de RedisCache")
    print("=" * 60)
    
    # Crear instancia de cache
    try:
        cache = RedisCache(
            host='localhost',
            port=6379,
            ttl=300,
            prefix="travel_planner:"
        )
        
        # Almacenar ruta
        ruta_data = {
            "origin": "Madrid",
            "destination": "Barcelona",
            "cost": 50,
            "duration": 3,
            "transport": "tren"
        }
        
        cache.set("route:mad_bcn", ruta_data)
        print(f"\n✓ Almacenado: route:mad_bcn")
        
        # Recuperar ruta
        retrieved = cache.get("route:mad_bcn")
        print(f"✓ Recuperado: {retrieved}")
        
        # Verificar TTL
        ttl = cache.ttl_remaining("route:mad_bcn")
        print(f"✓ TTL restante: {ttl} segundos")
        
        # Almacenar múltiples rutas
        rutas = {
            "route:bcn_par": {"origin": "Barcelona", "destination": "París", "cost": 100},
            "route:par_rom": {"origin": "París", "destination": "Roma", "cost": 150}
        }
        cache.set_many(rutas, ttl=600)
        print(f"\n✓ Almacenadas {len(rutas)} rutas en batch")
        
        # Recuperar múltiples
        keys = ["route:mad_bcn", "route:bcn_par", "route:par_rom"]
        results = cache.get_many(keys)
        print(f"✓ Recuperadas {len(results)} rutas: {list(results.keys())}")
        
        # Contador atómico (útil para rate limiting)
        cache.increment("user:123:requests")
        cache.increment("user:123:requests")
        count = cache.increment("user:123:requests")
        print(f"\n✓ Contador de requests: {count}")
        
        # Estadísticas
        stats = cache.get_stats()
        print(f"\n✓ Estadísticas Redis:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Limpiar claves de prueba
        deleted = cache.clear_pattern("route:*")
        print(f"\n✓ Eliminadas {deleted} claves de rutas")
        
        cache.close()
        
    except redis.ConnectionError:
        print("\n✗ No se pudo conectar a Redis.")
        print("  Asegúrate de que Redis esté ejecutándose:")
        print("    Windows: Descarga de https://github.com/microsoftarchive/redis/releases")
        print("    Linux: sudo apt-get install redis-server")
        print("    Mac: brew install redis")
        print("  Luego ejecuta: redis-server")
