"""
lru_cache.py - Implementación de LRU Cache desde cero. (Last Recently Used)
Cache que elimina los elementos menos recientemente usados cuando está lleno.
"""

from collections import OrderedDict
from typing import Any, Optional, Dict, Callable
from functools import wraps
import time


class LRUCache:
    """
    Implementación de un cache LRU (Least Recently Used).
    
    Usa OrderedDict para mantener el orden de acceso. Los elementos
    más recientemente usados se mueven al final.
    
    Complejidad:
        get(): O(1)
        put(): O(1)
        
    Thread-safe: No (para uso con asyncio considerar locks)
    """
    
    def __init__(self, capacity: int = 100) -> None:
        """
        Inicializa el cache LRU.
        
        Args:
            capacity: Capacidad máxima del cache.
        
        Raises:
            ValueError: Si capacity <= 0.
        """
        if capacity <= 0:
            raise ValueError("La capacidad debe ser mayor a 0")
        
        self.cache: OrderedDict = OrderedDict()
        self.capacity = capacity
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """
        Obtiene un valor del cache y lo marca como recientemente usado.
        
        Args:
            key: Clave a buscar.
        
        Returns:
            Valor asociado a la clave, o None si no existe.
        """
        if key not in self.cache:
            self.misses += 1
            return None
        
        # Mover al final (más reciente)
        self.cache.move_to_end(key)
        self.hits += 1
        return self.cache[key]
    
    def put(self, key: str, value: Any) -> None:
        """
        Almacena un valor en el cache.
        
        Si la clave ya existe, actualiza y marca como reciente.
        Si el cache está lleno, elimina el elemento menos reciente.
        
        Args:
            key: Clave para almacenar.
            value: Valor a almacenar.
        """
        if key in self.cache:
            # Actualizar valor existente
            self.cache.move_to_end(key)
            self.cache[key] = value
        else:
            # Nuevo valor
            self.cache[key] = value
            
            # Si excede capacidad, eliminar el menos reciente (primero)
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
    
    def delete(self, key: str) -> bool:
        """
        Elimina una clave del cache.
        
        Args:
            key: Clave a eliminar.
        
        Returns:
            True si se eliminó, False si no existía.
        """
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Limpia completamente el cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def size(self) -> int:
        """Retorna el número actual de elementos en el cache."""
        return len(self.cache)
    
    def hit_rate(self) -> float:
        """
        Calcula la tasa de aciertos del cache.
        
        Returns:
            Porcentaje de hits (0.0 a 1.0).
        """
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del cache.
        
        Returns:
            Diccionario con métricas del cache.
        """
        return {
            'capacity': self.capacity,
            'size': self.size(),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hit_rate(),
            'usage_percent': (self.size() / self.capacity) * 100
        }
    
    def __contains__(self, key: str) -> bool:
        """Permite usar 'key in cache'."""
        return key in self.cache
    
    def __len__(self) -> int:
        """Permite usar len(cache)."""
        return len(self.cache)
    
    def __repr__(self) -> str:
        """Representación del cache."""
        return f"LRUCache(capacity={self.capacity}, size={self.size()}, hit_rate={self.hit_rate():.2%})"


class TTLCache(LRUCache):
    """
    Extensión de LRUCache con Time-To-Live (TTL).
    
    Los elementos expiran después de un tiempo determinado.
    """
    
    def __init__(self, capacity: int = 100, ttl_seconds: float = 3600) -> None:
        """
        Inicializa cache con TTL.
        
        Args:
            capacity: Capacidad máxima.
            ttl_seconds: Tiempo de vida en segundos.
        """
        super().__init__(capacity)
        self.ttl = ttl_seconds
        self.timestamps: OrderedDict = OrderedDict()
    
    def put(self, key: str, value: Any) -> None:
        """Almacena valor con timestamp."""
        super().put(key, value)
        self.timestamps[key] = time.time()
        if key in self.cache:
            self.timestamps.move_to_end(key)
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene valor validando TTL."""
        if key not in self.cache:
            return None
        
        # Verificar expiración
        if time.time() - self.timestamps[key] > self.ttl:
            self.delete(key)
            self.misses += 1
            return None
        
        return super().get(key)
    
    def delete(self, key: str) -> bool:
        """Elimina clave y su timestamp."""
        if key in self.timestamps:
            del self.timestamps[key]
        return super().delete(key)
    
    def clear(self) -> None:
        """Limpia cache y timestamps."""
        super().clear()
        self.timestamps.clear()
    
    def cleanup_expired(self) -> int:
        """
        Elimina todas las entradas expiradas.
        
        Returns:
            Número de elementos eliminados.
        """
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.timestamps.items()
            if current_time - timestamp > self.ttl
        ]
        
        for key in expired_keys:
            self.delete(key)
        
        return len(expired_keys)


def lru_cache_decorator(maxsize: int = 128):
    """
    Decorador para cachear resultados de funciones.
    
    Args:
        maxsize: Tamaño máximo del cache.
    
    Example:
        @lru_cache_decorator(maxsize=100)
        def expensive_function(x, y):
            return x ** y
    """
    def decorator(func: Callable) -> Callable:
        cache = LRUCache(capacity=maxsize)
        
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Crear clave única basada en argumentos
            key = str(args) + str(sorted(kwargs.items()))
            
            # Buscar en cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Calcular y almacenar
            result = func(*args, **kwargs)
            cache.put(key, result)
            return result
        
        # Agregar métodos de utilidad
        wrapper.cache = cache
        wrapper.cache_clear = cache.clear
        wrapper.cache_info = cache.get_stats
        
        return wrapper
    return decorator


# Ejemplo de uso
if __name__ == "__main__":
    print("=" * 60)
    print("Prueba de LRUCache")
    print("=" * 60)
    
    # Crear cache pequeño
    cache = LRUCache(capacity=3)
    
    # Agregar elementos
    cache.put("ruta_1", {"from": "Madrid", "to": "Barcelona", "cost": 50})
    cache.put("ruta_2", {"from": "Barcelona", "to": "París", "cost": 100})
    cache.put("ruta_3", {"from": "París", "to": "Roma", "cost": 150})
    
    print(f"\nCache después de 3 inserts: {cache}")
    print(f"Contenido: {list(cache.cache.keys())}")
    
    # Acceder a ruta_1 (la hace más reciente)
    result = cache.get("ruta_1")
    print(f"\nAcceso a ruta_1: {result}")
    
    # Agregar cuarto elemento (debería eliminar ruta_2, que es la menos reciente)
    cache.put("ruta_4", {"from": "Roma", "to": "Madrid", "cost": 200})
    print(f"\nDespués de agregar ruta_4: {list(cache.cache.keys())}")
    
    # Verificar que ruta_2 fue eliminada
    print(f"¿ruta_2 en cache?: {cache.get('ruta_2')}")
    
    # Estadísticas
    print(f"\nEstadísticas: {cache.get_stats()}")
    
    print("\n" + "=" * 60)
    print("Prueba de TTLCache")
    print("=" * 60)
    
    # Cache con TTL de 2 segundos
    ttl_cache = TTLCache(capacity=5, ttl_seconds=2)
    
    ttl_cache.put("temp_data", {"value": 42})
    print(f"\nAlmacenado temp_data: {ttl_cache.get('temp_data')}")
    
    print("Esperando 2.5 segundos...")
    time.sleep(2.5)
    
    print(f"Acceso después de expirar: {ttl_cache.get('temp_data')}")
    
    print("\n" + "=" * 60)
    print("Prueba de Decorador")
    print("=" * 60)
    
    @lru_cache_decorator(maxsize=5)
    def calcular_ruta_costosa(origen: str, destino: str) -> Dict:
        """Simula cálculo costoso de ruta."""
        print(f"  → Calculando ruta {origen} -> {destino}...")
        time.sleep(0.1)  # Simula latencia
        return {"origen": origen, "destino": destino, "costo": 100}
    
    # Primera llamada (cache miss)
    print("\nPrimera llamada:")
    result1 = calcular_ruta_costosa("Madrid", "Barcelona")
    print(f"Resultado: {result1}")
    
    # Segunda llamada (cache hit)
    print("\nSegunda llamada (mismos parámetros):")
    result2 = calcular_ruta_costosa("Madrid", "Barcelona")
    print(f"Resultado: {result2}")
    
    # Estadísticas
    print(f"\nEstadísticas del cache: {calcular_ruta_costosa.cache_info()}")
