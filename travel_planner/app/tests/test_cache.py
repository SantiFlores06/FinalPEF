"""Tests para caches locales usados por el planificador."""

import pytest

from app.caches import lru_cache as cache_module
from app.caches.lru_cache import LRUCache, TTLCache, lru_cache_decorator


def test_lru_cache_evicts_least_recently_used_key():
    cache = LRUCache(capacity=2)
    cache.put("ruta_1", {"cost": 10})
    cache.put("ruta_2", {"cost": 20})

    assert cache.get("ruta_1") == {"cost": 10}

    cache.put("ruta_3", {"cost": 30})

    assert cache.get("ruta_2") is None
    assert cache.get("ruta_1") == {"cost": 10}
    assert cache.get("ruta_3") == {"cost": 30}


def test_lru_cache_stats_and_invalid_capacity():
    with pytest.raises(ValueError):
        LRUCache(capacity=0)

    cache = LRUCache(capacity=2)
    cache.put("a", 1)

    assert cache.get("a") == 1
    assert cache.get("missing") is None

    stats = cache.get_stats()
    assert stats["capacity"] == 2
    assert stats["size"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5


def test_ttl_cache_expires_items(monkeypatch):
    current_time = 1000.0
    monkeypatch.setattr(cache_module.time, "time", lambda: current_time)

    cache = TTLCache(capacity=2, ttl_seconds=10)
    cache.put("temporary", "value")

    assert cache.get("temporary") == "value"

    current_time = 1011.0

    assert cache.get("temporary") is None
    assert "temporary" not in cache
    assert cache.misses == 1


def test_lru_cache_decorator_reuses_cached_result():
    calls = {"count": 0}

    @lru_cache_decorator(maxsize=4)
    def calculate(origin, destination, optimize_by="cost"):
        calls["count"] += 1
        return {"origin": origin, "destination": destination, "mode": optimize_by}

    first = calculate("Madrid", "Barcelona")
    second = calculate("Madrid", "Barcelona")
    third = calculate("Madrid", "Barcelona", optimize_by="time")

    assert first == second
    assert third["mode"] == "time"
    assert calls["count"] == 2
    assert calculate.cache_info()["hits"] == 1
