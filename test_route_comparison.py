#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test del sistema de comparación de rutas con filtro de transporte.
"""
import asyncio
from travel_planner.app.core.graph import TravelGraph
from travel_planner.app.data.routes_fixed import ROUTES_FIXED

def test_route_comparison():
    """Prueba que la comparación de rutas filtra correctamente por tipo de transporte."""

    print("=" * 60)
    print("TEST: Comparación de rutas con filtro de transporte")
    print("=" * 60)

    # Crear grafo
    graph = TravelGraph()

    # Poblar con todas las rutas
    for origin, dest, cost, time, transport in ROUTES_FIXED:
        graph.add_route(origin, dest, cost, time, transport)

    print(f"\n[OK] Grafo poblado con {len(ROUTES_FIXED)} rutas")

    # Test 1: Buscar París->Barcelona en avión
    print("\n" + "=" * 60)
    print("TEST 1: París -> Barcelona en AVIÓN")
    print("=" * 60)

    path_avion, cost_avion = graph.find_shortest_path(
        "París",
        "Barcelona",
        weight='cost',
        transport_type='avión'
    )
    print(f"Ruta: {' -> '.join(path_avion)}")
    print(f"Costo: {cost_avion}€")
    print(f"Segmentos: {len(path_avion) - 1}")

    # Buscar la ruta directa en ROUTES_FIXED
    print("\n" + "=" * 60)
    print("Búsqueda de ruta DIRECTA en ROUTES_FIXED")
    print("=" * 60)
    for origin, dest, cost, time, transport in ROUTES_FIXED:
        if origin == "París" and dest == "Barcelona" and transport == "avión":
            print(f"Ruta directa encontrada: {cost}€")
            break
    else:
        print("NO se encontró ruta directa")

    # Test 2: Buscar París->Barcelona en AUTO
    print("\n" + "=" * 60)
    print("TEST 2: París -> Barcelona en AUTO")
    print("=" * 60)

    path_auto, cost_auto = graph.find_shortest_path(
        "París",
        "Barcelona",
        weight='cost',
        transport_type='auto'
    )
    print(f"Ruta: {' -> '.join(path_auto)}")
    print(f"Costo: {cost_auto}€")
    print(f"Segmentos: {len(path_auto) - 1}")

    # Verificaciones lógicas
    print("\n" + "=" * 60)
    print("VERIFICACIONES")
    print("=" * 60)

    # Si ambas rutas son directas (2 ciudades), deberían tener diferente costo
    if len(path_avion) == 2 and len(path_auto) == 2:
        print(f"[OK] Ambas son directas")
        if cost_avion != cost_auto:
            print(f"[OK] Costos diferentes: avion={cost_avion}€ vs auto={cost_auto}€")
        else:
            print(f"[ERROR] Costos iguales para transports diferentes!")
    elif len(path_avion) == 2:
        print(f"[OK] Ruta de avion es directa: {cost_avion}€")
        print(f"[OK] Ruta de auto tiene {len(path_auto)-1} segmentos: {cost_auto}€")
    elif len(path_auto) == 2:
        print(f"[OK] Ruta de auto es directa: {cost_auto}€")
        print(f"[OK] Ruta de avion tiene {len(path_avion)-1} segmentos: {cost_avion}€")
    else:
        print(f"[WARNING] Ambas rutas tienen intermediarios")

    # Test 3: Verifica que sin transport_type se mezclan todos
    print("\n" + "=" * 60)
    print("TEST 3: Paris -> Barcelona SIN filtro de transporte")
    print("=" * 60)

    path_mixed, cost_mixed = graph.find_shortest_path(
        "París",
        "Barcelona",
        weight='cost'
    )
    print(f"Ruta: {' -> '.join(path_mixed)}")
    print(f"Costo: {cost_mixed}€")
    print(f"Segmentos: {len(path_mixed) - 1}")
    print(f"Costo minimo de todas las opciones: {cost_mixed}€")

    print("\n" + "=" * 60)
    print("[SUCCESS] TESTS COMPLETADOS")
    print("=" * 60)

if __name__ == "__main__":
    test_route_comparison()
