#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test del endpoint /routes/compare con filtro de transporte.
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_compare_endpoint():
    """Prueba el endpoint /routes/compare."""

    print("=" * 60)
    print("TEST: Endpoint /routes/compare")
    print("=" * 60)

    # Test 1: Paris->Barcelona en avion
    print("\n[INFO] TEST 1: Paris -> Barcelona en AVION")
    print("-" * 60)

    params = {
        "origin": "París",
        "destination": "Barcelona",
        "transport": "avión",
        "optimize_by": "cost"
    }

    try:
        response = requests.get(f"{BASE_URL}/routes/compare", params=params)
        if response.status_code == 200:
            data = response.json()
            print(json.dumps(data, indent=2, ensure_ascii=False))

            # Verificar que ambas rutas existen
            if data["direct_route"]:
                print(f"[OK] Ruta directa: {data['direct_route']['total_cost']}€")
            if data["cheapest_route"]:
                print(f"[OK] Ruta economica: {data['cheapest_route']['total_cost']}€")

            # Verificar que tienen costos diferentes si ambas son directas
            if (data["direct_route"] and len(data["direct_route"]["path"]) == 2 and
                data["cheapest_route"] and len(data["cheapest_route"]["path"]) == 2):
                if data["direct_route"]["total_cost"] != data["cheapest_route"]["total_cost"]:
                    print("[OK] Los costos son diferentes (logico para transports distintos)")
                else:
                    print("[ERROR] Los costos son iguales pero son transports diferentes!")
        else:
            print(f"[ERROR] Status: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"[ERROR] No se pudo conectar: {e}")
        print("Asegurate de que el servidor este corriendo en http://localhost:8000")
        return

    # Test 2: Paris->Barcelona en auto
    print("\n[INFO] TEST 2: Paris -> Barcelona en AUTO")
    print("-" * 60)

    params["transport"] = "auto"

    try:
        response = requests.get(f"{BASE_URL}/routes/compare", params=params)
        if response.status_code == 200:
            data = response.json()
            print(json.dumps(data, indent=2, ensure_ascii=False))
            if data["direct_route"]:
                print(f"[OK] Ruta directa: {data['direct_route']['total_cost']}€")
        else:
            print(f"[ERROR] Status: {response.status_code}")
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    test_compare_endpoint()
