"""Checks the environment and, with --api, that the running API answers correctly."""

import argparse
import importlib.util
import sys
import time

REQUIRED_PACKAGES = [
    "fastapi",
    "uvicorn",
    "streamlit",
    "pydantic",
    "pandas",
    "requests",
    "folium",
    "streamlit_folium",
    "google.genai",
    "dotenv",
    "pytest",
    "httpx",
]

API_URL = "http://127.0.0.1:8000"
HEALTH_TIMEOUT_SECONDS = 30


def check_python_version() -> bool:
    if sys.version_info < (3, 10):
        print(f"[FALLO] Python {sys.version.split()[0]}: se necesita 3.10 o superior")
        return False
    print(f"[OK] Python {sys.version.split()[0]}")
    return True


def is_installed(package: str) -> bool:
    try:
        return importlib.util.find_spec(package) is not None
    except ModuleNotFoundError:
        return False


def check_packages() -> bool:
    missing = [package for package in REQUIRED_PACKAGES if not is_installed(package)]
    if missing:
        print(f"[FALLO] Paquetes faltantes: {', '.join(missing)}")
        return False
    print(f"[OK] {len(REQUIRED_PACKAGES)} paquetes instalados")
    return True


def wait_for_api(requests_module) -> bool:
    deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            if requests_module.get(f"{API_URL}/health", timeout=2).status_code == 200:
                print("[OK] API disponible")
                return True
        except requests_module.RequestException:
            pass
        time.sleep(0.5)
    print(f"[FALLO] La API no respondio en {HEALTH_TIMEOUT_SECONDS} segundos")
    return False


def check_endpoint(requests_module, name: str, method: str, path: str, **kwargs) -> bool:
    try:
        response = requests_module.request(method, f"{API_URL}{path}", timeout=30, **kwargs)
    except requests_module.RequestException as error:
        print(f"[FALLO] {name}: {error}")
        return False
    if response.status_code != 200:
        print(f"[FALLO] {name}: HTTP {response.status_code} {response.text[:120]}")
        return False
    print(f"[OK] {name}")
    return True


def check_api() -> bool:
    import requests

    if not wait_for_api(requests):
        return False

    checks = [
        ("Matriz de costos", "GET", "/routes/matrix", {"params": {"transport": "auto"}}),
        (
            "Camino minimo (Dijkstra)",
            "POST",
            "/routes/shortest",
            {"json": {"origin": "Madrid", "destination": "Berlín", "transport_type": "auto"}},
        ),
        (
            "Ruta multidestino (TSP)",
            "POST",
            "/routes/optimize-multi",
            {
                "json": {
                    "cities": ["Madrid", "París", "Roma"],
                    "cost_matrix": [[0, 10, 15], [10, 0, 20], [15, 20, 0]],
                }
            },
        ),
    ]
    results = [check_endpoint(requests, name, method, path, **kwargs) for name, method, path, kwargs in checks]
    return all(results)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", action="store_true", help="also verify the running API")
    args = parser.parse_args()

    checks_passed = check_python_version() and check_packages()
    if checks_passed and args.api:
        checks_passed = check_api()
    return 0 if checks_passed else 1


if __name__ == "__main__":
    sys.exit(main())
