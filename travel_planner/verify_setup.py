"""
verify_setup.py - Verifica que el entorno esté configurado correctamente.
"""

import sys
import importlib

def check_imports():
    """Verifica que todas las dependencias se puedan importar."""
    required_packages = [
        'fastapi',
        'uvicorn',
        'streamlit',
        'redis',
        'pandas',
        'numpy',
        'sklearn',
        'pytest',
        'aiohttp',
        'pydantic'
    ]
    
    print("Verificando dependencias...")
    all_ok = True
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NO INSTALADO")
            all_ok = False
    
    return all_ok

def check_python_version():
    """Verifica la versión de Python."""
    print(f"\nPython version: {sys.version}")
    if sys.version_info < (3, 9):
        print("⚠️  Se recomienda Python 3.9 o superior")
        return False
    print("✓ Versión de Python adecuada")
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("Verificación del Entorno de Desarrollo")
    print("=" * 50)
    
    python_ok = check_python_version()
    imports_ok = check_imports()
    
    print("\n" + "=" * 50)
    if python_ok and imports_ok:
        print("✓ Entorno configurado correctamente")
        sys.exit(0)
    else:
        print("✗ Hay problemas con el entorno")
        sys.exit(1)
