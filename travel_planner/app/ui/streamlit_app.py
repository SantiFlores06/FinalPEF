"""
streamlit_app.py - Interfaz gráfica para el sistema de planificación de viajes.
UI interactiva construida con Streamlit.
"""

import sys
import os
# Agrega travel_planner/ al path para que 'app' sea importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Set
import folium
from streamlit_folium import st_folium
from app.data.routes_fixed import ROUTES_FIXED, CITIES
from app.ai.gemini_recommendations import generate_city_recommendations, generate_itinerary_summary
from app.core.tsp_genetic import GeneticTSP


# ==========================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================================

st.set_page_config(
    page_title="Travel Planner - Sistema de Planificación",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:8000"

# ==========================================================
# ESTILOS CSS PERSONALIZADOS
# ==========================================================
st.markdown(
"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

.main-header {
    font-size: 2.7rem;
    font-weight: 800;
    background: linear-gradient(135deg, #1a56db 0%, #0ea5e9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    padding: 1.2rem 0 0.3rem;
    letter-spacing: -0.5px;
    line-height: 1.2;
}

h2 { font-weight: 700 !important; color: #0f172a !important; letter-spacing: -0.2px; }
h3 { font-weight: 600 !important; color: #1e293b !important; }

.algo-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.3px;
    margin-bottom: 6px;
}
.badge-dijkstra  { background: #dbeafe; color: #1d4ed8; border: 1px solid #bfdbfe; }
.badge-held-karp { background: #d1fae5; color: #065f46; border: 1px solid #a7f3d0; }
.badge-genetic   { background: #fce7f3; color: #9d174d; border: 1px solid #fbcfe8; }

.res-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(15,23,42,0.06), 0 1px 2px rgba(15,23,42,0.04);
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}
.res-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(15,23,42,0.10);
}
.res-header { font-size: 1.05rem; font-weight: 700; color: #1e40af; }
.res-sub    { font-size: 0.85rem; color: #64748b; }

.res-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 600;
    color: white;
    letter-spacing: 0.4px;
}
.status-pending    { background: #f59e0b; }
.status-processing { background: #3b82f6; }
.status-confirmed  { background: #10b981; }
.status-failed     { background: #ef4444; }
.status-cancelled  { background: #94a3b8; }

.success-box {
    padding: 1rem 1.2rem;
    border-radius: 10px;
    background: #f0f9ff;
    border-left: 4px solid #3b82f6;
}

[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] p {
    color: #cbd5e1 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f1f5f9 !important;
}

[data-testid="stMetricValue"] { font-weight: 700 !important; }
[data-testid="stMetricLabel"] { font-size: 0.8rem !important; color: #64748b !important; }

hr { border-color: #e2e8f0 !important; }
</style>
""",
unsafe_allow_html=True
)
# ==========================================================
# FUNCIONES AUXILIARES DE API
# ==========================================================
def log_to_console(message: str, level: str = "log"):
    """Muestra un mensaje en la consola del navegador."""
    import json
    safe_message = json.dumps(message)
    st.write(f"""
    <script>
    console.{level}({safe_message});
    </script>
    """, unsafe_allow_html=True)


def get_connected_cities(from_city: str, transport_type: str) -> Set[str]:
    """
    Obtiene las ciudades conectadas desde una ciudad específica dado un tipo de transporte.
    """
    connected = set()
    for origin, destination, cost, time, transport in ROUTES_FIXED:
        if origin == from_city and transport == transport_type:
            connected.add(destination)
    return connected


def get_all_cities_with_transport(transport_type: str) -> Set[str]:
    """
    Obtiene todas las ciudades disponibles para un tipo de transporte.
    """
    cities = set()
    for origin, destination, cost, time, transport in ROUTES_FIXED:
        if transport == transport_type:
            cities.add(origin)
            cities.add(destination)
    return sorted(list(cities))


@st.cache_data(ttl=60, show_spinner=False)
def check_api_health() -> bool:
    """Verifica si la API está disponible."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False

@st.cache_data(ttl=3600, show_spinner=False)
def get_matrix_from_api(transport_mode: str, optimize_by: str) -> Optional[Dict]:
    """Obtiene matriz de costos o tiempos desde el backend."""
    try:
        resp = requests.get(
            f"{API_URL}/routes/matrix",
            params={"transport": transport_mode, "optimize_by": optimize_by},
            timeout=5,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error("Error al obtener la matriz desde el backend")
            return None
    except Exception as e:
        st.error(f"Error de conexión: {e}")
        return None


def optimize_multi_destination(
    cities: List[str],
    cost_matrix: List[List[float]],
    return_to_start: bool = True,
) -> Optional[Dict]:
    """Optimiza ruta visitando múltiples destinos (TSP)."""
    try:
        response = requests.post(
            f"{API_URL}/routes/optimize-multi",
            json={
                "cities": cities,
                "cost_matrix": cost_matrix,
                "return_to_start": return_to_start,
            },
            timeout=30,
        )
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Error optimizando ruta: {e}")
        return None


def calculate_shortest_route(
    origin: str,
    destination: str,
    transport_type: str,
    optimize_by: str,
) -> Optional[Dict]:
    """Calcula el camino mínimo entre dos ciudades usando Dijkstra."""
    try:
        response = requests.post(
            f"{API_URL}/routes/shortest",
            json={
                "origin": origin,
                "destination": destination,
                "optimize_by": optimize_by,
                "transport_type": transport_type,
            },
            timeout=10,
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error calculando ruta mínima: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error calculando ruta mínima: {e}")
        return None


def create_reservation(user_id: str, itinerary: Dict) -> Optional[Dict]:
    """Crea una nueva reserva individual."""
    try:
        response = requests.post(
            f"{API_URL}/reservations",
            json={"user_id": user_id, "itinerary": itinerary},
            timeout=10,
        )
        if response.status_code != 200:
            st.error(f"Error creando reserva: {response.text}")
            return None
        return response.json()
    except Exception as e:
        st.error(f"Error creando reserva: {e}")
        return None


def create_reservations_batch(payload: List[Dict]) -> Optional[Dict]:
    """Crea un lote de reservas."""
    try:
        response = requests.post(f"{API_URL}/reservations/batch", json=payload, timeout=20)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error creando lote: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error enviando lote: {e}")
        return None


def get_user_reservations(user_id: str) -> List[Dict]:
    """Obtiene reservas de un usuario."""
    try:
        response = requests.get(f"{API_URL}/reservations/user/{user_id}", timeout=10)
        return response.json() if response.status_code == 200 else []
    except Exception:
        return []


def cancel_reservation_api(reservation_id: str) -> bool:
    """Cancela una reserva usando la API."""
    try:
        response = requests.delete(f"{API_URL}/reservations/{reservation_id}", timeout=10)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Error al cancelar reserva: {e}")
        return False


def compare_routes(origin: str, destination: str, transport_type: str, optimize_by: str) -> Optional[Dict]:
    """Compara ruta directa vs ruta más económica."""
    try:
        response = requests.get(
            f"{API_URL}/routes/compare",
            params={
                "origin": origin,
                "destination": destination,
                "transport": transport_type,
                "optimize_by": optimize_by
            },
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error comparando rutas: {e}")
        return None

@st.cache_data(ttl=10, show_spinner=False)
def get_system_stats() -> Dict:
    """Obtiene estadísticas del sistema."""
    try:
        response = requests.get(f"{API_URL}/stats", timeout=5)
        return response.json() if response.status_code == 200 else {}
    except Exception:
        return {}


def algo_badge_html(algorithm: str, elapsed_ms: float = 0) -> str:
    configs = {
        "dijkstra":  ("🔵 Dijkstra", "badge-dijkstra"),
        "tsp":       ("🟢 Held-Karp (exacto)", "badge-held-karp"),
        "genetic":   ("🧬 Algoritmo Genético", "badge-genetic"),
    }
    label, css = configs.get(algorithm, ("Algoritmo", "badge-dijkstra"))
    time_str = f" &nbsp;·&nbsp; {elapsed_ms:.0f} ms" if elapsed_ms > 0 else ""
    return f'<span class="algo-badge {css}">{label}{time_str}</span>'


def show_city_recommendations(cities: List[str]):
    """
    Muestra recomendaciones de lugares a visitar en cada ciudad del itinerario.
    Las recomendaciones se cachean en session_state para evitar llamadas
    repetidas a la IA cada vez que Streamlit re-renderiza la página.

    Args:
        cities: Lista de ciudades en el itinerario
    """
    if not cities or len(cities) == 0:
        return

    st.subheader("🎫 Lugares que debes visitar en cada ciudad")

    cols = st.columns(len(cities))

    for idx, city in enumerate(cities):
        with cols[idx]:
            # ✅ Si ya tenemos recomendaciones cacheadas para esta ciudad, usarlas directamente
            if city in st.session_state.city_recommendations:
                recommendations = st.session_state.city_recommendations[city]
            else:
                # Solo llamar a Gemini si no están cacheadas en session_state
                with st.spinner(f"Buscando lugares en {city}..."):
                    recommendations = generate_city_recommendations(city)
                # Guardar en session_state para no volver a pedirlas en re-renders
                st.session_state.city_recommendations[city] = recommendations

            if recommendations:
                st.markdown(f"### {city}")
                st.markdown(recommendations)
            else:
                st.info(f"No se pudieron generar recomendaciones para {city}")

# ==========================================================
# SESIÓN DE USUARIO
# ==========================================================
if "user_id" not in st.session_state:
    st.session_state.user_id = "user_1"

if "tsp_result" not in st.session_state:
    st.session_state.tsp_result = None

if "selected_cities" not in st.session_state:
    st.session_state.selected_cities = []

if "clear_selected_cities" not in st.session_state:
    st.session_state.clear_selected_cities = False

if st.session_state.clear_selected_cities:
    st.session_state.selected_cities = []
    st.session_state.clear_selected_cities = False

if "last_transport" not in st.session_state:
    st.session_state.last_transport = None

if "last_optimize_by" not in st.session_state:
    st.session_state.last_optimize_by = None

if "pending_city" not in st.session_state:
    st.session_state.pending_city = None

if "route_comparison" not in st.session_state:
    st.session_state.route_comparison = None

# ✅ Cache de recomendaciones por ciudad — persiste entre re-renders de Streamlit
if "city_recommendations" not in st.session_state:
    st.session_state.city_recommendations = {}

if "user_route_result" not in st.session_state:
    st.session_state.user_route_result = None

if "optimized_route_result" not in st.session_state:
    st.session_state.optimized_route_result = None

if "selected_route_for_booking" not in st.session_state:
    st.session_state.selected_route_for_booking = None

if "ga_result" not in st.session_state:
    st.session_state.ga_result = None

if "cost_submatrix" not in st.session_state:
    st.session_state.cost_submatrix = None


PAGES = ["🏠 Inicio", "🌍 Ruta Multidestino", "📋 Mis Reservas", "📊 Estadísticas"]
if "page" not in st.session_state:
    st.session_state.page = "🏠 Inicio"

if "pending_page" in st.session_state:
    st.session_state.page = st.session_state.pending_page
    del st.session_state.pending_page


# ==========================================================
# ENCABEZADO PRINCIPAL
# ==========================================================
st.markdown(
    '<h1 class="main-header">✈️ Sistema de Planificación de Viajes Multidestino</h1>',
    unsafe_allow_html=True,
)

if not check_api_health():
    log_to_console("🔴 API no disponible. Verifica que esté ejecutándose en http://localhost:8000", "error")
    st.stop()
else:
    log_to_console("🟢 API conectada y funcionando correctamente", "log")

# ==========================================================
# SIDEBAR
# ==========================================================
with st.sidebar:
    st.header("🚀 Travel Planner")
   
    page = st.radio(
        "Selecciona una opción:",
        PAGES,
        key="page",
    )
    st.divider()
    st.info(f"👤 Usuario: {st.session_state.user_id[:12]}...")

# ==========================================================
#  PÁGINA: RUTA MULTIDESTINO
# ==========================================================
if page == "🌍 Ruta Multidestino":
    st.header("🌍 Optimización de Ruta Multidestino (TSP)")
    st.write("Selecciona múltiples ciudades y compara la ruta optimizada con tu orden preferido.")

    transport_mode = st.selectbox("🚗 Tipo de transporte", ["auto", "avión", "tren"])
    optimize_by = st.selectbox("⚖️ Optimizar por", ["cost", "time"], format_func=lambda x: "Costo (€)" if x == "cost" else "Tiempo (h)")

    # Si cambia el transporte o criterio de optimización, resetear resultados pero NO las ciudades
    if st.session_state.last_transport != transport_mode or st.session_state.last_optimize_by != optimize_by:
        st.session_state.tsp_result = None
        st.session_state.user_route_result = None
        st.session_state.optimized_route_result = None
        st.session_state.selected_route_for_booking = None
        st.session_state.last_transport = transport_mode
        st.session_state.last_optimize_by = optimize_by
        st.session_state.city_recommendations = {}
        st.session_state.ga_result = None
        st.session_state.cost_submatrix = None

    # Obtener todas las ciudades disponibles para este transporte
    available_cities = get_all_cities_with_transport(transport_mode)

    if not available_cities:
        st.error(f"No hay ciudades disponibles para transporte en {transport_mode}")
        st.stop()

    # Filtrar ciudades seleccionadas que no existan en el nuevo transporte
    if st.session_state.selected_cities:
        valid = [c for c in st.session_state.selected_cities if c in available_cities]
        if len(valid) != len(st.session_state.selected_cities):
            st.session_state.selected_cities = valid

    MAX_HELD_KARP = 12  # umbral: sobre este número de ciudades usamos AG

    # Selector múltiple de ciudades
    st.subheader("📍 Seleccionar ciudades")
    st.caption(
        f"Hasta {MAX_HELD_KARP} ciudades → Held-Karp (exacto). "
        f"Más de {MAX_HELD_KARP} → Algoritmo Genético (heurístico, sin límite práctico)."
    )
    st.multiselect(
        "Elige las ciudades que quieres visitar:",
        available_cities,
        max_selections=25,
        key="selected_cities"
    )

    col1, col2, col3 = st.columns([0.4, 0.3, 0.3])
    with col1:
        st.metric("Ciudades seleccionadas", len(st.session_state.selected_cities))
    
    with col2:
        return_to_start = st.checkbox("Regresar al origen", value=True, key="return_to_start")
    
    with col3:
        st.write("")  # Espaciador

    if len(st.session_state.selected_cities) < 2:
        st.info("⚠️ Selecciona al menos 2 ciudades para calcular costos")
    else:
        # Botón para calcular costos
        if st.button("💰 Calcular costos", type="primary", use_container_width=True):
            # Obtener AMBAS matrices (costo y tiempo)
            matrix_cost = get_matrix_from_api(transport_mode, "cost")
            matrix_time = get_matrix_from_api(transport_mode, "time")
            
            if not matrix_cost or not matrix_time:
                st.error("No se pudieron cargar los datos de las rutas. Revisa el backend.")
            else:
                all_cities = matrix_cost["cities"]
                cost_matrix = matrix_cost["matrix"]
                time_matrix = matrix_time["matrix"]

                # Verificar que todas las ciudades están en la matriz
                missing_cities = [c for c in st.session_state.selected_cities if c not in all_cities]
                if missing_cities:
                    st.error(f"❌ Las siguientes ciudades no están disponibles: {', '.join(missing_cities)}")
                else:
                    # Calcular ruta en el orden seleccionado por el usuario
                    indices_user_order = [all_cities.index(c) for c in st.session_state.selected_cities]
                    submatrix_cost = [
                        [cost_matrix[i][j] for j in indices_user_order]
                        for i in indices_user_order
                    ]
                    submatrix_time = [
                        [time_matrix[i][j] for j in indices_user_order]
                        for i in indices_user_order
                    ]

                    # Calcular costo y tiempo de ruta usuario
                    user_route_cost = 0.0
                    user_route_time = 0.0
                    user_route_path = st.session_state.selected_cities.copy()
                    
                    for i in range(len(user_route_path) - 1):
                        city_from = user_route_path[i]
                        city_to = user_route_path[i + 1]
                        idx_from = all_cities.index(city_from)
                        idx_to = all_cities.index(city_to)
                        cost = cost_matrix[idx_from][idx_to]
                        time = time_matrix[idx_from][idx_to]
                        if cost == -1.0 or time == -1.0:
                            st.error(f"❌ No hay conexión entre {city_from} y {city_to}")
                            st.stop()
                        user_route_cost += cost
                        user_route_time += time

                    # Si debe regresar al inicio
                    if return_to_start and len(user_route_path) > 1:
                        idx_last = all_cities.index(user_route_path[-1])
                        idx_first = all_cities.index(user_route_path[0])
                        cost = cost_matrix[idx_last][idx_first]
                        time = time_matrix[idx_last][idx_first]
                        if cost != -1.0 and time != -1.0:
                            user_route_cost += cost
                            user_route_time += time
                            user_route_path.append(user_route_path[0])

                    st.session_state.user_route_result = {
                        "route": user_route_path,
                        "total_cost": user_route_cost,
                        "total_time": user_route_time
                    }

                    # Guardar submatrix para uso posterior del AG
                    st.session_state.cost_submatrix = submatrix_cost if optimize_by == "cost" else submatrix_time
                    st.session_state.ga_result = None

                    # Con 2 ciudades corresponde Dijkstra; con 3 o mas corresponde TSP.
                    if len(st.session_state.selected_cities) == 2:
                        origin = st.session_state.selected_cities[0]
                        destination = st.session_state.selected_cities[1]

                        with st.spinner("Calculando camino minimo con Dijkstra..."):
                            outbound = calculate_shortest_route(
                                origin,
                                destination,
                                transport_mode,
                                optimize_by,
                            )

                            optimized_result = None
                            if outbound:
                                optimal_route = outbound["path"]
                                total_metric = outbound["total_cost"]

                                if return_to_start:
                                    inbound = calculate_shortest_route(
                                        destination,
                                        origin,
                                        transport_mode,
                                        optimize_by,
                                    )
                                    if inbound:
                                        optimal_route = optimal_route + inbound["path"][1:]
                                        total_metric += inbound["total_cost"]
                                    else:
                                        st.error("No se pudo calcular el regreso al origen")

                                optimized_result = {
                                    "optimal_route": optimal_route,
                                    "total_cost": total_metric,
                                    "computation_time": 0.0,
                                    "cached": outbound.get("cached", False),
                                    "recommendations": [],
                                    "algorithm": "dijkstra",
                                }
                    elif len(st.session_state.selected_cities) <= MAX_HELD_KARP:
                        # Held-Karp exacto (viable hasta ~12 ciudades)
                        with st.spinner("Calculando ruta óptima con Held-Karp (TSP exacto)..."):
                            optimized_result = optimize_multi_destination(
                                st.session_state.selected_cities,
                                submatrix_cost if optimize_by == "cost" else submatrix_time,
                                return_to_start
                            )
                    else:
                        # Demasiadas ciudades para Held-Karp → AG directamente
                        n_sel = len(st.session_state.selected_cities)
                        _submatrix = submatrix_cost if optimize_by == "cost" else submatrix_time
                        st.info(
                            f"Con {n_sel} ciudades, Held-Karp necesitaría 2^{n_sel} = {2**n_sel:,} estados — inviable. "
                            f"Activando Algoritmo Genético (400 generaciones)..."
                        )
                        _ga = GeneticTSP(
                            cost_matrix=_submatrix,
                            city_names=st.session_state.selected_cities,
                            population_size=200,
                            generations=400,
                            mutation_rate=0.02,
                            tournament_size=5,
                            elitism=2,
                        )
                        _ga_bar = st.progress(0, text="Generación 0 / 400")
                        def _ga_cb(gen, total, best, _bar=_ga_bar):
                            _bar.progress(gen / total, text=f"Generación {gen} / {total}  —  Mejor: {best:.2f} €")
                        _ga_cost, _ga_idx = _ga.solve(start_city=0, return_to_start=return_to_start, progress_callback=_ga_cb)
                        _ga_bar.empty()
                        _ga_route = _ga.get_route_with_names(_ga_idx)
                        optimized_result = {
                            "optimal_route": _ga_route,
                            "total_cost": _ga_cost,
                            "computation_time": _ga.elapsed_ms,
                            "cached": False,
                            "recommendations": [],
                            "algorithm": "genetic",
                            "ga_history": _ga.history,
                            "ga_elapsed_ms": _ga.elapsed_ms,
                        }

                    if optimized_result:
                        st.session_state.optimized_route_result = optimized_result
                        st.session_state.city_recommendations = {}
                        st.success("✅ Costos calculados correctamente")
                        st.rerun()
                    else:
                        st.error("No se pudo calcular la ruta optimizada")

        st.divider()

    # Mostrar resultados de cálculo de costos
    if st.session_state.user_route_result and st.session_state.optimized_route_result:
        user_result = st.session_state.user_route_result
        opt_result = st.session_state.optimized_route_result

        st.subheader("📊 Comparación de Rutas")

        col1, col2 = st.columns(2)

        # Obtener AMBAS matrices (costo y tiempo) para mostrar en detalles
        cost_matrix_data = get_matrix_from_api(transport_mode, "cost")
        time_matrix_data = get_matrix_from_api(transport_mode, "time")
        
        all_cities_for_cost = cost_matrix_data["cities"] if cost_matrix_data else []
        cost_matrix = cost_matrix_data["matrix"] if cost_matrix_data else []
        
        all_cities_for_time = time_matrix_data["cities"] if time_matrix_data else []
        time_matrix = time_matrix_data["matrix"] if time_matrix_data else []

        # Columna 1: Ruta del Usuario
        with col1:
            st.markdown("### 🎯 Tu Ruta Seleccionada")
            
            # Mostrar costo en euros siempre
            user_cost_euros = user_result['total_cost']
            if optimize_by == "time":
                # Recalcular costo en euros si se optimizó por tiempo
                user_cost_euros = 0.0
                for i in range(len(user_result['route']) - 1):
                    city_from = user_result['route'][i]
                    city_to = user_result['route'][i + 1]
                    if city_from in all_cities_for_cost and city_to in all_cities_for_cost:
                        idx_from = all_cities_for_cost.index(city_from)
                        idx_to = all_cities_for_cost.index(city_to)
                        cost = cost_matrix[idx_from][idx_to]
                        if cost != -1.0:
                            user_cost_euros += cost
            
            st.metric(
                "Costo en €",
                f"{user_cost_euros:.2f} €"
            )
            st.info(" → ".join(user_result['route']))
            
            # Detalle de costos y tiempo por segmento
            with st.expander("🔍 Ver detalles por segmento"):
                _rows_user = []
                for i in range(len(user_result['route']) - 1):
                    _cf, _ct = user_result['route'][i], user_result['route'][i + 1]
                    _sc, _st = None, None
                    if _cf in all_cities_for_cost and _ct in all_cities_for_cost:
                        _c = cost_matrix[all_cities_for_cost.index(_cf)][all_cities_for_cost.index(_ct)]
                        _sc = _c if _c != -1.0 else None
                    if _cf in all_cities_for_time and _ct in all_cities_for_time:
                        _t = time_matrix[all_cities_for_time.index(_cf)][all_cities_for_time.index(_ct)]
                        _st = _t if _t != -1.0 else None
                    _rows_user.append({"#": i + 1, "Origen": _cf, "Destino": _ct, "Costo (€)": _sc, "Tiempo (h)": _st})
                if _rows_user:
                    _df_user_seg = pd.DataFrame(_rows_user)
                    st.dataframe(
                        _df_user_seg.style
                            .format({"Costo (€)": "{:.2f}", "Tiempo (h)": "{:.1f}"}, na_rep="—")
                            .highlight_max(subset=["Costo (€)"], color="#fee2e2", axis=0)
                            .highlight_min(subset=["Costo (€)"], color="#d1fae5", axis=0),
                        use_container_width=True,
                        hide_index=True,
                    )

            if st.button("✓ Seleccionar esta ruta", key="select_user_route", use_container_width=True):
                st.session_state.selected_route_for_booking = {
                    "route": user_result['route'],
                    "total_cost": user_cost_euros,
                    "type": "user_selected"
                }
                st.success("✅ Ruta seleccionada para reserva")
                st.rerun()

        # Columna 2: Ruta Económica u Heurística (AG)
        with col2:
            _col2_label = "🧬 Ruta AG — Heurística" if opt_result.get("algorithm") == "genetic" else "💰 Ruta Económica (Optimizada)"
            st.markdown(f"### {_col2_label}")
            _badge_algo = opt_result.get("algorithm", "tsp")
            _badge_ms = opt_result.get("ga_elapsed_ms", opt_result.get("computation_time", 0))
            st.markdown(algo_badge_html(_badge_algo, _badge_ms), unsafe_allow_html=True)
            
            # Mostrar costo en euros siempre
            opt_cost_euros = opt_result['total_cost']
            if optimize_by == "time":
                # Recalcular costo en euros si se optimizó por tiempo
                opt_cost_euros = 0.0
                for i in range(len(opt_result['optimal_route']) - 1):
                    city_from = opt_result['optimal_route'][i]
                    city_to = opt_result['optimal_route'][i + 1]
                    if city_from in all_cities_for_cost and city_to in all_cities_for_cost:
                        idx_from = all_cities_for_cost.index(city_from)
                        idx_to = all_cities_for_cost.index(city_to)
                        cost = cost_matrix[idx_from][idx_to]
                        if cost != -1.0:
                            opt_cost_euros += cost
            
            st.metric(
                "Costo en €",
                f"{opt_cost_euros:.2f} €"
            )
            st.info(" → ".join(opt_result['optimal_route']))
            
            # Detalle de costos y tiempo por segmento
            with st.expander("🔍 Ver detalles por segmento"):
                _rows_opt = []
                for i in range(len(opt_result['optimal_route']) - 1):
                    _cf, _ct = opt_result['optimal_route'][i], opt_result['optimal_route'][i + 1]
                    _sc, _st = None, None
                    if _cf in all_cities_for_cost and _ct in all_cities_for_cost:
                        _c = cost_matrix[all_cities_for_cost.index(_cf)][all_cities_for_cost.index(_ct)]
                        _sc = _c if _c != -1.0 else None
                    if _cf in all_cities_for_time and _ct in all_cities_for_time:
                        _t = time_matrix[all_cities_for_time.index(_cf)][all_cities_for_time.index(_ct)]
                        _st = _t if _t != -1.0 else None
                    _rows_opt.append({"#": i + 1, "Origen": _cf, "Destino": _ct, "Costo (€)": _sc, "Tiempo (h)": _st})
                if _rows_opt:
                    _df_opt_seg = pd.DataFrame(_rows_opt)
                    st.dataframe(
                        _df_opt_seg.style
                            .format({"Costo (€)": "{:.2f}", "Tiempo (h)": "{:.1f}"}, na_rep="—")
                            .highlight_max(subset=["Costo (€)"], color="#fee2e2", axis=0)
                            .highlight_min(subset=["Costo (€)"], color="#d1fae5", axis=0),
                        use_container_width=True,
                        hide_index=True,
                    )
            
            # Calcular ahorro
            savings = user_cost_euros - opt_cost_euros
            if savings > 0:
                st.success(f"💚 Ahorras: {savings:.2f} € ({(savings/user_cost_euros*100):.1f}%)")
            elif savings < 0:
                st.warning(f"⚠️ Cuesta más: {abs(savings):.2f} €")
            else:
                st.info("✓ Mismo costo que tu ruta")
            
            if st.button("✓ Seleccionar esta ruta", key="select_opt_route", use_container_width=True):
                st.session_state.selected_route_for_booking = {
                    "route": opt_result['optimal_route'],
                    "total_cost": opt_cost_euros,
                    "type": "optimized"
                }
                st.success("✅ Ruta seleccionada para reserva")
                st.rerun()

        # ----------------------------------------------------------
        # MÉTRICAS DE AHORRO + GRÁFICO COMPARATIVO
        # ----------------------------------------------------------
        st.divider()
        st.subheader("📊 Resumen Comparativo")
        _savings = user_cost_euros - opt_cost_euros
        _savings_pct = (_savings / user_cost_euros * 100) if user_cost_euros > 0 else 0
        _label_opt_chart = "AG Heurístico" if opt_result.get("algorithm") == "genetic" else "Held-Karp (Óptimo)"

        _ms1, _ms2, _ms3 = st.columns(3)
        _ms1.metric("Tu Ruta", f"{user_cost_euros:.2f} €")
        _ms2.metric(_label_opt_chart, f"{opt_cost_euros:.2f} €",
                    delta=f"{opt_cost_euros - user_cost_euros:.2f} €", delta_color="inverse")
        _ms3.metric("Diferencia", f"{abs(_savings):.2f} €",
                    delta=f"{abs(_savings_pct):.1f}%",
                    delta_color="normal" if _savings >= 0 else "off")

        _df_chart = pd.DataFrame(
            {"Costo (€)": [user_cost_euros, opt_cost_euros]},
            index=["Tu Ruta", _label_opt_chart],
        )
        st.bar_chart(_df_chart, horizontal=True, color=["#1a56db"])

        st.divider()

        # ----------------------------------------------------------
        # MAPA INTERACTIVO DE RUTAS
        # ----------------------------------------------------------
        st.subheader("🗺️ Visualización en Mapa")

        def build_route_map(route: List[str], line_color: str, label: str, seg_costs: List[float] = None) -> folium.Map:
            coords = [(CITIES[c][0], CITIES[c][1]) for c in route if c in CITIES]
            if not coords:
                return None
            center = [sum(p[0] for p in coords) / len(coords), sum(p[1] for p in coords) / len(coords)]
            m = folium.Map(location=center, zoom_start=5, tiles="CartoDB positron")
            seen: set = set()
            stop_num = 0
            for i, city in enumerate(route):
                if city not in CITIES:
                    continue
                lat, lon = CITIES[city]
                is_start = i == 0
                is_end = i == len(route) - 1 and city == route[0]
                if is_start or is_end:
                    icon = folium.Icon(color="red", icon="home", prefix="fa")
                elif city in seen:
                    continue
                else:
                    icon = folium.Icon(color="blue", icon="circle", prefix="fa")
                next_cost = ""
                if seg_costs and i < len(seg_costs) and not is_end:
                    next_cost = f"<br><span style='color:#64748b;font-size:11px'>➡ sig. tramo: {seg_costs[i]:.0f} €</span>"
                popup_html = (
                    f"<div style='font-family:Inter,sans-serif;min-width:130px'>"
                    f"<b style='font-size:13px'>{city}</b>"
                    f"<br><span style='color:#64748b;font-size:11px'>Parada #{stop_num + 1}</span>"
                    f"{next_cost}</div>"
                )
                folium.Marker(
                    [lat, lon],
                    popup=folium.Popup(popup_html, max_width=200),
                    tooltip=f"#{stop_num + 1} {city}",
                    icon=icon,
                ).add_to(m)
                seen.add(city)
                stop_num += 1
            polyline_coords = [(CITIES[c][0], CITIES[c][1]) for c in route if c in CITIES]
            folium.PolyLine(polyline_coords, color=line_color, weight=4, opacity=0.85, tooltip=label).add_to(m)
            return m

        def _get_seg_costs(route: List[str]) -> List[float]:
            costs = []
            for i in range(len(route) - 1):
                cf, ct = route[i], route[i + 1]
                if cf in all_cities_for_cost and ct in all_cities_for_cost:
                    c = cost_matrix[all_cities_for_cost.index(cf)][all_cities_for_cost.index(ct)]
                    costs.append(c if c != -1.0 else 0.0)
                else:
                    costs.append(0.0)
            return costs

        _opt_tab_label = "🧬 Ruta AG" if opt_result.get("algorithm") == "genetic" else "💰 Ruta Optimizada (TSP)"
        map_tab1, map_tab2 = st.tabs(["🎯 Tu Ruta", _opt_tab_label])

        with map_tab1:
            m1 = build_route_map(user_result["route"], "#1a56db", "Tu ruta", _get_seg_costs(user_result["route"]))
            if m1:
                st_folium(m1, width=None, height=430, key="map_user_route", returned_objects=[])
            else:
                st.info("No se encontraron coordenadas para las ciudades seleccionadas.")

        with map_tab2:
            _opt_color = "#e53935" if opt_result.get("algorithm") == "genetic" else "#059669"
            m2 = build_route_map(opt_result["optimal_route"], _opt_color, _opt_tab_label, _get_seg_costs(opt_result["optimal_route"]))
            if m2:
                st_folium(m2, width=None, height=430, key="map_opt_route", returned_objects=[])
            else:
                st.info("No se encontraron coordenadas para las ciudades seleccionadas.")

        st.divider()

        # ----------------------------------------------------------
        # SECCIÓN DE ALGORITMO GENÉTICO
        # ----------------------------------------------------------
        n_cities = len(st.session_state.selected_cities)
        algo_used = opt_result.get("algorithm", "tsp")

        if algo_used == "genetic":
            # AG fue el solver principal (muchas ciudades)
            st.subheader("🧬 Convergencia del Algoritmo Genético")
            n_sel = n_cities
            st.caption(
                f"Con {n_sel} ciudades, Held-Karp necesitaría explorar 2^{n_sel:,} = "
                f"{2**n_sel:,} estados — computacionalmente inviable. "
                f"El AG encontró una solución en {opt_result.get('ga_elapsed_ms', 0):.0f} ms."
            )
            if "ga_history" in opt_result and opt_result["ga_history"]:
                df_hist = pd.DataFrame(opt_result["ga_history"]).set_index("generation")
                st.line_chart(df_hist[["best_cost", "avg_cost"]])
            st.divider()

        elif n_cities >= 3 and algo_used != "dijkstra" and st.session_state.cost_submatrix is not None:
            # Held-Karp fue el solver → ofrecer comparación opcional con AG
            st.subheader("⚡ Algoritmo Genético vs Held-Karp")
            st.caption(
                f"Con {n_cities} ciudades, Held-Karp es exacto y rápido. "
                "¿Querés ver cómo se compara el AG heurístico?"
            )

            if st.button("🧬 Ejecutar Algoritmo Genético", type="secondary", use_container_width=True):
                ga_solver = GeneticTSP(
                    cost_matrix=st.session_state.cost_submatrix,
                    city_names=st.session_state.selected_cities,
                    population_size=150,
                    generations=300,
                    mutation_rate=0.02,
                    tournament_size=5,
                    elitism=2,
                )
                _opt_bar = st.progress(0, text="Generación 0 / 300")
                def _opt_cb(gen, total, best, _bar=_opt_bar):
                    _bar.progress(gen / total, text=f"Generación {gen} / {total}  —  Mejor: {best:.2f} €")
                ga_cost, ga_route_idx = ga_solver.solve(start_city=0, return_to_start=return_to_start, progress_callback=_opt_cb)
                _opt_bar.empty()
                ga_route_names = ga_solver.get_route_with_names(ga_route_idx)

                st.session_state.ga_result = {
                    "cost": ga_cost,
                    "route": ga_route_names,
                    "history": ga_solver.history,
                    "elapsed_ms": ga_solver.elapsed_ms,
                }
                st.rerun()

            if st.session_state.ga_result:
                ga = st.session_state.ga_result
                hk_cost = opt_result["total_cost"]
                diff = ga["cost"] - hk_cost
                diff_pct = (diff / hk_cost * 100) if hk_cost > 0 else 0

                col_ga1, col_ga2, col_ga3 = st.columns(3)
                col_ga1.metric("AG — Mejor ruta encontrada", f"{ga['cost']:.2f} €")
                col_ga2.metric("Held-Karp (exacto)", f"{hk_cost:.2f} €")
                col_ga3.metric("Diferencia", f"{diff:+.2f} € ({diff_pct:+.1f}%)", delta_color="inverse")

                st.info(f"🗺️ Ruta AG: {' → '.join(ga['route'])}")
                st.caption(f"⏱️ AG ejecutado en {ga['elapsed_ms']:.0f} ms")

                st.markdown("**Convergencia del AG** (costo por generación)")
                df_hist = pd.DataFrame(ga["history"]).set_index("generation")
                st.line_chart(df_hist[["best_cost", "avg_cost"]])

                if diff <= 0:
                    st.success("✅ El AG encontró una ruta igual o mejor que Held-Karp.")
                elif diff_pct < 5:
                    st.info(f"ℹ️ El AG está a solo {diff_pct:.1f}% del óptimo exacto.")
                else:
                    st.warning(
                        f"⚠️ El AG está {diff_pct:.1f}% por encima del óptimo. "
                        "Más generaciones o mayor población pueden acercarlo."
                    )

                map_tab3, = st.tabs(["🧬 Ruta del AG"])
                with map_tab3:
                    m3 = build_route_map(ga["route"], "#e53935", "Ruta AG")
                    if m3:
                        st_folium(m3, width=None, height=380, key="map_ga_route", returned_objects=[])

            st.divider()

        # Mostrar recomendaciones de la IA (si hay)
        if "recommendations" in opt_result and opt_result["recommendations"]:
            st.subheader(f"🧠 IA: Basado en tu destino final ({opt_result['optimal_route'][-1]}), ¡quizás te interese!")
            rec_list = opt_result["recommendations"]
            num_cols = min(len(rec_list), 4)
            if num_cols > 0:
                cols = st.columns(num_cols)
                for i, rec in enumerate(rec_list):
                    if i < num_cols:
                        with cols[i]:
                            st.button(f"📍 {rec['destination_name']}",
                                      help=f"Similitud: {rec['similarity']:.2f}",
                                      key=f"rec_compare_{i}",
                                      use_container_width=True)
            st.divider()

        # Mostrar recomendaciones de lugares por ciudad
        if st.session_state.selected_route_for_booking:
            selected_route = st.session_state.selected_route_for_booking["route"]
            show_city_recommendations(selected_route)
            st.divider()

    # Sección de reserva (solo si hay una ruta seleccionada)
    if st.session_state.selected_route_for_booking:
        selected = st.session_state.selected_route_for_booking
        
        st.subheader("🎟️ Realizar Reserva")
        st.info(f"**Ruta seleccionada:** {' → '.join(selected['route'])}")
        st.info(f"**Costo total:** {selected['total_cost']:.2f} €")

        num_tickets = st.number_input("Cantidad de pasajes", min_value=1, max_value=20, value=1, step=1)

        if st.button("📝 Crear reserva", type="secondary", use_container_width=True):
            # El costo ya está en euros gracias al cálculo anterior
            real_cost = selected['total_cost']

            itinerary = {
                "type": "multidestino",
                "transport_mode": transport_mode,
                "optimize_by": optimize_by,
                "cities": st.session_state.selected_cities,
                "optimal_route": selected['route'],
                "route_type": selected['type'],
                "total_cost": real_cost,
                "total_time": None,
            }

            if num_tickets == 1:
                with st.spinner("Creando reserva..."):
                    reservation = create_reservation(st.session_state.user_id, itinerary)
                if reservation:
                    st.success(f"Reserva creada correctamente ✅ ID: {reservation['reservation_id'][:12]}...")
                    st.balloons()
                    # Limpiar todo
                    st.session_state.clear_selected_cities = True
                    st.session_state.tsp_result = None
                    st.session_state.user_route_result = None
                    st.session_state.optimized_route_result = None
                    st.session_state.selected_route_for_booking = None
                    st.session_state.city_recommendations = {}
                    st.session_state.pending_page = "📋 Mis Reservas"
                    st.rerun()
            else:
                batch_payload = [
                    {"user_id": st.session_state.user_id, "itinerary": itinerary}
                    for _ in range(num_tickets)
                ]
                with st.spinner(f"Enviando lote de {num_tickets} reservas..."):
                    response = create_reservations_batch(batch_payload)
                if response:
                    st.success(f"🧩 Lote creado correctamente ({response['count']} reservas en cola)")
                    st.balloons()
                    # Limpiar todo
                    st.session_state.clear_selected_cities = True
                    st.session_state.tsp_result = None
                    st.session_state.user_route_result = None
                    st.session_state.optimized_route_result = None
                    st.session_state.selected_route_for_booking = None
                    st.session_state.city_recommendations = {}
                    st.session_state.pending_page = "📋 Mis Reservas"
                    st.rerun()


# ==========================================================
#  INICIO
# ==========================================================
elif page == "🏠 Inicio":
    st.markdown("## Bienvenido al Planificador de Viajes Europeo")
    st.markdown("Optimizá rutas entre **53 ciudades de Europa**, compará algoritmos de optimización y realizá reservas — todo en un solo lugar.")
    st.divider()

    _fc1, _fc2, _fc3, _fc4 = st.columns(4)
    _fc1.info("**🌍 Ruta Multidestino**\n\nOptimización TSP con Dijkstra, Held-Karp y Algoritmos Genéticos.")
    _fc2.info("**📋 Mis Reservas**\n\nEstado en tiempo real de reservas individuales y en lote.")
    _fc3.info("**📊 Estadísticas**\n\nCaché LRU, rendimiento y procesamiento batch asíncrono.")
    _fc4.info("**🧠 IA Gemini**\n\nRecomendaciones personalizadas de destinos y lugares.")

    st.divider()
    st.markdown(f"#### 🗺️ {len(CITIES)} ciudades disponibles en Europa")
    _m_home = folium.Map(location=[50.5, 10.0], zoom_start=4, tiles="CartoDB positron")
    for _city, (_lat, _lon) in CITIES.items():
        folium.CircleMarker(
            location=[_lat, _lon],
            radius=6,
            color="#1a56db",
            fill=True,
            fill_color="#1a56db",
            fill_opacity=0.75,
            tooltip=_city,
            popup=folium.Popup(f"<b>{_city}</b>", max_width=120),
        ).add_to(_m_home)
    st_folium(_m_home, width=None, height=430, key="home_city_map", returned_objects=[])


# ==========================================================
#  MIS RESERVAS
# ==========================================================
elif page == "📋 Mis Reservas":
    st.header("📋 Mis Reservas")

    reservations = get_user_reservations(st.session_state.user_id)
    if not reservations:
        st.info("No tienes reservas registradas.")
    else:
        st.info(f"Mostrando {len(reservations)} reservas para el usuario {st.session_state.user_id[:12]}...")

        for r in sorted(reservations, key=lambda x: x.get('created_at', ''), reverse=True):
            itinerary = r.get("itinerary", {})
            optimal_route = itinerary.get("optimal_route", [])
            status = r.get("status", "pending")
            reservation_id = r.get('reservation_id')

            status_class = (
                "status-confirmed" if status == "confirmed"
                else "status-failed" if status == "failed"
                else "status-cancelled" if status == "cancelled"
                else "status-processing" if status == "processing"
                else "status-pending"
            )

            col_card, col_action = st.columns([0.80, 0.20])

            with col_card:
                st.markdown(f"""
                <div class="res-card">
                    <div class="res-header">🧾 Reserva #{r.get('reservation_id')[:8]}...</div>
                    <div class="res-sub">Creada: {r.get('created_at', '').split('.')[0].replace('T', ' a las ')}</div>
                    <br>
                    <b>🧭 Tipo:</b> {itinerary.get('type', 'N/A').capitalize()} <br>
                    <b>🚗 Transporte:</b> {itinerary.get('transport_mode', 'N/A')} <br>
                    <b>🗺️ Ruta óptima:</b> {" → ".join(optimal_route)} <br>
                    <b>💰 Total:</b> {itinerary.get('total_cost', 0):.2f} € <br><br>
                    <span class="res-badge {status_class}">Estado: {status.upper()}</span>
                </div>
                """, unsafe_allow_html=True)

            with col_action:
                if status != "cancelled":
                    if st.button("❌ Cancelar", key=f"cancel_{reservation_id}", use_container_width=True):
                        with st.spinner("Cancelando reserva..."):
                            success = cancel_reservation_api(reservation_id)

                        if success:
                            st.success("✅ Reserva cancelada exitosamente")
                            st.rerun()
                        else:
                            st.error("Error al cancelar la reserva")

                    st.markdown("""
                    <style>
                    button[kind="secondary"] {
                        padding-top: 120px !important;
                        padding-bottom: 120px !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)

# ==========================================================
#  ESTADÍSTICAS
# ==========================================================
elif page == "📊 Estadísticas":
    st.header("📊 Estadísticas del Sistema")

    if st.button("Actualizar Estadísticas"):
        st.cache_data.clear()

    stats = get_system_stats()
    if not stats:
        st.warning("No hay estadísticas disponibles actualmente.")
        st.stop()

    cache = stats.get("cache", {})
    reservations = stats.get("reservations", {})
    batch = stats.get("batch_processor", {})

    st.subheader("📦 Resumen General")
    col1, col2, col3 = st.columns(3)
    col1.metric("🧠 Total de Reservas", reservations.get("total_reservations", 0))
    by_status = reservations.get("by_status", {})
    col2.metric("❌ Canceladas", by_status.get("cancelled", 0) + by_status.get("processing", 0))
    st.divider()

    st.subheader("🧩 Estado del Caché (Rutas y TSP)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Capacidad", cache.get("capacity", 0))
    col2.metric("Items Cacheados", cache.get("size", 0))
    hit_rate = cache.get("hit_rate", 0) * 100
    col3.metric("Tasa de Aciertos", f"{hit_rate:.1f}%")

    usage = cache.get("usage_percent", 0)
    st.progress(min(usage / 100, 1.0), text=f"Uso actual del caché: {usage:.1f}%")
    st.divider()

    st.subheader("🧾 Estado de Reservas")
    if by_status:
        filtered_status = {k: v for k, v in by_status.items() if v > 0}
        if filtered_status:
            df_status = pd.DataFrame(filtered_status.items(), columns=["Estado", "Cantidad"])
            st.bar_chart(df_status.set_index("Estado"))
        else:
            st.info("No hay datos de estado de reservas.")
    else:
        st.info("No hay datos de estado de reservas.")
    st.divider()

    st.subheader("⚙️ Procesamiento Batch (Reservas en Cola)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Items Totales Recibidos", batch.get("total_items", 0))
    col2.metric("Batches Procesados", batch.get("total_batches", 0))
    col3.metric("Items en Cola Ahora", batch.get("queue_size", 0))

    processing = "✅ Procesando" if batch.get("processing") else "🟡 En espera"
    st.info(f"**Estado actual del procesador:** {processing}")
    st.caption(f"Última actualización: {stats.get('timestamp', '').split('.')[0].replace('T', ' a las ')}")
