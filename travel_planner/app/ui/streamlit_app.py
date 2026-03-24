"""
streamlit_app.py - Interfaz gráfica para el sistema de planificación de viajes.
UI interactiva construida con Streamlit.
"""

import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Set
from app.data.routes_fixed import ROUTES_FIXED
from app.ai.gemini_recommendations import generate_city_recommendations, generate_itinerary_summary


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
.main-header {
    font-size: 3rem;
    color: #1E88E5;
    text-align: center;
    padding: 1rem 0;
}
.res-card {
    background-color: #f9fafc;
    border: 1px solid #d1d9e6;
    border-radius: 16px;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.05);
    transition: transform 0.2s ease-in-out;
}
.res-card:hover {
    transform: scale(1.01);
    box-shadow: 0px 6px 15px rgba(0,0,0,0.1);
}
.res-header {
    font-size: 1.2rem;
    font-weight: bold;
    color: #1565C0;
}
.res-sub {
    font-size: 0.9rem;
    color: #555;
}
.res-badge {
    display: inline-block;
    padding: 0.25rem 0.6rem;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 600;
    color: white;
}
.status-pending { background-color: #fbc02d; }
.status-processing { background-color: #1E88E5; }
.status-confirmed { background-color: #43a047; }
.status-failed { background-color: #e53935; }
.status-cancelled { background-color: #757575; }
.success-box {
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: #F0F8FF;
    border-left: 5px solid #1E88E5;
}
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


@st.cache_data(ttl=60)
def check_api_health() -> bool:
    """Verifica si la API está disponible."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False

@st.cache_data(ttl=3600)
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

@st.cache_data(ttl=10)
def get_system_stats() -> Dict:
    """Obtiene estadísticas del sistema."""
    try:
        response = requests.get(f"{API_URL}/stats", timeout=5)
        return response.json() if response.status_code == 200 else {}
    except Exception:
        return {}


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


PAGES = ["🏠 Inicio", "🌍 Ruta Multidestino", "📋 Mis Reservas", "📊 Estadísticas"]
if "page" not in st.session_state:
    st.session_state.page = "🏠 Inicio"


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
   
    selected = st.radio(
        "Selecciona una opción:",
        PAGES,
        index=PAGES.index(st.session_state.page),
        key="nav",
    )
    st.session_state.page = selected
    st.divider()
    st.info(f"👤 Usuario: {st.session_state.user_id[:12]}...")

page = st.session_state.page

# ==========================================================
#  PÁGINA: RUTA MULTIDESTINO
# ==========================================================
if page == "🌍 Ruta Multidestino":
    st.header("🌍 Optimización de Ruta Multidestino (TSP)")
    st.write("Selecciona múltiples ciudades y compara la ruta optimizada con tu orden preferido.")

    transport_mode = st.selectbox("🚗 Tipo de transporte", ["auto", "avión", "tren"])
    optimize_by = st.selectbox("⚖️ Optimizar por", ["cost", "time"], format_func=lambda x: "Costo (€)" if x == "cost" else "Tiempo (h)")

    # Si cambia el transporte o criterio de optimización, resetear los resultados
    if st.session_state.last_transport != transport_mode or st.session_state.last_optimize_by != optimize_by:
        st.session_state.selected_cities = []
        st.session_state.tsp_result = None
        st.session_state.user_route_result = None
        st.session_state.optimized_route_result = None
        st.session_state.selected_route_for_booking = None
        st.session_state.last_transport = transport_mode
        st.session_state.last_optimize_by = optimize_by
        st.session_state.city_recommendations = {}

    # Obtener todas las ciudades disponibles para este transporte
    available_cities = get_all_cities_with_transport(transport_mode)

    if not available_cities:
        st.error(f"No hay ciudades disponibles para transporte en {transport_mode}")
        st.stop()

    # Selector múltiple de ciudades (máximo 10)
    st.subheader("📍 Seleccionar ciudades (máximo 10)")
    st.session_state.selected_cities = st.multiselect(
        "Elige las ciudades que quieres visitar:",
        available_cities,
        default=st.session_state.selected_cities,
        max_selections=10,
        key="cities_selector"
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

                    # Calcular ruta económica optimizada
                    with st.spinner("Estimando ruta económica..."):
                        optimized_result = optimize_multi_destination(
                            st.session_state.selected_cities,
                            submatrix_cost if optimize_by == "cost" else submatrix_time,
                            return_to_start
                        )

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
            with st.expander("🔍 Ver detalles de costo y tiempo"):
                total_cost_detail = 0.0
                total_time_detail = 0.0
                
                for i in range(len(user_result['route']) - 1):
                    city_from = user_result['route'][i]
                    city_to = user_result['route'][i + 1]
                    
                    segment_cost = None
                    segment_time = None
                    
                    # Obtener costo
                    if city_from in all_cities_for_cost and city_to in all_cities_for_cost:
                        idx_from = all_cities_for_cost.index(city_from)
                        idx_to = all_cities_for_cost.index(city_to)
                        cost = cost_matrix[idx_from][idx_to]
                        if cost != -1.0:
                            segment_cost = cost
                            total_cost_detail += cost
                    
                    # Obtener tiempo
                    if city_from in all_cities_for_time and city_to in all_cities_for_time:
                        idx_from = all_cities_for_time.index(city_from)
                        idx_to = all_cities_for_time.index(city_to)
                        time = time_matrix[idx_from][idx_to]
                        if time != -1.0:
                            segment_time = time
                            total_time_detail += time
                    
                    # Mostrar segmento con ambas métricas
                    if segment_cost is not None and segment_time is not None:
                        st.text(f"{i+1}. {city_from} → {city_to}: {segment_cost:.2f} € | {segment_time:.1f}h")
                    elif segment_cost is not None:
                        st.text(f"{i+1}. {city_from} → {city_to}: {segment_cost:.2f} €")
                    elif segment_time is not None:
                        st.text(f"{i+1}. {city_from} → {city_to}: {segment_time:.1f}h")
                
                st.divider()
                st.text(f"**Total: {total_cost_detail:.2f} € | {total_time_detail:.1f}h**")
            
            if st.button("✓ Seleccionar esta ruta", key="select_user_route", use_container_width=True):
                st.session_state.selected_route_for_booking = {
                    "route": user_result['route'],
                    "total_cost": user_cost_euros,
                    "type": "user_selected"
                }
                st.success("✅ Ruta seleccionada para reserva")
                st.rerun()

        # Columna 2: Ruta Económica
        with col2:
            st.markdown("### 💰 Ruta Económica (Optimizada)")
            
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
            with st.expander("🔍 Ver detalles de costo y tiempo"):
                total_cost_detail = 0.0
                total_time_detail = 0.0
                
                for i in range(len(opt_result['optimal_route']) - 1):
                    city_from = opt_result['optimal_route'][i]
                    city_to = opt_result['optimal_route'][i + 1]
                    
                    segment_cost = None
                    segment_time = None
                    
                    # Obtener costo
                    if city_from in all_cities_for_cost and city_to in all_cities_for_cost:
                        idx_from = all_cities_for_cost.index(city_from)
                        idx_to = all_cities_for_cost.index(city_to)
                        cost = cost_matrix[idx_from][idx_to]
                        if cost != -1.0:
                            segment_cost = cost
                            total_cost_detail += cost
                    
                    # Obtener tiempo
                    if city_from in all_cities_for_time and city_to in all_cities_for_time:
                        idx_from = all_cities_for_time.index(city_from)
                        idx_to = all_cities_for_time.index(city_to)
                        time = time_matrix[idx_from][idx_to]
                        if time != -1.0:
                            segment_time = time
                            total_time_detail += time
                    
                    # Mostrar segmento con ambas métricas
                    if segment_cost is not None and segment_time is not None:
                        st.text(f"{i+1}. {city_from} → {city_to}: {segment_cost:.2f} € | {segment_time:.1f}h")
                    elif segment_cost is not None:
                        st.text(f"{i+1}. {city_from} → {city_to}: {segment_cost:.2f} €")
                    elif segment_time is not None:
                        st.text(f"{i+1}. {city_from} → {city_to}: {segment_time:.1f}h")
                
                st.divider()
                st.text(f"**Total: {total_cost_detail:.2f} € | {total_time_detail:.1f}h**")
            
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
                    st.session_state.selected_cities = []
                    st.session_state.tsp_result = None
                    st.session_state.user_route_result = None
                    st.session_state.optimized_route_result = None
                    st.session_state.selected_route_for_booking = None
                    st.session_state.city_recommendations = {}
                    st.session_state.page = "📋 Mis Reservas"
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
                    st.session_state.selected_cities = []
                    st.session_state.tsp_result = None
                    st.session_state.user_route_result = None
                    st.session_state.optimized_route_result = None
                    st.session_state.selected_route_for_booking = None
                    st.session_state.city_recommendations = {}
                    st.session_state.page = "📋 Mis Reservas"
                    st.rerun()


# ==========================================================
#  INICIO
# ==========================================================
elif page == "🏠 Inicio":
    st.header("🏠 Bienvenido al Sistema de Planificación de Viajes")
    st.info("Usa el menú lateral para navegar entre las funciones disponibles.")

    st.subheader("Funcionalidades Principales")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("- **🌍 Ruta Multidestino**: Optimiza tu viaje por múltiples ciudades usando el Algoritmo del Viajante (TSP).")
        st.markdown("- **📋 Mis Reservas**: Visualiza el estado en tiempo real de todas tus reservas.")
    with col2:
        st.markdown("- **📊 Estadísticas**: Monitorea el rendimiento del sistema, el estado del caché y el procesamiento de lotes.")
        st.markdown("- **🧠 Recomendaciones IA**: Recibe sugerencias de destinos similares basadas en tu ruta calculada.")


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