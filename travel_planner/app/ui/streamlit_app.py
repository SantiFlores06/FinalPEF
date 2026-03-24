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
    st.image("https://raw.githubusercontent.com/streamlit/demo-travel-app/main/assets/road-trip.jpeg", use_column_width=True)

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
    st.write("Encuentra el orden óptimo para visitar múltiples ciudades minimizando costo o tiempo.")

    transport_mode = st.selectbox("🚗 Tipo de transporte", ["auto", "avión", "tren"])
    optimize_by = st.selectbox("⚖️ Optimizar por", ["cost", "time"], format_func=lambda x: "Costo (€)" if x == "cost" else "Tiempo (h)")

    # Si cambia el transporte o criterio de optimización, resetear los resultados
    if st.session_state.last_transport != transport_mode or st.session_state.last_optimize_by != optimize_by:
        st.session_state.selected_cities = []
        st.session_state.tsp_result = None
        st.session_state.last_transport = transport_mode
        st.session_state.last_optimize_by = optimize_by
        # ✅ Limpiar caché de recomendaciones al cambiar transporte/criterio
        st.session_state.city_recommendations = {}

    # Obtener todas las ciudades disponibles para este transporte
    available_cities = get_all_cities_with_transport(transport_mode)

    if not available_cities:
        st.error(f"No hay ciudades disponibles para transporte en {transport_mode}")
        st.stop()

    # Selector secuencial de ciudades
    st.subheader("📍 Seleccionar ciudades en orden")

    col1, col2 = st.columns([0.8, 0.2])

    with col1:
        # Primera ciudad: mostrar todas disponibles
        if len(st.session_state.selected_cities) == 0:
            first_city = st.selectbox(
                "Primera ciudad (origen):",
                available_cities,
                key="first_city"
            )
            if st.button("✓ Agregar primera ciudad", key="add_first_city", use_container_width=True):
                st.session_state.selected_cities = [first_city]
                st.rerun()

        # Siguientes ciudades: mostrar solo las conectadas desde la última
        elif len(st.session_state.selected_cities) < 10:
            last_city = st.session_state.selected_cities[-1]
            connected_cities = get_connected_cities(last_city, transport_mode)

            if not connected_cities:
                st.warning(f"❌ No hay ciudades conectadas desde {last_city} en {transport_mode}")
            else:
                available_for_next = sorted(list(connected_cities))
                next_city = st.selectbox(
                    f"Siguiente ciudad (desde {last_city}):",
                    available_for_next,
                    key=f"next_city_{len(st.session_state.selected_cities)}"
                )
                if st.button("✓ Ver opciones de ruta", key="compare_routes_btn", use_container_width=True):
                    st.session_state.pending_city = next_city
                    comparison = compare_routes(last_city, next_city, transport_mode, optimize_by)
                    st.session_state.route_comparison = comparison
                    st.rerun()

    with col2:
        st.metric("Ciudades", len(st.session_state.selected_cities))

    # Mostrar comparación de rutas si está disponible
    if st.session_state.route_comparison:
        st.divider()
        comparison = st.session_state.route_comparison

        st.subheader("🛣️ Opciones de conexión")

        if "return_to_start_comparison" not in st.session_state:
            st.session_state.return_to_start_comparison = True

        st.checkbox(
            "Regresar a la ciudad de inicio al final",
            value=True,
            key="return_to_start_comparison"
        )

        col1, col2 = st.columns(2)

        # Opción 1: Ruta Directa
        with col1:
            if comparison["direct_exists"]:
                st.markdown("### 🎯 Ruta Directa")
                direct = comparison["direct_route"]
                st.metric(
                    "Costo/Tiempo",
                    f"{direct['total_cost']:.2f} {'€' if optimize_by == 'cost' else 'h'}"
                )
                st.caption(f"Segmentos: {len(direct['path']) - 1}")
                st.caption(direct["description"])

                if st.button("✓ Usar ruta directa", key="select_direct", use_container_width=True):
                    st.session_state.selected_cities.append(st.session_state.pending_city)
                    st.session_state.route_comparison = None
                    st.session_state.pending_city = None

                    if len(st.session_state.selected_cities) >= 2:
                        matrix_data = get_matrix_from_api(transport_mode, optimize_by)
                        if matrix_data:
                            all_cities = matrix_data["cities"]
                            cost_matrix = matrix_data["matrix"]

                            if all(c in all_cities for c in st.session_state.selected_cities):
                                indices = [all_cities.index(c) for c in st.session_state.selected_cities]
                                submatrix = [
                                    [cost_matrix[i][j] for j in indices]
                                    for i in indices
                                ]
                                result = optimize_multi_destination(st.session_state.selected_cities, submatrix, st.session_state.return_to_start_comparison)
                                if result:
                                    st.session_state.tsp_result = result
                                    # ✅ Limpiar caché de recomendaciones al calcular nueva ruta
                                    st.session_state.city_recommendations = {}

                    st.rerun()
            else:
                st.markdown("### ⛔ Ruta Directa")
                st.caption("No hay conexión directa disponible")

        # Opción 2: Ruta más Económica
        with col2:
            st.markdown("### 💰 Ruta Más Económica")
            cheapest = comparison["cheapest_route"]
            st.metric(
                "Costo/Tiempo",
                f"{cheapest['total_cost']:.2f} {'€' if optimize_by == 'cost' else 'h'}"
            )
            st.caption(f"Segmentos: {len(cheapest['path']) - 1}")
            st.caption(cheapest["description"])

            if comparison["savings"]:
                st.info(f"💚 **Ahorras:** {comparison['savings']:.2f} {'€' if optimize_by == 'cost' else 'h'}")

            if len(cheapest["path"]) > 2:
                st.caption(f"Vía: {' → '.join(cheapest['path'])}")

            if st.button("✓ Usar ruta económica", key="select_cheapest", use_container_width=True):
                for city in cheapest["path"]:
                    if city not in st.session_state.selected_cities:
                        st.session_state.selected_cities.append(city)

                st.session_state.route_comparison = None
                st.session_state.pending_city = None

                if len(st.session_state.selected_cities) >= 2:
                    matrix_data = get_matrix_from_api(transport_mode, optimize_by)
                    if matrix_data:
                        all_cities = matrix_data["cities"]
                        cost_matrix = matrix_data["matrix"]

                        if all(c in all_cities for c in st.session_state.selected_cities):
                            indices = [all_cities.index(c) for c in st.session_state.selected_cities]
                            submatrix = [
                                [cost_matrix[i][j] for j in indices]
                                for i in indices
                            ]
                            result = optimize_multi_destination(st.session_state.selected_cities, submatrix, st.session_state.return_to_start_comparison)
                            if result:
                                st.session_state.tsp_result = result
                                # ✅ Limpiar caché de recomendaciones al calcular nueva ruta
                                st.session_state.city_recommendations = {}

                st.rerun()

        if st.button("❌ Cancelar selección", key="cancel_comparison", use_container_width=True):
            st.session_state.route_comparison = None
            st.session_state.pending_city = None
            st.rerun()

        st.divider()

    # Obtener matriz y calcular ruta (solo si hay al menos 2 ciudades Y no hay resultado calculado)
    if len(st.session_state.selected_cities) >= 2 and not st.session_state.tsp_result:
        matrix_data = get_matrix_from_api(transport_mode, optimize_by)
        if not matrix_data:
            st.error("No se pudieron cargar los datos de las rutas. Revisa el backend.")
            st.stop()

        all_cities = matrix_data["cities"]
        cost_matrix = matrix_data["matrix"]

        missing_cities = [c for c in st.session_state.selected_cities if c not in all_cities]
        if missing_cities:
            st.error(f"❌ Las siguientes ciudades no están disponibles en la matriz: {', '.join(missing_cities)}")
            st.info("💡 Esto puede ocurrir si una ciudad no tiene rutas de conexión. Intenta seleccionar otra ciudad.")
            st.stop()

        return_to_start = st.checkbox("Regresar al origen", value=True)

        if st.button("🧠 Calcular Ruta Óptima", type="primary", use_container_width=True):
            indices = [all_cities.index(c) for c in st.session_state.selected_cities]
            submatrix = [
                [cost_matrix[i][j] for j in indices]
                for i in indices
            ]

            st.write("📊 Matriz de valores desde backend:")
            df = pd.DataFrame(submatrix, index=st.session_state.selected_cities, columns=st.session_state.selected_cities)
            st.dataframe(df.style.format("{:.2f}"))

            with st.spinner("Calculando mejor ruta... (IA incluida)"):
                result = optimize_multi_destination(st.session_state.selected_cities, submatrix, return_to_start)
            if result:
                st.session_state.tsp_result = result
                # ✅ Limpiar caché de recomendaciones al calcular nueva ruta
                st.session_state.city_recommendations = {}
                st.success("✅ Ruta óptima encontrada")
            else:
                st.error("No se pudo calcular la ruta.")

    # Mostrar resultado guardado (permite crear reserva)
    if st.session_state.tsp_result:
        result = st.session_state.tsp_result

        # --- CAJA DE RESULTADOS ---
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.metric("Costo total" if optimize_by == "cost" else "Tiempo total",
                  f"{result['total_cost']:.2f} {'€' if optimize_by == 'cost' else 'h'}")
        st.info(" → ".join(result["optimal_route"]))

        # ==========================================================
        #  BLOQUE DE IA INSERTADO
        # ==========================================================
        if "recommendations" in result and result["recommendations"]:
            st.subheader(f"🧠 IA: Basado en tu destino final ({result['optimal_route'][-1]}), ¡quizás te interese!")

            rec_list = result["recommendations"]
            num_cols = min(len(rec_list), 4)

            if num_cols > 0:
                cols = st.columns(num_cols)
                for i, rec in enumerate(rec_list):
                    if i < num_cols:
                        with cols[i]:
                            st.button(f"📍 {rec['destination_name']}",
                                      help=f"Similitud con {result['optimal_route'][-1]}: {rec['similarity']:.2f}",
                                      key=f"rec_tsp_{i}",
                                      use_container_width=True)
        # ==========================================================
        #  FIN DEL BLOQUE DE IA
        # ==========================================================

        st.markdown('</div>', unsafe_allow_html=True)
        st.divider()

        # ==========================================================
        #  RECOMENDACIONES DE LUGARES POR CIUDAD
        #  Las recomendaciones se leen desde session_state si ya fueron
        #  generadas, evitando llamadas innecesarias a la IA de Gemini.
        # ==========================================================
        show_city_recommendations(result["optimal_route"])

        st.divider()

        # ==========================================================
        #  NUEVA SECCIÓN: CREAR RESERVA
        # ==========================================================
        st.subheader("🎟️ Reservar viaje")
        num_tickets = st.number_input("Cantidad de pasajes", min_value=1, max_value=20, value=1, step=1)

        if st.button("📝 Crear reserva", type="secondary", use_container_width=True):
            # Si se optimizó por tiempo, el total_cost del resultado es en horas,
            # no en euros. Hay que recalcular el costo real usando la matriz de costos.
            real_cost = result["total_cost"]  # por defecto (cuando optimize_by == "cost")

            if optimize_by == "time":
                # Obtener la matriz de COSTOS (€) para calcular el costo real del viaje
                cost_matrix_data = get_matrix_from_api(transport_mode, "cost")
                if cost_matrix_data:
                    all_cities_cost = cost_matrix_data["cities"]
                    cost_matrix = cost_matrix_data["matrix"]
                    route = result["optimal_route"]
                    real_cost = 0.0
                    for i in range(len(route) - 1):
                        if route[i] in all_cities_cost and route[i+1] in all_cities_cost:
                            ci = all_cities_cost.index(route[i])
                            cj = all_cities_cost.index(route[i+1])
                            val = cost_matrix[ci][cj]
                            if val != -1.0:  # -1.0 representa "sin conexión"
                                real_cost += val
            itinerary = {
                "type": "multidestino",
                "transport_mode": transport_mode,
                "optimize_by": optimize_by,
                "cities": st.session_state.selected_cities,
                "optimal_route": result["optimal_route"],
                "total_cost": real_cost,           # ✅ siempre en euros
                "total_time": result["total_cost"] if optimize_by == "time" else None,  # opcional
            }

            if num_tickets == 1:
                with st.spinner("Creando reserva..."):
                    reservation = create_reservation(st.session_state.user_id, itinerary)
                if reservation:
                    st.success(f"Reserva creada correctamente ✅ ID: {reservation['reservation_id'][:12]}...")
                    st.balloons()
                    st.session_state.tsp_result = None
                    st.session_state.selected_cities = []
                    # ✅ Limpiar caché de recomendaciones al finalizar la reserva
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
                    st.session_state.tsp_result = None
                    st.session_state.selected_cities = []
                    # ✅ Limpiar caché de recomendaciones al finalizar el lote
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