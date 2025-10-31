"""
streamlit_app.py - Interfaz gráfica para el sistema de planificación de viajes.
UI interactiva construida con Streamlit.
"""

import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional

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
</style>
""",
    unsafe_allow_html=True
)

# ==========================================================
# FUNCIONES AUXILIARES DE API
# ==========================================================

def check_api_health() -> bool:
    """Verifica si la API está disponible."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


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


def get_system_stats() -> Dict:
    """Obtiene estadísticas del sistema."""
    try:
        response = requests.get(f"{API_URL}/stats", timeout=5)
        return response.json() if response.status_code == 200 else {}
    except Exception:
        return {}

# ==========================================================
# SESIÓN DE USUARIO
# ==========================================================
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{int(datetime.now().timestamp())}"

if "tsp_result" not in st.session_state:
    st.session_state.tsp_result = None

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
    st.error("🔴 API no disponible. Verifica que esté ejecutándose en http://localhost:8000")
    st.stop()
else:
    st.success("🟢 API conectada y funcionando correctamente")

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
# 🌍 PÁGINA: RUTA MULTIDESTINO
# ==========================================================
if page == "🌍 Ruta Multidestino":
    st.header("🌍 Optimización de Ruta Multidestino (TSP)")
    st.write("Encuentra el orden óptimo para visitar múltiples ciudades minimizando costo o tiempo.")

    transport_mode = st.selectbox("🚗 Tipo de transporte", ["auto", "avión", "tren"])
    optimize_by = st.selectbox("⚖️ Optimizar por", ["cost", "time"], format_func=lambda x: "Costo (€)" if x == "cost" else "Tiempo (h)")

    matrix_data = get_matrix_from_api(transport_mode, optimize_by)
    if not matrix_data:
        st.stop()

    all_cities = matrix_data["cities"]
    cost_matrix = matrix_data["matrix"]

    selected_cities = st.multiselect("Selecciona ciudades:", all_cities, default=all_cities[:3])
    return_to_start = st.checkbox("Regresar al origen", value=True)

    if len(selected_cities) < 2:
        st.warning("⚠️ Selecciona al menos dos ciudades.")
    else:
        if st.button("🧠 Calcular Ruta Óptima", type="primary"):
            indices = [all_cities.index(c) for c in selected_cities]
            submatrix = [[cost_matrix[i][j] for j in indices] for i in indices]
            st.write("📊 Matriz de valores desde backend:")
            df = pd.DataFrame(submatrix, index=selected_cities, columns=selected_cities)
            st.dataframe(df.style.format("{:.2f}"))
            with st.spinner("Calculando mejor ruta..."):
                result = optimize_multi_destination(selected_cities, submatrix, return_to_start)
            if result:
                st.session_state.tsp_result = result
                st.success("✅ Ruta óptima encontrada")

    # Mostrar resultado guardado (permite crear reserva)
    if st.session_state.tsp_result:
        result = st.session_state.tsp_result
        st.metric("Costo total" if optimize_by == "cost" else "Tiempo total",
                  f"{result['total_cost']:.2f} {'€' if optimize_by == 'cost' else 'h'}")
        st.info(" → ".join(result["optimal_route"]))
        st.divider()

        # ==========================================================
        # 🎟️ NUEVA SECCIÓN: CREAR RESERVA (individual o lote)
        # ==========================================================
        st.subheader("🎟️ Reservar viaje")
        num_tickets = st.number_input("Cantidad de pasajes", min_value=1, max_value=20, value=1, step=1)

        if st.button("📝 Crear reserva", type="secondary"):
            itinerary = {
                "type": "multidestino",
                "transport_mode": transport_mode,
                "optimize_by": optimize_by,
                "cities": selected_cities,
                "optimal_route": result["optimal_route"],
                "total_cost": result["total_cost"]
            }

            if num_tickets == 1:
                reservation = create_reservation(st.session_state.user_id, itinerary)
                if reservation:
                    st.success(f"Reserva creada correctamente ✅ ID: {reservation['reservation_id'][:12]}...")
                    st.balloons()
                    st.session_state.tsp_result = None
                    st.session_state.page = "📋 Mis Reservas"
                    st.rerun()
            else:
                batch_payload = [
                    {"user_id": st.session_state.user_id, "itinerary": itinerary}
                    for _ in range(num_tickets)
                ]
                response = create_reservations_batch(batch_payload)
                if response:
                    st.success(f"🧩 Lote creado correctamente ({response['count']} reservas en cola)")
                    st.balloons()
                    st.session_state.tsp_result = None
                    st.session_state.page = "📋 Mis Reservas"
                    st.rerun()

# ==========================================================
# 🏠 INICIO
# ==========================================================
elif page == "🏠 Inicio":
    st.header("🏠 Bienvenido al Sistema de Planificación de Viajes")
    st.info("Usa el menú lateral para navegar entre las funciones disponibles.")

# ==========================================================
# 📋 MIS RESERVAS
# ==========================================================
elif page == "📋 Mis Reservas":
    st.header("📋 Mis Reservas")

    st.markdown("""
    <style>
    .res-card {
        background-color: #f9fafc;
        border: 1px solid #d1d9e6;
        border-radius: 16px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0px 3px 10px rgba(0,0,0,0.05);
        transition: transform 0.2s ease-in-out;
    }
    .res-card:hover { transform: scale(1.01); box-shadow: 0px 6px 15px rgba(0,0,0,0.1); }
    .res-header { font-size: 1.2rem; font-weight: bold; color: #1565C0; }
    .res-sub { font-size: 0.9rem; color: #555; }
    .res-badge { display: inline-block; padding: 0.25rem 0.6rem; border-radius: 8px; font-size: 0.8rem; font-weight: 600; color: white; }
    .status-pending { background-color: #fbc02d; }
    .status-confirmed { background-color: #43a047; }
    .status-error { background-color: #e53935; }
    </style>
    """, unsafe_allow_html=True)

    reservations = get_user_reservations(st.session_state.user_id)
    if not reservations:
        st.info("No tienes reservas registradas.")
    else:
        for r in reservations:
            itinerary = r.get("itinerary", {})
            cities = itinerary.get("cities", [])
            optimal_route = itinerary.get("optimal_route", [])
            status = r.get("status", "pending")
            status_class = (
                "status-confirmed" if status == "confirmed"
                else "status-error" if status == "error"
                else "status-pending"
            )

            st.markdown(f"""
            <div class="res-card">
                <div class="res-header">🧾 Reserva #{r.get('reservation_id')[:8]}</div>
                <div class="res-sub">Creada: {r.get('created_at', '').replace('T', ' ')}</div>
                <div class="res-sub">Usuario: <b>{r.get('user_id')}</b></div>
                <br>
                <b>🧭 Tipo:</b> {itinerary.get('type', '').capitalize()} <br>
                <b>🚗 Transporte:</b> {itinerary.get('transport_mode', '')} <br>
                <b>⚖️ Criterio:</b> {itinerary.get('optimize_by', '')} <br>
                <b>🌆 Ciudades:</b> {" → ".join(cities)} <br>
                <b>🗺️ Ruta óptima:</b> {" → ".join(optimal_route)} <br>
                <b>💰 Total:</b> {itinerary.get('total_cost', 0)} € <br><br>
                <span class="res-badge {status_class}">Estado: {status.upper()}</span>
            </div>
            """, unsafe_allow_html=True)

# ==========================================================
# 📊 ESTADÍSTICAS
# ==========================================================
elif page == "📊 Estadísticas":
    st.header("📊 Estadísticas del Sistema")

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
    col2.metric("⏳ Pendientes", by_status.get("pending", 0))
    col3.metric("⚙️ Máx. Concurrentes", reservations.get("max_concurrent", 0))
    st.divider()

    st.subheader("🧩 Estado del Caché")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Capacidad", cache.get("capacity", 0))
    col2.metric("Ocupado", cache.get("size", 0))
    col3.metric("Tasa de Aciertos (%)", cache.get("hit_rate", 0))
    col4.metric("Uso (%)", cache.get("usage_percent", 0))
    usage = cache.get("usage_percent", 0)
    st.progress(min(usage / 100, 1.0), text=f"Uso actual del caché: {usage:.1f}%")
    st.divider()

    st.subheader("🧾 Estado de Reservas")
    st.markdown(f"""
    - **Total:** {reservations.get("total_reservations", 0)}
    - **Pendientes:** {by_status.get("pending", 0)}
    - **Confirmadas:** {by_status.get("confirmed", 0)}
    - **Fallidas:** {by_status.get("failed", 0)}
    - **Canceladas:** {by_status.get("cancelled", 0)}
    """)
    st.divider()

    st.subheader("⚙️ Procesamiento Batch (Reservas en Cola)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Items Totales", batch.get("total_items", 0))
    col2.metric("Batches Procesados", batch.get("total_batches", 0))
    col3.metric("Tamaño de Lote", batch.get("batch_size", 0))
    col1, col2, col3 = st.columns(3)
    col1.metric("Procesados OK", batch.get("items_processed", 0))
    col2.metric("Fallidos", batch.get("items_failed", 0))
    col3.metric("En Cola", batch.get("queue_size", 0))
    processing = "✅ Procesando" if batch.get("processing") else "🟡 En espera"
    st.info(f"**Estado actual del procesador:** {processing}")
    st.caption(f"Última actualización: {stats.get('timestamp', '')}")
