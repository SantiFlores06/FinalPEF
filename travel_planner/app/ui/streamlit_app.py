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
    .status-pending { background-color: #fbc02d; } /* Amarillo */
    .status-processing { background-color: #1E88E5; } /* Azul */
    .status-confirmed { background-color: #43a047; } /* Verde */
    .status-failed { background-color: #e53935; } /* Rojo */
    .status-cancelled { background-color: #757575; } /* Gris */
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
        # Nota: La respuesta de la API ahora incluye la IA
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

@st.cache_data(ttl=10)
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
if "user_id" not in st.session_state: st.session_state.user_id = "user_demo_123"

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
    # [cite_start]Corrección de error de imagen [cite: 3773-3776]
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

    matrix_data = get_matrix_from_api(transport_mode, optimize_by)
    if not matrix_data:
        st.error("No se pudieron cargar los datos de las rutas. Revisa el backend.")
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
            submatrix = [
                [
                    float('inf') if cost_matrix[i][j] is None else cost_matrix[i][j]
                    for j in indices
                ]
                for i in indices
            ]
            
            st.write("📊 Matriz de valores desde backend:")
            df = pd.DataFrame(submatrix, index=selected_cities, columns=selected_cities)
            st.dataframe(df.style.format("{:.2f}"))
            
            with st.spinner("Calculando mejor ruta... (IA incluida)"):
                result = optimize_multi_destination(selected_cities, submatrix, return_to_start)
            if result:
                st.session_state.tsp_result = result
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
            # Asegurarse de no crear más columnas que recomendaciones
            num_cols = min(len(rec_list), 4) 
            
            if num_cols > 0:
                cols = st.columns(num_cols)
                for i, rec in enumerate(rec_list):
                    if i < num_cols: # Solo mostrar tantas como columnas tengamos
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
        #  NUEVA SECCIÓN: CREAR RESERVA 
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
                with st.spinner("Creando reserva..."):
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
                with st.spinner(f"Enviando lote de {num_tickets} reservas..."):
                    response = create_reservations_batch(batch_payload)
                if response:
                    st.success(f"🧩 Lote creado correctamente ({response['count']} reservas en cola)")
                    st.balloons()
                    st.session_state.tsp_result = None
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
        if st.button("Actualizar Estados"):
            st.rerun()
            
        for r in sorted(reservations, key=lambda x: x.get('created_at', ''), reverse=True):
            itinerary = r.get("itinerary", {})
            optimal_route = itinerary.get("optimal_route", [])
            status = r.get("status", "pending")
            status_class = (
                "status-confirmed" if status == "confirmed"
                else "status-failed" if status == "failed"
                else "status-cancelled" if status == "cancelled"
                else "status-processing" if status == "processing"
                else "status-pending"
            )

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
    col2.metric("⏳ Pendientes", by_status.get("pending", 0) + by_status.get("processing", 0))
    col3.metric("⚙️ Máx. Concurrentes", reservations.get("max_concurrent", 0))
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
        # Filtrar estados con 0 para que el gráfico sea más limpio
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