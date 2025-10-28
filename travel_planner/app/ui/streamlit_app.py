"""
streamlit_app.py - Interfaz gráfica para el sistema de planificación de viajes.
UI interactiva construida con Streamlit.
"""

import streamlit as st
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json

# Configuración de la página
st.set_page_config(
    page_title="Travel Planner - Sistema de Planificación",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL de la API
API_URL = "http://localhost:8000"

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .route-step {
        padding: 0.5rem;
        margin: 0.5rem 0;
        background-color: #f8f9fa;
        border-left: 3px solid #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Funciones auxiliares para API

def check_api_health() -> bool:
    """Verifica si la API está disponible."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def calculate_route(origin: str, destination: str, optimize_by: str = "cost") -> Dict:
    """Calcula ruta más corta entre dos ciudades."""
    try:
        response = requests.post(
            f"{API_URL}/routes/shortest",
            json={
                "origin": origin,
                "destination": destination,
                "optimize_by": optimize_by
            },
            timeout=10
        )
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Error conectando con API: {e}")
        return None

def optimize_multi_destination(cities: List[str], cost_matrix: List[List[float]], return_to_start: bool = True) -> Dict:
    """Optimiza ruta visitando múltiples destinos (TSP)."""
    try:
        response = requests.post(
            f"{API_URL}/routes/optimize-multi",
            json={
                "cities": cities,
                "cost_matrix": cost_matrix,
                "return_to_start": return_to_start
            },
            timeout=30
        )
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Error optimizando ruta: {e}")
        return None

def plan_itinerary(user_id: str, origin: str, destinations: List[str], max_budget: float) -> Dict:
    """Planifica itinerario completo."""
    try:
        response = requests.post(
            f"{API_URL}/itinerary/plan",
            json={
                "user_id": user_id,
                "origin": origin,
                "destinations": destinations,
                "max_budget": max_budget,
                "max_duration_hours": 72.0,
                "transport_preferences": ["tren", "avión", "bus"]
            },
            timeout=15
        )
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Error planificando itinerario: {e}")
        return None

def create_reservation(user_id: str, itinerary: Dict) -> Dict:
    """Crea una nueva reserva."""
    try:
        response = requests.post(
            f"{API_URL}/reservations",
            json={
                "user_id": user_id,
                "itinerary": itinerary
            },
            timeout=10
        )
        return response.json() if response.status_code == 200 else None
    except Exception as e:
        st.error(f"Error creando reserva: {e}")
        return None

def get_user_reservations(user_id: str) -> List[Dict]:
    """Obtiene reservas de un usuario."""
    try:
        response = requests.get(f"{API_URL}/reservations/user/{user_id}", timeout=10)
        return response.json() if response.status_code == 200 else []
    except:
        return []

def get_reservation_status(reservation_id: str) -> Dict:
    """Obtiene el estado de una reserva."""
    try:
        response = requests.get(f"{API_URL}/reservations/{reservation_id}", timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def cancel_reservation(reservation_id: str) -> bool:
    """Cancela una reserva."""
    try:
        response = requests.delete(f"{API_URL}/reservations/{reservation_id}", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_system_stats() -> Dict:
    """Obtiene estadísticas del sistema."""
    try:
        response = requests.get(f"{API_URL}/stats", timeout=5)
        return response.json() if response.status_code == 200 else {}
    except:
        return {}

# Datos de ciudades y costos (simulados)
CITIES = ["Madrid", "Barcelona", "Valencia", "Sevilla", "París", "Roma", "Lisboa", "Berlín"]

# Matriz de costos simplificada (para TSP)
def get_cost_matrix(cities: List[str]) -> List[List[float]]:
    """Genera matriz de costos para las ciudades seleccionadas."""
    # Matriz de costos predefinida (simplified)
    base_costs = {
        ("Madrid", "Barcelona"): 50,
        ("Madrid", "Valencia"): 40,
        ("Madrid", "Sevilla"): 60,
        ("Madrid", "Lisboa"): 55,
        ("Barcelona", "París"): 100,
        ("Barcelona", "Valencia"): 45,
        ("Barcelona", "Roma"): 130,
        ("París", "Roma"): 150,
        ("París", "Berlín"): 120,
        ("Sevilla", "Lisboa"): 50,
        ("Valencia", "París"): 120,
        ("Roma", "Berlín"): 140,
    }
    
    n = len(cities)
    matrix = [[0.0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i != j:
                key = (cities[i], cities[j])
                reverse_key = (cities[j], cities[i])
                if key in base_costs:
                    matrix[i][j] = base_costs[key]
                elif reverse_key in base_costs:
                    matrix[i][j] = base_costs[reverse_key]
                else:
                    # Costo por defecto si no hay ruta directa
                    matrix[i][j] = 200
    
    return matrix

# Inicializar session_state
if 'user_id' not in st.session_state:
    st.session_state.user_id = f"user_{int(datetime.now().timestamp())}"
if 'reservations' not in st.session_state:
    st.session_state.reservations = []
if 'last_route' not in st.session_state:
    st.session_state.last_route = None

# Header principal
st.markdown('<h1 class="main-header">✈️ Sistema de Planificación de Viajes</h1>', unsafe_allow_html=True)

# Verificar estado de la API
api_status = check_api_health()
if api_status:
    st.success("🟢 API conectada y funcionando")
else:
    st.error("🔴 API no disponible. Verifica que esté ejecutándose en http://localhost:8000")
    st.stop()

# Sidebar con navegación
with st.sidebar:
    st.header("🚀 Travel Planner")
    st.header("Navegación")
    page = st.radio(
        "Selecciona una opción:",
        ["🏠 Inicio", "🗺️ Ruta Simple", "🌍 Ruta Multidestino", "📋 Mis Reservas", "📊 Estadísticas"]
    )
    
    st.divider()
    st.subheader("Usuario Actual")
    st.info(f"ID: {st.session_state.user_id[:12]}...")

# Página: Inicio
if page == "🏠 Inicio":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Ciudades Disponibles", len(CITIES))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        reservations = get_user_reservations(st.session_state.user_id)
        st.metric("Mis Reservas", len(reservations))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Algoritmos", "3")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    st.header("Bienvenido al Sistema de Planificación de Viajes")
    st.write("""
    Este sistema te permite:
    - 🗺️ **Calcular rutas óptimas** entre dos ciudades usando Dijkstra
    - 🌍 **Optimizar viajes multidestino** con programación dinámica (TSP)
    - 📦 **Crear reservas** procesadas de forma asíncrona
    - 💾 **Caché inteligente** para respuestas rápidas
    - 📊 **Estadísticas en tiempo real** del sistema
    """)
    
    st.info("👈 Usa el menú lateral para navegar")
    
    st.subheader("🌍 Ciudades Disponibles")
    cols = st.columns(4)
    for i, city in enumerate(CITIES):
        cols[i % 4].button(f"📍 {city}", disabled=True, key=f"city_{i}")

# Página: Ruta Simple
elif page == "🗺️ Ruta Simple":
    st.header("Calcular Ruta Simple (A → B)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        origin = st.selectbox("Ciudad de Origen", CITIES, key="origin_simple")
    
    with col2:
        destination = st.selectbox(
            "Ciudad de Destino",
            [c for c in CITIES if c != origin],
            key="dest_simple"
        )
    
    with col3:
        optimize_by = st.selectbox("Optimizar por", ["cost", "time"])
    
    if st.button("🔍 Calcular Ruta", type="primary"):
        with st.spinner("Calculando mejor ruta..."):
            result = calculate_route(origin, destination, optimize_by)
            
            if result:
                st.session_state.last_route = result
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.success("✅ Ruta calculada exitosamente")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.metric("Costo Total", f"€{result['total_cost']:.2f}")
                    st.write(f"**Ruta:** {' → '.join(result['path'])}")
                
                with col_b:
                    if result.get('cached'):
                        st.info("⚡ Resultado desde caché")
                    else:
                        st.info("🔄 Resultado calculado")
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    # Botón de reserva si hay ruta calculada
    if st.session_state.last_route:
        st.divider()
        if st.button("📝 Crear Reserva con esta Ruta"):
            itinerary = {
                "origin": st.session_state.last_route['origin'],
                "destination": st.session_state.last_route['destination'],
                "path": st.session_state.last_route['path'],
                "total_cost": st.session_state.last_route['total_cost']
            }
            reservation = create_reservation(st.session_state.user_id, itinerary)
            
            if reservation:
                st.success(f"✅ Reserva creada: {reservation['reservation_id'][:12]}...")
                st.balloons()

# Página: Ruta Multidestino (TSP)
elif page == "🌍 Ruta Multidestino":
    st.header("Optimización de Ruta Multidestino (TSP)")
    st.write("Encuentra el orden óptimo para visitar múltiples ciudades minimizando el costo total")
    
    st.subheader("Selecciona las ciudades a visitar")
    
    selected_cities = st.multiselect(
        "Ciudades (selecciona al menos 3)",
        CITIES,
        default=["Madrid", "Barcelona", "París"],
        help="Selecciona las ciudades que deseas visitar"
    )
    
    return_to_start = st.checkbox("Regresar a la ciudad de origen", value=True)
    
    if len(selected_cities) < 2:
        st.warning("⚠️ Selecciona al menos 2 ciudades")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"📍 Ciudades seleccionadas: {len(selected_cities)}")
        
        with col2:
            st.info(f"🔄 Complejidad: O(n² × 2^n) = O({len(selected_cities)}² × 2^{len(selected_cities)})")
        
        if st.button("🧠 Calcular Ruta Óptima", type="primary"):
            if len(selected_cities) > 15:
                st.error("❌ Máximo 15 ciudades para mantener el rendimiento")
            else:
                with st.spinner(f"Calculando ruta óptima para {len(selected_cities)} ciudades..."):
                    cost_matrix = get_cost_matrix(selected_cities)
                    result = optimize_multi_destination(selected_cities, cost_matrix, return_to_start)
                    
                    if result:
                        st.success("✅ Ruta óptima encontrada!")
                        
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.metric("Costo Total", f"€{result['total_cost']:.2f}")
                        
                        with col_b:
                            st.metric("Tiempo de Cálculo", f"{result['computation_time']:.3f}s")
                        
                        with col_c:
                            st.metric("Ciudades", len(selected_cities))
                        
                        st.subheader("🗺️ Ruta Óptima")
                        route_display = " → ".join(result['optimal_route'])
                        st.info(route_display)
                        
                        # Detalles paso a paso
                        st.subheader("📋 Detalles del Recorrido")
                        for i in range(len(result['optimal_route']) - 1):
                            from_city = result['optimal_route'][i]
                            to_city = result['optimal_route'][i + 1]
                            cost = cost_matrix[selected_cities.index(from_city)][selected_cities.index(to_city)]
                            
                            st.markdown(
                                f'<div class="route-step">Paso {i+1}: {from_city} → {to_city} | Costo: €{cost:.2f}</div>',
                                unsafe_allow_html=True
                            )
                        
                        # Opción de crear reserva
                        st.divider()
                        if st.button("📝 Crear Reserva con Ruta Multidestino"):
                            itinerary = {
                                "type": "multidestino",
                                "cities": selected_cities,
                                "optimal_route": result['optimal_route'],
                                "total_cost": result['total_cost'],
                                "computation_time": result['computation_time']
                            }
                            reservation = create_reservation(st.session_state.user_id, itinerary)
                            
                            if reservation:
                                st.success(f"✅ Reserva multidestino creada: {reservation['reservation_id'][:12]}...")
                                st.balloons()

# Página: Mis Reservas
elif page == "📋 Mis Reservas":
    st.header("Mis Reservas")
    
    reservations = get_user_reservations(st.session_state.user_id)
    
    if not reservations:
        st.info("📭 No tienes reservas activas. Crea una en 'Ruta Simple' o 'Ruta Multidestino'")
    else:
        st.write(f"**Total de reservas:** {len(reservations)}")
        
        # Botón para refrescar
        if st.button("🔄 Actualizar Estado de Reservas"):
            st.rerun()
        
        st.divider()
        
        for idx, res in enumerate(reservations):
            with st.expander(f"📋 Reserva {idx+1}: {res['reservation_id'][:12]}... - {res['status'].upper()}", expanded=(idx==0)):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    status_emoji = {
                        'pending': '⏳',
                        'processing': '⚙️',
                        'confirmed': '✅',
                        'failed': '❌',
                        'cancelled': '🚫'
                    }
                    st.write(f"**Estado:** {status_emoji.get(res['status'], '❓')} {res['status'].upper()}")
                    st.write(f"**Costo:** €{res['total_cost']:.2f}")
                
                with col2:
                    created = datetime.fromisoformat(res['created_at'])
                    st.write(f"**Creada:** {created.strftime('%d/%m/%Y %H:%M')}")
                    st.write(f"**ID:** {res['reservation_id'][:16]}...")
                
                with col3:
                    # Mostrar itinerario
                    if 'itinerary' in res:
                        itinerary = res['itinerary']
                        if itinerary.get('type') == 'multidestino':
                            st.write(f"**Tipo:** 🌍 Multidestino")
                            st.write(f"**Ciudades:** {len(itinerary.get('cities', []))}")
                        else:
                            st.write(f"**Tipo:** 🗺️ Simple")
                            st.write(f"**Ruta:** {itinerary.get('origin', '?')} → {itinerary.get('destination', '?')}")
                
                # Botón de cancelación
                if res['status'] in ['pending', 'processing', 'confirmed']:
                    if st.button(f"❌ Cancelar Reserva", key=f"cancel_{res['reservation_id']}"):
                        if cancel_reservation(res['reservation_id']):
                            st.success("Reserva cancelada exitosamente")
                            st.rerun()
                        else:
                            st.error("No se pudo cancelar la reserva")

# Página: Estadísticas
elif page == "📊 Estadísticas":
    st.header("Estadísticas del Sistema")
    
    if st.button("🔄 Actualizar Estadísticas"):
        st.rerun()
    
    stats = get_system_stats()
    
    if stats:
        tab1, tab2, tab3 = st.tabs(["📦 Cache", "📋 Reservas", "⚙️ Sistema"])
        
        with tab1:
            st.subheader("Estadísticas de Caché")
            if 'cache' in stats:
                cache_stats = stats['cache']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Capacidad", cache_stats.get('capacity', 0))
                with col2:
                    st.metric("Items", cache_stats.get('size', 0))
                with col3:
                    st.metric("Hits", cache_stats.get('hits', 0))
                with col4:
                    st.metric("Misses", cache_stats.get('misses', 0))
                
                hit_rate = cache_stats.get('hit_rate', 0) * 100
                st.progress(hit_rate / 100)
                st.write(f"**Hit Rate:** {hit_rate:.1f}%")
        
        with tab2:
            st.subheader("Estadísticas de Reservas")
            if 'reservations' in stats:
                res_stats = stats['reservations']
                
                st.metric("Total de Reservas", res_stats.get('total_reservations', 0))
                
                if 'by_status' in res_stats and res_stats['by_status']:
                    df = pd.DataFrame(list(res_stats['by_status'].items()), columns=['Estado', 'Cantidad'])
                    st.bar_chart(df.set_index('Estado'))
                else:
                    st.info("No hay reservas registradas todavía")
        
        with tab3:
            st.subheader("Información del Sistema")
            st.write(f"**Timestamp:** {stats.get('timestamp', 'N/A')}")
            st.json(stats)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Sistema de Planificación de Viajes Multidestino | Programación Eficiente 2025</p>
</div>
""", unsafe_allow_html=True)
