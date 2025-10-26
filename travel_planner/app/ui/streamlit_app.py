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

# URL de la API (cambiar si se ejecuta en otro puerto)
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
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# Funciones auxiliares para interactuar con la API

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

def get_system_stats() -> Dict:
    """Obtiene estadísticas del sistema."""
    try:
        response = requests.get(f"{API_URL}/stats", timeout=5)
        return response.json() if response.status_code == 200 else {}
    except:
        return {}

# Inicializar session_state
if 'user_id' not in st.session_state:
    st.session_state.user_id = f"user_{datetime.now().timestamp()}"
if 'reservations' not in st.session_state:
    st.session_state.reservations = []

# Header principal
st.markdown('<h1 class="main-header">✈️ Sistema de Planificación de Viajes</h1>', unsafe_allow_html=True)

# Verificar estado de la API
api_status = check_api_health()
if api_status:
    st.success("🟢 API conectada y funcionando")
else:
    st.error("🔴 API no disponible. Asegúrate de ejecutar: `uvicorn app.api.server:app --reload`")
    st.stop()

# Sidebar con navegación
with st.sidebar:
    st.image("https://via.placeholder.com/300x100?text=Travel+Planner", use_column_width=True)
    st.header("Navegación")
    page = st.radio(
        "Selecciona una opción:",
        ["🏠 Inicio", "🗺️ Planificar Ruta", "📋 Mis Reservas", "📊 Estadísticas"]
    )
    
    st.divider()
    st.subheader("Usuario Actual")
    st.info(f"ID: {st.session_state.user_id[:12]}...")

# Página: Inicio
if page == "🏠 Inicio":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Rutas Disponibles", "10+")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Ciudades", "8")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Tipos de Transporte", "3")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    st.header("Bienvenido al Sistema de Planificación de Viajes")
    st.write("""
    Este sistema te permite:
    - 🗺️ **Calcular rutas óptimas** entre ciudades usando algoritmo de Dijkstra
    - 🔄 **Optimizar viajes multidestino** con programación dinámica (TSP)
    - 📦 **Crear reservas** procesadas de forma asíncrona
    - 💾 **Caché inteligente** para respuestas rápidas
    - 📊 **Estadísticas en tiempo real** del sistema
    """)
    
    st.info("👈 Usa el menú lateral para navegar entre las diferentes opciones")
    
    # Mostrar ciudades disponibles
    st.subheader("🌍 Ciudades Disponibles")
    cities = ["Madrid", "Barcelona", "Valencia", "Sevilla", "París", "Roma", "Lisboa"]
    cols = st.columns(4)
    for i, city in enumerate(cities):
        cols[i % 4].button(f"📍 {city}", disabled=True)

# Página: Planificar Ruta
elif page == "🗺️ Planificar Ruta":
    st.header("Planificación de Ruta")
    
    tab1, tab2 = st.tabs(["Ruta Simple", "Ruta Multidestino"])
    
    with tab1:
        st.subheader("Calcular Ruta entre Dos Ciudades")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            origin = st.selectbox(
                "Ciudad de Origen",
                ["Madrid", "Barcelona", "Valencia", "Sevilla", "París", "Roma", "Lisboa"]
            )
        
        with col2:
            destination = st.selectbox(
                "Ciudad de Destino",
                ["Barcelona", "París", "Valencia", "Roma", "Madrid", "Sevilla", "Lisboa"]
            )
        
        with col3:
            optimize_by = st.selectbox(
                "Optimizar por",
                ["cost", "time"]
            )
        
        if st.button("🔍 Calcular Ruta", type="primary"):
            if origin == destination:
                st.warning("El origen y destino deben ser diferentes")
            else:
                with st.spinner("Calculando mejor ruta..."):
                    result = calculate_route(origin, destination, optimize_by)
                    
                    if result:
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.success("✅ Ruta calculada exitosamente")
                        
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            st.metric(
                                "Costo Total",
                                f"€{result['total_cost']:.2f}"
                            )
                            st.write(f"**Ruta:** {' → '.join(result['path'])}")
                        
                        with col_b:
                            if result.get('cached'):
                                st.info("⚡ Resultado obtenido desde caché")
                            else:
                                st.info("🔄 Resultado calculado")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Opción para crear reserva
                        if st.button("📝 Crear Reserva con esta Ruta"):
                            itinerary = {
                                "origin": origin,
                                "destination": destination,
                                "path": result['path'],
                                "total_cost": result['total_cost']
                            }
                            reservation = create_reservation(st.session_state.user_id, itinerary)
                            
                            if reservation:
                                st.session_state.reservations.append(reservation)
                                st.success(f"✅ Reserva creada: {reservation['reservation_id']}")
    
    with tab2:
        st.subheader("Optimización de Ruta Multidestino (TSP)")
        st.write("**Próximamente:** Optimización de rutas visitando múltiples ciudades")
        st.info("🔧 Funcionalidad en desarrollo")

# Página: Mis Reservas
elif page == "📋 Mis Reservas":
    st.header("Mis Reservas")
    
    reservations = get_user_reservations(st.session_state.user_id)
    
    if not reservations:
        st.info("No tienes reservas activas. Crea una en la sección 'Planificar Ruta'")
    else:
        st.write(f"**Total de reservas:** {len(reservations)}")
        
        for res in reservations:
            with st.expander(f"📋 Reserva: {res['reservation_id'][:12]}..."):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Estado:** {res['status']}")
                    st.write(f"**Costo:** €{res['total_cost']:.2f}")
                
                with col2:
                    st.write(f"**Creada:** {res['created_at'][:10]}")
                
                with col3:
                    if res['status'] == 'confirmed':
                        if st.button(f"❌ Cancelar", key=f"cancel_{res['reservation_id']}"):
                            try:
                                response = requests.delete(f"{API_URL}/reservations/{res['reservation_id']}")
                                if response.status_code == 200:
                                    st.success("Reserva cancelada")
                                    st.rerun()
                            except:
                                st.error("Error cancelando reserva")

# Página: Estadísticas
elif page == "📊 Estadísticas":
    st.header("Estadísticas del Sistema")
    
    stats = get_system_stats()
    
    if stats:
        tab1, tab2, tab3 = st.tabs(["Cache", "Reservas", "Sistema"])
        
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
                
                if 'by_status' in res_stats:
                    df = pd.DataFrame(list(res_stats['by_status'].items()), columns=['Estado', 'Cantidad'])
                    st.bar_chart(df.set_index('Estado'))
        
        with tab3:
            st.subheader("Información del Sistema")
            st.write(f"**Timestamp:** {stats.get('timestamp', 'N/A')}")
            st.json(stats)
    
    if st.button("🔄 Actualizar Estadísticas"):
        st.rerun()

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Sistema de Planificación de Viajes Multidestino | Programación Eficiente 2025</p>
</div>
""", unsafe_allow_html=True)
