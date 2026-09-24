"""System statistics page."""

from typing import Any, Dict

import pandas as pd
import streamlit as st

from app.ui.api_client import get_system_stats
from app.ui.formatting import format_timestamp
from app.ui.styles import render_page_header


def render_statistics() -> None:
    """Render the reservation, cache and batch processing statistics."""
    render_page_header("Estadísticas del Sistema", "Reservas, caché de rutas y procesamiento por lotes.")
    if st.button("Actualizar Estadísticas", key="refresh_statistics"):
        get_system_stats.clear()
    stats = get_system_stats()
    if not stats:
        st.warning("No hay estadísticas disponibles actualmente.")
        return
    reservation_stats = stats.get("reservations", {})
    render_reservation_overview(reservation_stats)
    render_cache_status(stats.get("cache", {}))
    render_status_chart(reservation_stats.get("by_status", {}))
    render_batch_status(stats.get("batch_processor", {}))
    st.caption(f"Última actualización: {format_timestamp(stats.get('timestamp', ''))}")


def render_reservation_overview(reservation_stats: Dict[str, Any]) -> None:
    """Render the total and cancelled reservation counts."""
    st.subheader("Resumen General")
    by_status = reservation_stats.get("by_status", {})
    total_column, cancelled_column = st.columns(2)
    total_column.metric("Total de Reservas", reservation_stats.get("total_reservations", 0))
    cancelled_column.metric("Canceladas", by_status.get("cancelled", 0))
    st.divider()


def render_cache_status(cache_stats: Dict[str, Any]) -> None:
    """Render the route cache capacity, size, hit rate and usage."""
    st.subheader("Estado del Caché (Rutas y TSP)")
    capacity_column, size_column, hit_rate_column = st.columns(3)
    capacity_column.metric("Capacidad", cache_stats.get("capacity", 0))
    size_column.metric("Items Cacheados", cache_stats.get("size", 0))
    hit_rate_column.metric("Tasa de Aciertos", f"{cache_stats.get('hit_rate', 0) * 100:.1f}%")
    usage_percent = cache_stats.get("usage_percent", 0)
    st.progress(min(usage_percent / 100, 1.0), text=f"Uso actual del caché: {usage_percent:.1f}%")
    st.divider()


def render_status_chart(by_status: Dict[str, int]) -> None:
    """Render a bar chart of the reservations per status."""
    st.subheader("Estado de Reservas")
    non_empty_statuses = {status: count for status, count in by_status.items() if count > 0}
    if not non_empty_statuses:
        st.caption("No hay datos de estado de reservas.")
    else:
        status_counts = pd.DataFrame(non_empty_statuses.items(), columns=["Estado", "Cantidad"])
        st.bar_chart(status_counts.set_index("Estado"))
    st.divider()


def render_batch_status(batch_stats: Dict[str, Any]) -> None:
    """Render the batch processor counters and state."""
    st.subheader("Procesamiento Batch (Reservas en Cola)")
    received_column, batches_column, queue_column = st.columns(3)
    received_column.metric("Items Totales Recibidos", batch_stats.get("total_items", 0))
    batches_column.metric("Batches Procesados", batch_stats.get("total_batches", 0))
    queue_column.metric("Items en Cola Ahora", batch_stats.get("queue_size", 0))
    processor_state = "Procesando" if batch_stats.get("processing") else "En espera"
    st.markdown(f"**Estado actual del procesador:** {processor_state}")
