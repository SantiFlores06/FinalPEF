"""Reservations page."""

from typing import Any, Dict, List

import streamlit as st

from app.ui.api_client import cancel_reservation, get_user_reservations
from app.ui.formatting import format_route, format_timestamp, format_transport
from app.ui.state import is_batch_settling
from app.ui.styles import render_page_header, status_pill_html

ACTIVE_STATUSES = {"pending", "processing"}
STATUS_REFRESH_SECONDS = 1.0


def render_reservations() -> None:
    """Render the current user's reservations, refreshing them while any is still in progress."""
    render_page_header("Mis Reservas", "Seguí el estado de tus reservas individuales y en lote.")
    refresh_interval = STATUS_REFRESH_SECONDS if st.session_state.reservations_auto_refresh else None
    st.fragment(render_reservation_list, run_every=refresh_interval)()


def needs_status_refresh(reservations: List[Dict[str, Any]]) -> bool:
    """Return whether some reservation may still change status on its own."""
    has_active = any(reservation.get("status") in ACTIVE_STATUSES for reservation in reservations)
    return has_active or is_batch_settling()


def sync_auto_refresh(reservations: List[Dict[str, Any]]) -> None:
    """Turn the auto-refresh on or off, rerunning the page when that changes."""
    should_refresh = needs_status_refresh(reservations)
    if should_refresh != st.session_state.reservations_auto_refresh:
        st.session_state.reservations_auto_refresh = should_refresh
        st.rerun()


def render_reservation_list() -> None:
    """Render the reservations newest first, with a note while they are being processed."""
    user_id = st.session_state.user_id
    reservations = get_user_reservations(user_id)
    sync_auto_refresh(reservations)
    if st.session_state.reservations_auto_refresh:
        st.caption("Procesando reservas... el estado se actualiza automáticamente.")
    if not reservations:
        st.info("No tienes reservas registradas.")
        return
    st.caption(f"Mostrando {len(reservations)} reservas para el usuario {user_id[:12]}...")
    for reservation in sorted(reservations, key=lambda item: item.get("created_at", ""), reverse=True):
        render_reservation(reservation)


def reservation_card_html(reservation: Dict[str, Any]) -> str:
    """Return the HTML card describing a reservation."""
    itinerary = reservation.get("itinerary", {})
    return f"""
<div class="res-card">
    <div class="res-header">Reserva #{reservation.get('reservation_id', '')[:8]}...</div>
    <div class="res-sub">Creada: {format_timestamp(reservation.get('created_at', ''))}</div>
    <b>Tipo:</b> {itinerary.get('type', 'N/A').capitalize()}<br>
    <b>Transporte:</b> {format_transport(itinerary.get('transport_mode', 'N/A'))}<br>
    <b>Ruta óptima:</b> {format_route(itinerary.get('optimal_route', []))}<br>
    <b>Total:</b> {itinerary.get('total_cost', 0):.2f} €<br>
    {status_pill_html(reservation.get('status', 'pending'))}
</div>
"""


def render_reservation(reservation: Dict[str, Any]) -> None:
    """Render a reservation card with its cancel button."""
    card_column, action_column = st.columns([0.80, 0.20])
    card_column.markdown(reservation_card_html(reservation), unsafe_allow_html=True)
    if reservation.get("status", "pending") == "cancelled":
        return
    reservation_id = reservation.get("reservation_id")
    if action_column.button("Cancelar", key=f"cancel_{reservation_id}", use_container_width=True):
        with st.spinner("Cancelando reserva..."):
            was_cancelled = cancel_reservation(reservation_id)
        if was_cancelled:
            st.toast("Reserva cancelada exitosamente")
            st.rerun()
        else:
            st.error("Error al cancelar la reserva")
