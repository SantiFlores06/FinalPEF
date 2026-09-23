"""Reservations page."""

from typing import Any, Dict

import streamlit as st

from app.ui.api_client import cancel_reservation, get_user_reservations
from app.ui.formatting import format_route, format_timestamp
from app.ui.styles import status_pill_html


def render_reservations() -> None:
    """Render the current user's reservations, newest first."""
    st.header("Mis Reservas")
    user_id = st.session_state.user_id
    reservations = get_user_reservations(user_id)
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
    <b>Transporte:</b> {itinerary.get('transport_mode', 'N/A')}<br>
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
