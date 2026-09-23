"""Custom CSS and small HTML snippets used across the views."""

import streamlit as st

CSS = """
<style>
.algo-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-bottom: 6px;
}
.badge-dijkstra  { background: #DBEAFE; color: #1D4ED8; }
.badge-held-karp { background: #D1FAE5; color: #065F46; }
.badge-genetic   { background: #FCE7F3; color: #9D174D; }

.res-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
}
.res-header { font-size: 1.05rem; font-weight: 600; color: #1F2937; }
.res-sub    { font-size: 0.85rem; color: #6B7280; margin-bottom: 0.6rem; }

.status-pill {
    display: inline-block;
    margin-top: 0.6rem;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 600;
}
.status-pending    { background: #FEF3C7; color: #92400E; }
.status-processing { background: #DBEAFE; color: #1D4ED8; }
.status-confirmed  { background: #D1FAE5; color: #065F46; }
.status-failed     { background: #FEE2E2; color: #991B1B; }
.status-cancelled  { background: #F3F4F6; color: #4B5563; }
</style>
"""

ALGORITHM_BADGES = {
    "dijkstra": ("Dijkstra", "badge-dijkstra"),
    "held_karp": ("Held-Karp (exacto)", "badge-held-karp"),
    "genetic": ("Algoritmo genético", "badge-genetic"),
}

KNOWN_STATUSES = {"pending", "processing", "confirmed", "failed", "cancelled"}


def inject_styles() -> None:
    """Add the custom CSS to the page."""
    st.markdown(CSS, unsafe_allow_html=True)


def algo_badge_html(algorithm: str, elapsed_ms: float = 0.0) -> str:
    """Return a pill naming the algorithm and how long it took."""
    label, css_class = ALGORITHM_BADGES.get(algorithm, ("Algoritmo", "badge-dijkstra"))
    elapsed_text = f" · {elapsed_ms:.0f} ms" if elapsed_ms > 0 else ""
    return f'<span class="algo-badge {css_class}">{label}{elapsed_text}</span>'


def status_pill_html(status: str) -> str:
    """Return a soft-colored pill for a reservation status."""
    css_status = status if status in KNOWN_STATUSES else "pending"
    return f'<span class="status-pill status-{css_status}">Estado: {status.upper()}</span>'
