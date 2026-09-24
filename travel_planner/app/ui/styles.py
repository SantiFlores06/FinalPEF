"""Custom CSS and small HTML snippets used across the views."""

from html import escape

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

BASE_CSS = """
html { font-size: 17px; }
[data-testid="stHeading"] h2, [data-testid="stHeading"] h3 { color: #312E81; font-weight: 700; }
[data-testid="stHeading"] h3 { border-left: 5px solid #7C3AED; padding-left: 0.65rem; }
[data-testid="stCaptionContainer"] { color: #475569; }
hr { border-color: #C3C9E8; }
"""

SIDEBAR_CSS = """
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E1B4B 0%, #312E81 45%, #4338CA 100%);
}
[data-testid="stSidebar"][aria-expanded="true"] {
    width: 232px !important;
    min-width: 232px !important;
    max-width: 232px !important;
}
[data-testid="stSidebar"] :is(h1, h2, h3, p, label, span, small) { color: #F8FAFC; }
[data-testid="stSidebar"] hr { border-color: rgba(255, 255, 255, 0.25); }
[data-testid="stSidebar"] label[data-baseweb="radio"] {
    width: 100%;
    padding: 0.45rem 0.75rem;
    margin-bottom: 0.25rem;
    border-radius: 10px;
    transition: background 0.15s ease;
}
[data-testid="stSidebar"] label[data-baseweb="radio"]:hover { background: rgba(255, 255, 255, 0.10); }
[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) {
    background: rgba(255, 255, 255, 0.22);
    font-weight: 600;
}
"""

HERO_CSS = """
.page-hero {
    background: linear-gradient(120deg, #4F46E5 0%, #7C3AED 55%, #DB2777 100%);
    border-radius: 18px;
    padding: 1.6rem 2rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 10px 28px rgba(79, 70, 229, 0.28);
}
.page-hero .page-hero-title { color: #FFFFFF; font-size: 2rem; font-weight: 800; line-height: 1.2; }
.page-hero .page-hero-subtitle { color: rgba(255, 255, 255, 0.92); font-size: 1.05rem; margin-top: 0.4rem; }
"""

CARD_CSS = """
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #C3C9E8;
    border-left: 5px solid #4F46E5;
    border-radius: 14px;
    padding: 0.9rem 1.1rem;
    box-shadow: 0 6px 16px rgba(30, 27, 75, 0.12);
}
[data-testid="stMetricLabel"] p { color: #475569; font-weight: 600; }
[data-testid="stMetricValue"] { color: #1E1B4B; font-weight: 700; }
[data-testid="stElementContainer"]:has(:is(.card-anchor, .map-anchor)) { display: none; }
[data-testid="stVerticalBlockBorderWrapper"]:has(
    > div > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] .card-anchor
) {
    background: #FFFFFF;
    border: 1px solid #C3C9E8;
    border-top: 4px solid #7C3AED;
    border-radius: 16px;
    box-shadow: 0 8px 22px rgba(30, 27, 75, 0.14);
}
[data-testid="stExpander"] details,
[data-testid="stDataFrame"],
[data-testid="stVegaLiteChart"] {
    background: #FFFFFF;
    border: 1px solid #C3C9E8;
    border-radius: 12px;
}
[data-testid="stVegaLiteChart"] { padding: 0.6rem; box-shadow: 0 4px 12px rgba(30, 27, 75, 0.08); }
"""

MAP_FRAME_CSS = """
[data-testid="stVerticalBlockBorderWrapper"]:has(
    > div > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] .map-anchor
) {
    background: #FFFFFF;
    border: 1px solid #A5B4FC;
    border-radius: 16px;
    padding: 0.4rem;
    box-shadow: 0 8px 22px rgba(30, 27, 75, 0.16);
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.map-anchor) iframe { border-radius: 12px; }
"""

BUTTON_CSS = """
[data-testid="stBaseButton-primary"] {
    background: linear-gradient(90deg, #4F46E5, #7C3AED);
    border: none;
    border-radius: 10px;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.30);
}
[data-testid="stBaseButton-primary"]:hover { filter: brightness(1.08); }
[data-testid="stBaseButton-secondary"] {
    background: #FFFFFF;
    border: 1.5px solid #C7D2FE;
    border-radius: 10px;
    color: #3730A3;
    font-weight: 600;
}
[data-testid="stBaseButton-secondary"]:hover { background: #EEF2FF; border-color: #4F46E5; color: #4F46E5; }
"""

BADGE_CSS = """
.algo-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 6px;
    box-shadow: 0 2px 8px rgba(17, 24, 39, 0.15);
}
.badge-dijkstra  { background: linear-gradient(90deg, #2563EB, #3B82F6); }
.badge-held-karp { background: linear-gradient(90deg, #059669, #10B981); }
.badge-genetic   { background: linear-gradient(90deg, #DB2777, #EC4899); }
"""

RESERVATION_CSS = """
.res-card {
    background: #FFFFFF;
    border: 1px solid #C3C9E8;
    border-left: 5px solid #4F46E5;
    border-radius: 14px;
    padding: 1rem 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 6px 16px rgba(30, 27, 75, 0.12);
}
.res-header { font-size: 1.1rem; font-weight: 700; color: #1E1B4B; }
.res-sub    { font-size: 0.85rem; color: #475569; margin-bottom: 0.6rem; }

.status-pill {
    display: inline-block;
    margin-top: 0.6rem;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 700;
}
.status-pending    { background: #FDE68A; color: #78350F; }
.status-processing { background: #BFDBFE; color: #1E3A8A; }
.status-confirmed  { background: #A7F3D0; color: #064E3B; }
.status-failed     { background: #FECACA; color: #7F1D1D; }
.status-cancelled  { background: #E5E7EB; color: #374151; }
"""

STYLESHEET = "<style>{}</style>".format(
    "".join((BASE_CSS, SIDEBAR_CSS, HERO_CSS, CARD_CSS, MAP_FRAME_CSS, BUTTON_CSS, BADGE_CSS, RESERVATION_CSS))
)

CARD_ANCHOR_CLASS = "card-anchor"
MAP_ANCHOR_CLASS = "map-anchor"

ALGORITHM_BADGES = {
    "dijkstra": ("Dijkstra", "badge-dijkstra"),
    "held_karp": ("Held-Karp (exacto)", "badge-held-karp"),
    "genetic": ("Algoritmo genético", "badge-genetic"),
}

KNOWN_STATUSES = {"pending", "processing", "confirmed", "failed", "cancelled"}


def inject_styles() -> None:
    """Add the custom CSS to the page."""
    st.markdown(STYLESHEET, unsafe_allow_html=True)


def page_header_html(title: str, subtitle: str = "") -> str:
    """Return the gradient banner that opens a page."""
    subtitle_html = f'<div class="page-hero-subtitle">{escape(subtitle)}</div>' if subtitle else ""
    return f'<div class="page-hero"><div class="page-hero-title">{escape(title)}</div>{subtitle_html}</div>'


def render_page_header(title: str, subtitle: str = "") -> None:
    """Render the gradient banner that opens a page."""
    st.markdown(page_header_html(title, subtitle), unsafe_allow_html=True)


def anchored_container(anchor_class: str) -> DeltaGenerator:
    """Return a bordered container tagged with a hidden anchor, so the CSS can style it."""
    container = st.container(border=True)
    container.markdown(f'<span class="{anchor_class}"></span>', unsafe_allow_html=True)
    return container


def card_container() -> DeltaGenerator:
    """Return a bordered container styled as a white card."""
    return anchored_container(CARD_ANCHOR_CLASS)


def map_frame() -> DeltaGenerator:
    """Return a framed container that sets a map apart from the page background."""
    return anchored_container(MAP_ANCHOR_CLASS)


def algo_badge_html(algorithm: str, elapsed_ms: float = 0.0) -> str:
    """Return a pill naming the algorithm and how long it took."""
    label, css_class = ALGORITHM_BADGES.get(algorithm, ("Algoritmo", "badge-dijkstra"))
    elapsed_text = f" · {elapsed_ms:.0f} ms" if elapsed_ms > 0 else ""
    return f'<span class="algo-badge {css_class}">{label}{elapsed_text}</span>'


def status_pill_html(status: str) -> str:
    """Return a soft-colored pill for a reservation status."""
    css_status = status if status in KNOWN_STATUSES else "pending"
    return f'<span class="status-pill status-{css_status}">Estado: {status.upper()}</span>'
