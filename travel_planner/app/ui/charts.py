"""Chart conventions shared by the statistics views: algorithm names, colors and sizes."""

import altair as alt

ALGORITHM_LABELS = {"dijkstra": "Dijkstra", "held_karp": "Held-Karp", "genetic": "Genético"}
ALGORITHM_COLORS = {"Dijkstra": "#2563EB", "Held-Karp": "#059669", "Genético": "#DB2777"}
CHART_HEIGHT = 320


def algorithm_label(algorithm: str) -> str:
    return ALGORITHM_LABELS.get(algorithm, algorithm)


def algorithm_color_scale() -> alt.Scale:
    """Return the color scale that paints each algorithm like its badge."""
    return alt.Scale(domain=list(ALGORITHM_COLORS), range=list(ALGORITHM_COLORS.values()))
