"""Real usage statistics of the route algorithms, as recorded by the API."""

from typing import Any, Dict, List

import altair as alt
import pandas as pd
import streamlit as st

from app.ui.api_client import get_algorithm_stats
from app.ui.formatting import format_timestamp

ALGORITHM_LABELS = {"dijkstra": "Dijkstra", "held_karp": "Held-Karp", "genetic": "Genético"}
ALGORITHM_COLORS = {"Dijkstra": "#2563EB", "Held-Karp": "#059669", "Genético": "#DB2777"}
LATEST_RUNS_SHOWN = 15
CHART_HEIGHT = 320


def algorithm_label(algorithm: str) -> str:
    return ALGORITHM_LABELS.get(algorithm, algorithm)


def algorithm_color_scale() -> alt.Scale:
    """Return the color scale that paints each algorithm like its badge."""
    return alt.Scale(domain=list(ALGORITHM_COLORS), range=list(ALGORITHM_COLORS.values()))


def render_usage_statistics() -> None:
    """Render the KPIs, charts and latest runs of the routes computed by the users."""
    st.subheader("Estadísticas de uso real")
    st.caption("Tiempos medidos por el servidor en cada ruta calculada; las respuestas desde caché no cuentan.")
    st.button("Actualizar estadísticas", key="refresh_usage_stats")
    stats = get_algorithm_stats()
    if not stats:
        st.warning("No se pudieron obtener las estadísticas de uso desde la API.")
        return
    if not stats["total_runs"]:
        st.info("Todavía no hay ejecuciones registradas. Planificá algunas rutas en «Ruta Multidestino» y volvé acá.")
        return
    render_usage_kpis(stats)
    chart_column, bar_column = st.columns([0.6, 0.4])
    with chart_column:
        render_elapsed_by_city_count_chart(stats["by_city_count"])
    with bar_column:
        render_runs_per_algorithm_chart(stats["by_algorithm"])
    render_latest_runs(stats["runs"])


def render_usage_kpis(stats: Dict[str, Any]) -> None:
    """Render the global counters and one card per algorithm with its runs and average time."""
    total_column, cache_column, average_column = st.columns(3)
    total_column.metric("Ejecuciones reales", stats["total_runs"])
    cache_column.metric("Respuestas desde caché", stats["cache_hits"])
    average_column.metric("Tiempo promedio", f"{overall_average_ms(stats['runs']):.1f} ms")
    by_algorithm = stats["by_algorithm"]
    for column, (algorithm, summary) in zip(st.columns(len(by_algorithm)), by_algorithm.items()):
        column.metric(
            algorithm_label(algorithm),
            f"{summary['runs']} ejecuciones",
            f"prom. {summary['avg_elapsed_ms']:.1f} ms · máx. {summary['max_elapsed_ms']:.1f} ms",
            delta_color="off",
        )


def overall_average_ms(runs: List[Dict[str, Any]]) -> float:
    return sum(run["elapsed_ms"] for run in runs) / len(runs)


def render_elapsed_by_city_count_chart(by_city_count: List[Dict[str, Any]]) -> None:
    """Render the average elapsed time per number of cities, one line per algorithm, on a log scale."""
    st.markdown("**Tiempo promedio según cantidad de ciudades**")
    timings = (
        pd.DataFrame(by_city_count)
        .query("avg_elapsed_ms > 0")
        .assign(Algoritmo=lambda frame: frame["algorithm"].map(algorithm_label))
    )
    chart = (
        alt.Chart(timings)
        .mark_line(point=alt.OverlayMarkDef(size=70))
        .encode(
            x=alt.X("n_cities:Q", title="Cantidad de ciudades (n)", axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("avg_elapsed_ms:Q", title="Tiempo promedio (ms, escala log)", scale=alt.Scale(type="log")),
            color=alt.Color("Algoritmo:N", scale=algorithm_color_scale()),
            tooltip=[
                "Algoritmo",
                alt.Tooltip("n_cities:Q", title="Ciudades"),
                alt.Tooltip("runs:Q", title="Ejecuciones"),
                alt.Tooltip("avg_elapsed_ms:Q", title="Promedio (ms)", format=".2f"),
            ],
        )
        .properties(height=CHART_HEIGHT)
    )
    st.altair_chart(chart, use_container_width=True)


def render_runs_per_algorithm_chart(by_algorithm: Dict[str, Dict[str, float]]) -> None:
    """Render how many times each algorithm actually ran."""
    st.markdown("**Ejecuciones por algoritmo**")
    runs = pd.DataFrame(
        [{"Algoritmo": algorithm_label(algorithm), "Ejecuciones": summary["runs"]}
         for algorithm, summary in by_algorithm.items()]
    )
    chart = (
        alt.Chart(runs)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Algoritmo:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Ejecuciones:Q", axis=alt.Axis(tickMinStep=1)),
            color=alt.Color("Algoritmo:N", scale=algorithm_color_scale(), legend=None),
            tooltip=["Algoritmo", "Ejecuciones"],
        )
        .properties(height=CHART_HEIGHT)
    )
    st.altair_chart(chart, use_container_width=True)


def render_latest_runs(runs: List[Dict[str, Any]]) -> None:
    """Render the most recent runs as a table."""
    st.markdown("**Últimas ejecuciones**")
    latest_runs = pd.DataFrame(
        [
            {
                "Fecha": format_timestamp(run["timestamp"]),
                "Algoritmo": algorithm_label(run["algorithm"]),
                "Ciudades": run["n_cities"],
                "Tiempo (ms)": round(run["elapsed_ms"], 2),
                "Costo": round(run["total_cost"], 2),
            }
            for run in runs[:LATEST_RUNS_SHOWN]
        ]
    )
    st.dataframe(latest_runs, use_container_width=True, hide_index=True)
