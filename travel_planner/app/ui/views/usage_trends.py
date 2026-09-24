"""How the route algorithms and the reservation batches behave over time, as recorded by the API."""

from typing import Any, Dict, List

import altair as alt
import pandas as pd
import streamlit as st

from app.ui.charts import CHART_HEIGHT, algorithm_color_scale, algorithm_label

PERCENT = 100
POINT_OPACITY = 0.35
ROLLING_LINE_WIDTH = 3
LATENCY_SERIES = {"avg_latency_ms": "Promedio", "max_latency_ms": "Máximo"}


def with_algorithm_label(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(Algoritmo=frame["algorithm"].map(algorithm_label))


def render_usage_trends(stats: Dict[str, Any]) -> None:
    """Render the time series, percentiles, throughput, cache hit rate and genetic quality."""
    st.markdown("#### Evolución en el tiempo")
    render_elapsed_trend(stats["elapsed_series"], stats["rolling_window"])
    render_latency_table(stats["latency_percentiles"], stats["efficiency"])
    throughput_column, hit_rate_column = st.columns(2)
    with throughput_column:
        render_throughput_chart(stats["per_minute"])
    with hit_rate_column:
        render_cache_hit_rate_chart(stats["per_minute"])
    render_genetic_quality(stats["genetic_quality"])


def render_elapsed_trend(elapsed_series: List[Dict[str, Any]], rolling_window: int) -> None:
    """Render every run's time as a dot and its algorithm's rolling mean as a line."""
    st.markdown(f"**Tiempo por ejecución y media móvil ({rolling_window} ejecuciones)**")
    st.caption("Si la línea sube con el tiempo para el mismo algoritmo, el servidor se está degradando.")
    series = with_algorithm_label(pd.DataFrame(elapsed_series)).query("elapsed_ms > 0")
    base = alt.Chart(series).encode(
        x=alt.X("timestamp:T", title="Momento de la ejecución"),
        color=alt.Color("Algoritmo:N", scale=algorithm_color_scale()),
    )
    runs = base.mark_circle(size=45, opacity=POINT_OPACITY).encode(
        y=alt.Y("elapsed_ms:Q", title="Tiempo (ms, escala log)", scale=alt.Scale(type="log")),
        tooltip=[
            "Algoritmo",
            alt.Tooltip("run_number:Q", title="Ejecución #"),
            alt.Tooltip("elapsed_ms:Q", title="Tiempo (ms)", format=".2f"),
            alt.Tooltip("rolling_mean_ms:Q", title="Media móvil (ms)", format=".2f"),
        ],
    )
    rolling_mean = base.mark_line(strokeWidth=ROLLING_LINE_WIDTH).encode(y="rolling_mean_ms:Q")
    st.altair_chart((runs + rolling_mean).properties(height=CHART_HEIGHT), use_container_width=True)


def render_latency_table(percentiles: Dict[str, Dict[str, float]], efficiency: Dict[str, Dict[str, float]]) -> None:
    """Render the latency percentiles and the per-city efficiency of every algorithm."""
    st.markdown("**Latencia y eficiencia por algoritmo**")
    rows = [
        {
            "Algoritmo": algorithm_label(algorithm),
            "p50 (ms)": latency["p50_ms"],
            "p95 (ms)": latency["p95_ms"],
            "Máximo (ms)": latency["max_ms"],
            "ms por ciudad": efficiency[algorithm]["avg_ms_per_city"],
            "Costo por ciudad": efficiency[algorithm]["avg_cost_per_city"],
        }
        for algorithm, latency in percentiles.items()
    ]
    st.dataframe(pd.DataFrame(rows).round(2), use_container_width=True, hide_index=True)


def minute_frame(per_minute: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(per_minute).assign(minute=lambda frame: pd.to_datetime(frame["minute"]))


def render_throughput_chart(per_minute: List[Dict[str, Any]]) -> None:
    """Render how many routes were actually computed per minute."""
    st.markdown("**Throughput (ejecuciones por minuto)**")
    chart = (
        alt.Chart(minute_frame(per_minute))
        .mark_line(point=True, color="#4F46E5")
        .encode(
            x=alt.X("minute:T", title="Minuto"),
            y=alt.Y("runs:Q", title="Ejecuciones", axis=alt.Axis(tickMinStep=1)),
            tooltip=[alt.Tooltip("minute:T", title="Minuto", format="%H:%M"), alt.Tooltip("runs:Q", title="Ejecuciones")],
        )
        .properties(height=CHART_HEIGHT)
    )
    st.altair_chart(chart, use_container_width=True)


def render_cache_hit_rate_chart(per_minute: List[Dict[str, Any]]) -> None:
    """Render the share of requests answered from the cache per minute."""
    st.markdown("**Tasa de aciertos de caché por minuto**")
    hit_rates = minute_frame(per_minute).assign(hit_rate_percent=lambda frame: frame["hit_rate"] * PERCENT)
    chart = (
        alt.Chart(hit_rates)
        .mark_line(point=True, color="#0891B2")
        .encode(
            x=alt.X("minute:T", title="Minuto"),
            y=alt.Y("hit_rate_percent:Q", title="Aciertos (%)", scale=alt.Scale(domain=[0, PERCENT])),
            tooltip=[
                alt.Tooltip("minute:T", title="Minuto", format="%H:%M"),
                alt.Tooltip("cache_hits:Q", title="Desde caché"),
                alt.Tooltip("runs:Q", title="Calculadas"),
                alt.Tooltip("hit_rate_percent:Q", title="Aciertos (%)", format=".1f"),
            ],
        )
        .properties(height=CHART_HEIGHT)
    )
    st.altair_chart(chart, use_container_width=True)


def render_genetic_quality(genetic_quality: List[Dict[str, Any]]) -> None:
    """Render when each genetic run converged, how much it improved and its gap to the optimum."""
    st.markdown("**Calidad del algoritmo genético**")
    if not genetic_quality:
        st.caption("Todavía no hay ejecuciones del algoritmo genético.")
        return
    st.caption(
        "Generación de convergencia: la primera en la que el mejor costo dejó de mejorar. "
        "El gap contra el óptimo solo aparece si Held-Karp ya resolvió esas mismas ciudades "
        "(por ejemplo, con «Ejecutar algoritmo genético» en la ruta multidestino)."
    )
    quality = pd.DataFrame(genetic_quality)
    chart = (
        alt.Chart(quality)
        .mark_line(point=alt.OverlayMarkDef(size=70), color="#DB2777")
        .encode(
            x=alt.X("timestamp:T", title="Momento de la ejecución"),
            y=alt.Y("convergence_generation:Q", title="Generación de convergencia"),
            tooltip=[
                alt.Tooltip("n_cities:Q", title="Ciudades"),
                alt.Tooltip("convergence_generation:Q", title="Convergencia"),
                alt.Tooltip("improvement_percent:Q", title="Mejora (%)", format=".1f"),
                alt.Tooltip("gap_percent:Q", title="Gap vs óptimo (%)", format=".2f"),
            ],
        )
        .properties(height=CHART_HEIGHT)
    )
    st.altair_chart(chart, use_container_width=True)
    render_genetic_quality_kpis(quality)


def render_genetic_quality_kpis(quality: pd.DataFrame) -> None:
    """Render the average improvement and the average gap to the optimum of the genetic runs."""
    improvement_column, gap_column = st.columns(2)
    improvement_column.metric("Mejora promedio (1ª → última generación)", format_percent(quality["improvement_percent"].mean()))
    gap_column.metric("Gap promedio vs Held-Karp", format_percent(quality["gap_percent"].mean()))


def format_percent(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.2f}%"


def render_reservation_latency(recent_batches: List[Dict[str, Any]]) -> None:
    """Render how long the queued reservations waited until their batch was processed, and the batch sizes."""
    st.markdown("#### Reservas por lote")
    if not recent_batches:
        st.caption("Todavía no se procesaron lotes de reservas.")
        return
    batches = pd.DataFrame(recent_batches).assign(timestamp=lambda frame: pd.to_datetime(frame["timestamp"]))
    latency_column, size_column = st.columns(2)
    with latency_column:
        st.markdown("**Latencia de encolado a procesado (ms)**")
        st.altair_chart(batch_latency_chart(batches), use_container_width=True)
    with size_column:
        st.markdown("**Tamaño de cada lote**")
        st.altair_chart(batch_size_chart(batches), use_container_width=True)


def batch_latency_chart(batches: pd.DataFrame) -> alt.Chart:
    """Return the average and maximum wait of every batch over time."""
    latencies = batches.melt(
        id_vars=["timestamp"], value_vars=list(LATENCY_SERIES), var_name="series", value_name="latency_ms"
    ).assign(Serie=lambda frame: frame["series"].map(LATENCY_SERIES))
    return (
        alt.Chart(latencies)
        .mark_line(point=True)
        .encode(
            x=alt.X("timestamp:T", title="Lote procesado"),
            y=alt.Y("latency_ms:Q", title="Latencia (ms)"),
            color=alt.Color("Serie:N", title=""),
            tooltip=["Serie", alt.Tooltip("latency_ms:Q", title="Latencia (ms)", format=".0f")],
        )
        .properties(height=CHART_HEIGHT)
    )


def batch_size_chart(batches: pd.DataFrame) -> alt.Chart:
    """Return the number of reservations in every batch over time."""
    return (
        alt.Chart(batches)
        .mark_bar(color="#7C3AED")
        .encode(
            x=alt.X("timestamp:T", title="Lote procesado"),
            y=alt.Y("size:Q", title="Reservas", axis=alt.Axis(tickMinStep=1)),
            tooltip=[alt.Tooltip("size:Q", title="Reservas")],
        )
        .properties(height=CHART_HEIGHT)
    )
