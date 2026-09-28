"""Laboratory page: real usage statistics, then exact Held-Karp versus the genetic heuristic run in-process."""

import random
import time
from typing import Dict, List, Optional, Tuple

import altair as alt
import pandas as pd
import streamlit as st

from app.core.graph import TravelGraph
from app.core.tsp_dp import HELD_KARP_MAX, TSPSolver
from app.core.tsp_genetic import GeneticTSP
from app.data.cities import MAINLAND_EUROPEAN_CITIES
from app.data.routes_fixed import ROUTES_FIXED
from app.ui.styles import render_page_header
from app.ui.views.usage_stats import render_usage_statistics

LAB_TRANSPORT_MODES = ["auto", "tren", "avión"]
# Every pair of these cities is linked by all the lab transports (maybe with layovers), so any sample has a full matrix.
BENCHMARK_CITY_POOL = MAINLAND_EUROPEAN_CITIES
BENCHMARK_CITY_COUNTS = list(range(4, 15))
PROJECTION_LIMIT = 18
MEASURED = "medido"
PROJECTED = "proyectado"
HELD_KARP_GROUP = "Held-Karp"
GENETIC_GROUP = "Genético"


@st.cache_resource(show_spinner=False)
def build_route_graph() -> TravelGraph:
    """Return the graph of every fixed route, built once and shared by the benchmark runs."""
    graph = TravelGraph()
    for origin, destination, cost, hours, transport in ROUTES_FIXED:
        graph.add_route(origin, destination, cost, hours, transport)
    return graph


def build_cost_matrix(city_names: List[str], transport: str) -> List[List[float]]:
    """Return the n×n matrix of the cheapest itinerary cost between the cities for a transport."""
    graph = build_route_graph()
    matrix = []
    for origin in city_names:
        path_totals = graph.shortest_path_totals(origin, "cost", transport)
        matrix.append([float(path_totals[destination].cost) for destination in city_names])
    return matrix


def run_held_karp(matrix: List[List[float]], city_names: List[str]) -> Tuple[float, float]:
    """Run Held-Karp and return (cost, elapsed milliseconds)."""
    started_at = time.perf_counter()
    cost, _route = TSPSolver(matrix, city_names).solve(start_city=0, return_to_start=True)
    return cost, (time.perf_counter() - started_at) * 1000


def run_genetic(matrix: List[List[float]], city_names: List[str],
                population: int = 150, generations: int = 300) -> Tuple[float, float]:
    """Run the genetic algorithm and return (cost, elapsed milliseconds)."""
    genetic_solver = GeneticTSP(matrix, city_names, population_size=population, generations=generations)
    cost, _route = genetic_solver.solve(start_city=0, return_to_start=True)
    return cost, genetic_solver.elapsed_ms


def render_laboratory() -> None:
    """Render the laboratory page."""
    render_page_header(
        "Laboratorio de algoritmos",
        "Cómo rinden Dijkstra, Held-Karp y el algoritmo genético con el uso real y en un benchmark controlado.",
    )
    render_usage_statistics()
    st.divider()
    render_controlled_benchmark()


def render_controlled_benchmark() -> None:
    """Render the in-process comparison and benchmark of Held-Karp against the genetic algorithm."""
    st.subheader("Benchmark controlado: Held-Karp vs Algoritmo Genético")
    st.write(
        "Compará el algoritmo **exacto** (Held-Karp, O(n²·2ⁿ)) contra el **heurístico** "
        "(genético). La idea es ver empíricamente *dónde el exacto deja de escalar* y el "
        "heurístico se vuelve la única opción práctica."
    )
    transport = st.selectbox("Transporte", LAB_TRANSPORT_MODES, key="lab_transport")
    count_column, seed_column = st.columns(2)
    city_count = count_column.slider("Cantidad de ciudades (n)", 2, 20, 8, key="lab_n")
    seed = int(seed_column.number_input("Semilla (reproducibilidad)", value=42, step=1, key="lab_seed"))
    st.caption(
        f"Held-Karp corre solo hasta n={HELD_KARP_MAX}. Por encima, se muestra únicamente "
        "el genético (el exacto se vuelve inviable). Las ciudades se sortean entre las de "
        "Europa continental, conectadas por los tres transportes; cada par usa su ruta más barata "
        "(en avión puede incluir escalas)."
    )
    if st.button("Comparar", type="primary", use_container_width=True, key="lab_compare"):
        render_single_comparison(transport, city_count, seed)
    st.divider()
    st.markdown("#### Benchmark completo")
    st.caption(
        "Corre ambos algoritmos para n = 4 … 14 y grafica el tiempo con **eje Y "
        "logarítmico**. Ahí se ve la curva exponencial de Held-Karp despegar."
    )
    if st.button("Correr benchmark", use_container_width=True, key="lab_benchmark"):
        render_benchmark(transport, seed)


def comparison_row(algorithm: str, cost: float, elapsed_ms: float, gap_percent: Optional[float]) -> Dict:
    """Return one row of the single comparison table."""
    return {
        "Algoritmo": algorithm,
        "Costo": round(cost),
        "Tiempo (ms)": round(elapsed_ms, 2),
        "Gap %": None if gap_percent is None else round(gap_percent, 2),
    }


def render_single_comparison(transport: str, city_count: int, seed: int) -> None:
    """Run both algorithms on random cities and render the comparison."""
    city_names = random.Random(seed).sample(BENCHMARK_CITY_POOL, city_count)
    matrix = build_cost_matrix(city_names, transport)
    rows = []
    held_karp_cost = None
    with st.spinner("Ejecutando algoritmos..."):
        if city_count <= HELD_KARP_MAX:
            held_karp_cost, held_karp_ms = run_held_karp(matrix, city_names)
            rows.append(comparison_row("Held-Karp (exacto)", held_karp_cost, held_karp_ms, 0.0))
        else:
            st.warning(
                f"Held-Karp no es viable para n={city_count}: requeriría explorar 2^{city_count} "
                f"= {2 ** city_count:,} estados. Solo se ejecuta el genético."
            )
        genetic_cost, genetic_ms = run_genetic(matrix, city_names)
    gap_percent = None if held_karp_cost is None else (genetic_cost - held_karp_cost) / held_karp_cost * 100
    rows.append(comparison_row("Genético (heurístico)", genetic_cost, genetic_ms, gap_percent))
    st.subheader("Resultado")
    st.caption("Ciudades: " + " · ".join(city_names))
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if held_karp_cost is None:
        return
    if round(genetic_cost) == round(held_karp_cost):
        st.success(f"El genético alcanzó el óptimo exacto (costo {round(held_karp_cost)}).")
    else:
        st.info(
            f"El genético quedó a un **{gap_percent:.2f}%** del óptimo "
            f"(exacto: {round(held_karp_cost)}, genético: {round(genetic_cost)})."
        )


def timing_record(city_count: int, group: str, series: str, elapsed_ms: float) -> Dict:
    """Return one timing point of the benchmark."""
    return {"n": city_count, "Grupo": group, "Serie": series, "Tiempo (ms)": elapsed_ms}


def measure_benchmark(transport: str, seed: int) -> Tuple[List[Dict], Dict[int, float], List[float]]:
    """Time both algorithms for every benchmark size, showing progress."""
    rng = random.Random(seed)
    records: List[Dict] = []
    held_karp_times: Dict[int, float] = {}
    genetic_times: List[float] = []
    progress = st.progress(0.0, text="Corriendo benchmark...")
    for position, city_count in enumerate(BENCHMARK_CITY_COUNTS, start=1):
        city_names = rng.sample(BENCHMARK_CITY_POOL, city_count)
        matrix = build_cost_matrix(city_names, transport)
        if city_count <= HELD_KARP_MAX:
            _cost, held_karp_ms = run_held_karp(matrix, city_names)
            records.append(timing_record(city_count, HELD_KARP_GROUP, MEASURED, held_karp_ms))
            held_karp_times[city_count] = held_karp_ms
        _cost, genetic_ms = run_genetic(matrix, city_names)
        records.append(timing_record(city_count, GENETIC_GROUP, MEASURED, genetic_ms))
        genetic_times.append(genetic_ms)
        progress.progress(position / len(BENCHMARK_CITY_COUNTS), text=f"n = {city_count}")
    progress.empty()
    return records, held_karp_times, genetic_times


def project_timings(held_karp_times: Dict[int, float], genetic_average_ms: float) -> Tuple[List[Dict], Optional[int]]:
    """Extrapolate Held-Karp as C·n²·2ⁿ and the genetic as flat, returning the estimated crossover."""
    largest_measured = max(held_karp_times)
    constant = held_karp_times[largest_measured] / (largest_measured ** 2 * 2 ** largest_measured)
    projected_sizes = range(largest_measured, PROJECTION_LIMIT)
    records = [
        timing_record(size, HELD_KARP_GROUP, PROJECTED, constant * size ** 2 * 2 ** size) for size in projected_sizes
    ]
    records += [
        timing_record(size, GENETIC_GROUP, PROJECTED, genetic_average_ms)
        for size in range(max(BENCHMARK_CITY_COUNTS), PROJECTION_LIMIT)
    ]
    crossover = next((size for size in projected_sizes if constant * size ** 2 * 2 ** size >= genetic_average_ms), None)
    return records, crossover


def benchmark_chart(timings: pd.DataFrame) -> alt.Chart:
    """Return the log-scale timing chart, solid when measured and dashed when projected."""
    return (
        alt.Chart(timings)
        .mark_line(point=True)
        .encode(
            x=alt.X("n:Q", title="Cantidad de ciudades (n)"),
            y=alt.Y("Tiempo (ms):Q", title="Tiempo de ejecución (ms, escala log)", scale=alt.Scale(type="log")),
            color=alt.Color("Grupo:N", title="Algoritmo"),
            strokeDash=alt.StrokeDash("Serie:N", title="", sort=[MEASURED, PROJECTED]),
            tooltip=["n", "Grupo", "Serie", alt.Tooltip("Tiempo (ms):Q", format=".2f")],
        )
        .properties(height=420)
    )


def render_benchmark(transport: str, seed: int) -> None:
    """Run the benchmark and render its chart, crossover and measured table."""
    records, held_karp_times, genetic_times = measure_benchmark(transport, seed)
    projected_records, crossover = project_timings(held_karp_times, sum(genetic_times) / len(genetic_times))
    timings = pd.DataFrame(records + projected_records)
    st.altair_chart(benchmark_chart(timings), use_container_width=True)
    if crossover is not None:
        st.info(
            f"**Cruce estimado en n ≈ {crossover}**: a partir de ahí el exacto "
            f"(línea punteada, proyectada con t = C·n²·2ⁿ) tardaría más que el "
            f"genético. Held-Karp se corta en n={HELD_KARP_MAX} porque más allá "
            f"deja de ser viable en la práctica."
        )
    measured_table = (
        timings[timings["Serie"] == MEASURED]
        .pivot(index="n", columns="Grupo", values="Tiempo (ms)")
        .round(2)
    )
    st.dataframe(measured_table, use_container_width=True)
    st.caption(
        "Líneas sólidas = tiempos medidos. Líneas punteadas = extrapolación "
        "(Held-Karp por su complejidad teórica; genético como referencia plana)."
    )
