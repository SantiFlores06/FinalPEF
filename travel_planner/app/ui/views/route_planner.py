"""Multi-destination route planner page."""

import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from app.ai.gemini_recommendations import generate_recommendations_for_cities
from app.core.tsp_dp import HELD_KARP_MAX, MAX_TSP_CITIES
from app.data.routes_fixed import ROUTES_FIXED, TRANSPORT_TYPES
from app.ui.api_client import (
    ApiError,
    calculate_shortest_route,
    create_reservation,
    create_reservations_batch,
    optimize_multi_destination,
)
from app.ui.formatting import format_route, format_transport
from app.ui.maps import GENETIC_ROUTE_COLOR, OPTIMAL_ROUTE_COLOR, USER_ROUTE_COLOR, build_route_map, show_map
from app.ui.route_matrices import RouteMatrices, load_route_matrices
from app.ui.state import RESERVATIONS_PAGE, mark_batch_submitted, request_page, reset_route_results
from app.ui.styles import algo_badge_html, card_container, render_page_header

OPTIMIZE_BY_LABELS = {"cost": "Costo (€)", "time": "Tiempo (h)"}
ALGORITHM_NAMES = {"dijkstra": "Dijkstra", "held_karp": "Held-Karp (óptimo)", "genetic": "AG heurístico"}
NEAR_OPTIMAL_GAP_PERCENT = 5
MAX_TICKETS = 20
RECOMMENDATION_COLUMNS = 3
SEGMENT_TABLE_FORMAT = {"Costo (€)": "{:.2f}", "Tiempo (h)": "{:.1f}"}


def render_route_planner() -> None:
    """Render the multi-destination route planner page."""
    render_page_header(
        "Optimización de Ruta Multidestino (TSP)",
        "Selecciona múltiples ciudades y compara la ruta optimizada con tu orden preferido.",
    )
    transport_mode, optimize_by = render_trip_options()
    available_cities = cities_served_by(transport_mode)
    if not available_cities:
        st.error(f"No hay ciudades disponibles para transporte en {format_transport(transport_mode)}")
        return
    return_to_start = render_city_selector(available_cities)
    render_compute_button(transport_mode, optimize_by, return_to_start)
    render_results(transport_mode, return_to_start)
    booking = st.session_state.selected_route_for_booking
    if booking:
        render_booking(booking, transport_mode, optimize_by)


def render_trip_options() -> Tuple[str, str]:
    """Render the transport and criterion pickers, resetting results when they change."""
    transport_mode = st.selectbox(
        "Tipo de transporte", TRANSPORT_TYPES, format_func=format_transport, key="transport_mode"
    )
    optimize_by = st.selectbox(
        "Optimizar por", list(OPTIMIZE_BY_LABELS), format_func=OPTIMIZE_BY_LABELS.get, key="optimize_by"
    )
    previous_options = (st.session_state.last_transport, st.session_state.last_optimize_by)
    if previous_options != (transport_mode, optimize_by):
        reset_route_results()
        st.session_state.last_transport = transport_mode
        st.session_state.last_optimize_by = optimize_by
    return transport_mode, optimize_by


def cities_served_by(transport_mode: str) -> List[str]:
    """Return every city reachable with a transport mode, sorted."""
    cities = set()
    for origin, destination, _cost, _time, transport in ROUTES_FIXED:
        if transport == transport_mode:
            cities.update((origin, destination))
    return sorted(cities)


def render_city_selector(available_cities: List[str]) -> bool:
    """Render the city multiselect and return whether the trip goes back to the start."""
    selected_cities = st.session_state.selected_cities
    st.session_state.selected_cities = [city for city in selected_cities if city in available_cities]
    st.subheader("Seleccionar ciudades")
    st.caption(
        f"Hasta {HELD_KARP_MAX} ciudades se resuelve con Held-Karp (exacto); "
        f"con más, con un algoritmo genético (heurístico)."
    )
    selected_cities = st.multiselect(
        "Elige las ciudades que quieres visitar:",
        available_cities,
        max_selections=MAX_TSP_CITIES,
        key="selected_cities",
    )
    count_column, return_column = st.columns([0.4, 0.6])
    count_column.metric("Ciudades seleccionadas", len(selected_cities))
    with return_column:
        return st.checkbox("Regresar al origen", value=True, key="return_to_start")


def render_compute_button(transport_mode: str, optimize_by: str, return_to_start: bool) -> None:
    """Render the compute button, or a hint while fewer than two cities are selected."""
    if len(st.session_state.selected_cities) < 2:
        st.info("Selecciona al menos 2 ciudades para calcular costos")
        return
    if st.button("Calcular costos", type="primary", use_container_width=True, key="compute_routes"):
        compute_routes(transport_mode, optimize_by, return_to_start)
    st.divider()


def compute_routes(transport_mode: str, optimize_by: str, return_to_start: bool) -> None:
    """Compute the user's route and the optimal one, then store both."""
    matrices = load_route_matrices(transport_mode)
    if matrices is None:
        st.error("No se pudieron cargar los datos de las rutas. Revisa el backend.")
        return
    selected_cities = list(st.session_state.selected_cities)
    missing_cities = matrices.missing_cities(selected_cities)
    if missing_cities:
        st.error(f"Las siguientes ciudades no están disponibles: {', '.join(missing_cities)}")
        return
    user_route = compute_user_route(selected_cities, matrices, return_to_start)
    if user_route is None:
        return
    route_cost_matrix = matrices.submatrix(selected_cities, optimize_by)
    optimal_result = request_optimal_route(
        selected_cities, route_cost_matrix, transport_mode, optimize_by, return_to_start
    )
    if optimal_result is None:
        return
    reset_route_results()
    st.session_state.user_route_result = user_route
    st.session_state.optimized_route_result = optimal_result
    st.session_state.cost_submatrix = {"cities": selected_cities, "matrix": route_cost_matrix}
    st.rerun()


def compute_user_route(
    selected_cities: List[str], matrices: RouteMatrices, return_to_start: bool
) -> Optional[Dict[str, Any]]:
    """Return the route in the user's order, or None when a segment is unreachable."""
    route = list(selected_cities)
    for origin, destination in zip(route, route[1:]):
        if not matrices.is_connected(origin, destination):
            st.error(f"No hay conexión entre {origin} y {destination}")
            return None
    if return_to_start and matrices.is_connected(route[-1], route[0]):
        route.append(route[0])
    return {"route": route, "total_cost": matrices.route_cost(route)}


def build_route_result(
    algorithm: str,
    optimal_route: List[str],
    total_cost: float,
    elapsed_ms: float,
    history: Optional[List[Dict]] = None,
    cached: bool = False,
) -> Dict[str, Any]:
    """Return a route result with the same keys whatever algorithm produced it."""
    return {
        "algorithm": algorithm,
        "optimal_route": optimal_route,
        "total_cost": total_cost,
        "elapsed_ms": elapsed_ms,
        "history": history,
        "cached": cached,
    }


def request_optimal_route(
    selected_cities: List[str],
    route_cost_matrix: List[List[float]],
    transport_mode: str,
    optimize_by: str,
    return_to_start: bool,
) -> Optional[Dict[str, Any]]:
    """Ask the API for the best route: shortest path for two cities, TSP otherwise."""
    if len(selected_cities) == 2:
        origin, destination = selected_cities
        with st.spinner("Calculando camino mínimo con Dijkstra..."):
            shortest_result = request_shortest_route(origin, destination, transport_mode, optimize_by, return_to_start)
        if shortest_result is None:
            st.error("No se pudo calcular la ruta optimizada")
        return shortest_result
    with st.spinner("Calculando la ruta óptima..."):
        return request_tsp_route(selected_cities, route_cost_matrix, return_to_start)


def request_shortest_route(
    origin: str, destination: str, transport_mode: str, optimize_by: str, return_to_start: bool
) -> Optional[Dict[str, Any]]:
    """Return the Dijkstra route between two cities, including the way back when asked."""
    started_at = time.perf_counter()
    outbound = calculate_shortest_route(origin, destination, transport_mode, optimize_by)
    if outbound is None:
        return None
    route = outbound["path"]
    total_cost = outbound["total_cost"]
    if return_to_start:
        inbound = calculate_shortest_route(destination, origin, transport_mode, optimize_by)
        if inbound:
            route = route + inbound["path"][1:]
            total_cost += inbound["total_cost"]
        else:
            st.error("No se pudo calcular el regreso al origen")
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    return build_route_result("dijkstra", route, total_cost, elapsed_ms, cached=outbound.get("cached", False))


def request_tsp_route(
    cities: List[str],
    route_cost_matrix: List[List[float]],
    return_to_start: bool,
    algorithm: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Return the TSP route solved by the API, showing its error message on failure."""
    try:
        response = optimize_multi_destination(cities, route_cost_matrix, return_to_start, algorithm)
    except ApiError as error:
        st.error(str(error))
        return None
    return build_route_result(
        response["algorithm"],
        response["optimal_route"],
        response["total_cost"],
        response["elapsed_ms"],
        response.get("history"),
        response.get("cached", False),
    )


def render_results(transport_mode: str, return_to_start: bool) -> None:
    """Render the comparison, maps, algorithm details and recommendations of computed routes."""
    user_route = st.session_state.user_route_result
    optimal_result = st.session_state.optimized_route_result
    if not user_route or not optimal_result:
        return
    matrices = load_route_matrices(transport_mode)
    if matrices is None:
        st.error("No se pudieron cargar los datos de las rutas. Revisa el backend.")
        return
    render_route_summary(user_route["route"], optimal_result, matrices)
    render_route_maps(user_route["route"], optimal_result, matrices, transport_mode)
    render_algorithm_details(optimal_result, return_to_start, transport_mode)
    booking = st.session_state.selected_route_for_booking
    if booking:
        render_city_recommendations(booking["route"])
        st.divider()


def render_route_summary(user_route: List[str], optimal_result: Dict[str, Any], matrices: RouteMatrices) -> None:
    """Render both routes side by side with their costs, and the comparative summary."""
    optimal_route = optimal_result["optimal_route"]
    algorithm = optimal_result["algorithm"]
    user_cost = matrices.route_cost(user_route)
    optimal_cost = matrices.route_cost(optimal_route)
    st.subheader("Comparación de rutas")
    user_column, optimal_column = st.columns(2)
    with user_column:
        st.markdown("### Tu ruta seleccionada")
        render_route_details(user_route, user_cost, matrices)
        render_select_button(user_route, user_cost, "user_selected", "select_user_route")
    with optimal_column:
        st.markdown("### Ruta heurística (AG)" if algorithm == "genetic" else "### Ruta optimizada")
        st.markdown(algo_badge_html(algorithm, optimal_result["elapsed_ms"]), unsafe_allow_html=True)
        render_route_details(optimal_route, optimal_cost, matrices)
        render_savings(user_cost, optimal_cost)
        render_select_button(optimal_route, optimal_cost, "optimized", "select_opt_route")
    render_cost_comparison(user_cost, optimal_cost, ALGORITHM_NAMES.get(algorithm, algorithm))


def render_route_details(route: List[str], cost: float, matrices: RouteMatrices) -> None:
    """Render the cost, the stops and the per-segment table of a route."""
    st.metric("Costo en €", f"{cost:.2f} €")
    st.write(format_route(route))
    with st.expander("Ver detalles por segmento"):
        segment_rows = matrices.segment_rows(route)
        if not segment_rows:
            return
        segment_table = (
            pd.DataFrame(segment_rows).style
            .format(SEGMENT_TABLE_FORMAT, na_rep="—")
            .highlight_max(subset=["Costo (€)"], color="#FEE2E2", axis=0)
            .highlight_min(subset=["Costo (€)"], color="#D1FAE5", axis=0)
        )
        st.dataframe(segment_table, use_container_width=True, hide_index=True)


def render_savings(user_cost: float, optimal_cost: float) -> None:
    """Render how much the optimal route saves compared with the user's."""
    savings = user_cost - optimal_cost
    if savings > 0:
        st.success(f"Ahorras: {savings:.2f} € ({savings / user_cost * 100:.1f}%)")
    elif savings < 0:
        st.warning(f"Cuesta más: {abs(savings):.2f} €")
    else:
        st.caption("Mismo costo que tu ruta")


def render_select_button(route: List[str], cost: float, route_type: str, key: str) -> None:
    """Render the button that picks a route for booking."""
    if st.button("Seleccionar esta ruta", key=key, use_container_width=True):
        st.session_state.selected_route_for_booking = {"route": route, "total_cost": cost, "type": route_type}
        st.rerun()


def render_cost_comparison(user_cost: float, optimal_cost: float, optimal_label: str) -> None:
    """Render the cost metrics and bar chart of both routes."""
    st.divider()
    st.subheader("Resumen comparativo")
    savings = user_cost - optimal_cost
    savings_percent = (savings / user_cost * 100) if user_cost > 0 else 0
    user_metric, optimal_metric, difference_metric = st.columns(3)
    user_metric.metric("Tu Ruta", f"{user_cost:.2f} €")
    optimal_metric.metric(
        optimal_label, f"{optimal_cost:.2f} €", delta=f"{optimal_cost - user_cost:.2f} €", delta_color="inverse"
    )
    difference_metric.metric(
        "Diferencia",
        f"{abs(savings):.2f} €",
        delta=f"{abs(savings_percent):.1f}%",
        delta_color="normal" if savings >= 0 else "off",
    )
    cost_chart = pd.DataFrame({"Costo (€)": [user_cost, optimal_cost]}, index=["Tu Ruta", optimal_label])
    st.bar_chart(cost_chart, horizontal=True, color=[USER_ROUTE_COLOR])
    st.divider()


def render_route_maps(
    user_route: List[str], optimal_result: Dict[str, Any], matrices: RouteMatrices, transport_mode: str
) -> None:
    """Render the user's route and the optimal route on two maps side by side."""
    optimal_route = optimal_result["optimal_route"]
    is_genetic = optimal_result["algorithm"] == "genetic"
    optimal_label = "Ruta AG" if is_genetic else "Ruta optimizada"
    optimal_color = GENETIC_ROUTE_COLOR if is_genetic else OPTIMAL_ROUTE_COLOR
    st.subheader("Visualización en mapa")
    user_column, optimal_column = st.columns(2)
    with user_column:
        st.markdown("**Tu ruta**")
        user_route_map = build_route_map(
            user_route, USER_ROUTE_COLOR, "Tu ruta", matrices.segment_costs(user_route), transport_mode
        )
        show_map(user_route_map, key="map_user_route")
    with optimal_column:
        st.markdown(f"**{optimal_label}**")
        optimal_route_map = build_route_map(
            optimal_route, optimal_color, optimal_label, matrices.segment_costs(optimal_route), transport_mode
        )
        show_map(optimal_route_map, key="map_opt_route")
    st.divider()


def render_algorithm_details(optimal_result: Dict[str, Any], return_to_start: bool, transport_mode: str) -> None:
    """Render the genetic convergence, or the genetic comparison when Held-Karp was used."""
    algorithm = optimal_result["algorithm"]
    if algorithm == "genetic":
        render_genetic_convergence(optimal_result)
    elif algorithm == "held_karp" and st.session_state.cost_submatrix:
        render_genetic_comparison(optimal_result, return_to_start, transport_mode)


def render_convergence_chart(history: Optional[List[Dict]]) -> None:
    """Render the best and average cost per generation."""
    if not history:
        return
    convergence = pd.DataFrame(history).set_index("generation")
    st.line_chart(convergence[["best_cost", "avg_cost"]])


def render_genetic_convergence(genetic_result: Dict[str, Any]) -> None:
    """Render why the genetic algorithm was used and how it converged."""
    city_count = len(set(genetic_result["optimal_route"]))
    st.subheader("Convergencia del algoritmo genético")
    st.caption(
        f"Con {city_count} ciudades, Held-Karp necesitaría explorar 2^{city_count} = "
        f"{2 ** city_count:,} estados, computacionalmente inviable. "
        f"El AG encontró una solución en {genetic_result['elapsed_ms']:.0f} ms."
    )
    render_convergence_chart(genetic_result["history"])
    st.divider()


def render_genetic_comparison(held_karp_result: Dict[str, Any], return_to_start: bool, transport_mode: str) -> None:
    """Render the optional run of the genetic algorithm against the exact Held-Karp result."""
    city_count = len(st.session_state.cost_submatrix["cities"])
    st.subheader("Algoritmo genético vs Held-Karp")
    st.caption(
        f"Con {city_count} ciudades, Held-Karp es exacto y rápido. "
        "Ejecuta el AG heurístico para ver cómo se compara."
    )
    if st.button("Ejecutar algoritmo genético", use_container_width=True, key="run_genetic_comparison"):
        run_genetic_comparison(return_to_start)
    genetic_result = st.session_state.ga_result
    if genetic_result:
        render_genetic_comparison_result(genetic_result, held_karp_result["total_cost"], transport_mode)
    st.divider()


def run_genetic_comparison(return_to_start: bool) -> None:
    """Ask the API for the genetic solution of the current cities and store it."""
    route_cost_matrix = st.session_state.cost_submatrix
    with st.spinner("Ejecutando algoritmo genético..."):
        genetic_result = request_tsp_route(
            route_cost_matrix["cities"], route_cost_matrix["matrix"], return_to_start, algorithm="genetic"
        )
    if genetic_result:
        st.session_state.ga_result = genetic_result
        st.rerun()


def render_genetic_comparison_result(
    genetic_result: Dict[str, Any], held_karp_cost: float, transport_mode: str
) -> None:
    """Render the genetic result next to the exact cost, with its convergence and map."""
    genetic_cost = genetic_result["total_cost"]
    gap = genetic_cost - held_karp_cost
    gap_percent = (gap / held_karp_cost * 100) if held_karp_cost > 0 else 0
    genetic_metric, held_karp_metric, gap_metric = st.columns(3)
    genetic_metric.metric("AG — Mejor ruta encontrada", f"{genetic_cost:.2f} €")
    held_karp_metric.metric("Held-Karp (exacto)", f"{held_karp_cost:.2f} €")
    gap_metric.metric("Diferencia", f"{gap:+.2f} € ({gap_percent:+.1f}%)")
    st.markdown(algo_badge_html("genetic", genetic_result["elapsed_ms"]), unsafe_allow_html=True)
    st.write(format_route(genetic_result["optimal_route"]))
    st.markdown("**Convergencia del AG** (costo por generación)")
    render_convergence_chart(genetic_result["history"])
    render_genetic_gap_verdict(gap, gap_percent)
    genetic_route_map = build_route_map(
        genetic_result["optimal_route"], GENETIC_ROUTE_COLOR, "Ruta AG", transport=transport_mode
    )
    show_map(genetic_route_map, key="map_ga_route")


def render_genetic_gap_verdict(gap: float, gap_percent: float) -> None:
    """Render how close the genetic result is to the exact optimum."""
    if gap <= 0:
        st.success("El AG encontró una ruta igual o mejor que Held-Karp.")
    elif gap_percent < NEAR_OPTIMAL_GAP_PERCENT:
        st.info(f"El AG está a solo {gap_percent:.1f}% del óptimo exacto.")
    else:
        st.warning(
            f"El AG está {gap_percent:.1f}% por encima del óptimo. "
            "Más generaciones o mayor población pueden acercarlo."
        )


def load_city_recommendations(cities: List[str]) -> Dict[str, Optional[str]]:
    """Return the AI recommendations of the cities, fetching the missing ones in parallel."""
    session_recommendations = st.session_state.city_recommendations
    missing_cities = [city for city in cities if city not in session_recommendations]
    if missing_cities:
        with st.spinner(f"Buscando lugares imperdibles en {len(missing_cities)} ciudades..."):
            session_recommendations.update(generate_recommendations_for_cities(missing_cities))
    return {city: session_recommendations[city] for city in cities}


def render_city_recommendations(route: List[str]) -> None:
    """Render one card per city of the route with the places worth visiting."""
    cities = list(dict.fromkeys(route))
    if not cities:
        return
    st.subheader("Lugares que debes visitar en cada ciudad")
    recommendations = load_city_recommendations(cities)
    for row_start in range(0, len(cities), RECOMMENDATION_COLUMNS):
        row_cities = cities[row_start:row_start + RECOMMENDATION_COLUMNS]
        for column, city in zip(st.columns(RECOMMENDATION_COLUMNS), row_cities):
            with column:
                render_city_card(city, recommendations[city])


def render_city_card(city: str, recommendations: Optional[str]) -> None:
    """Render the card of a city with its recommendations, or a note when they failed."""
    with card_container():
        st.markdown(f"#### 📍 {city}")
        if recommendations:
            st.markdown(recommendations)
        else:
            st.caption("No se pudieron generar recomendaciones para esta ciudad.")


def render_booking(booking: Dict[str, Any], transport_mode: str, optimize_by: str) -> None:
    """Render the booking form of the selected route."""
    st.subheader("Realizar reserva")
    st.markdown(
        f"**Ruta seleccionada:** {format_route(booking['route'])}  \n"
        f"**Costo total:** {booking['total_cost']:.2f} €"
    )
    ticket_count = st.number_input(
        "Cantidad de pasajes", min_value=1, max_value=MAX_TICKETS, value=1, step=1, key="ticket_count"
    )
    if st.button("Crear reserva", use_container_width=True, key="create_reservation"):
        itinerary = build_itinerary(booking, transport_mode, optimize_by)
        confirmation = submit_reservations(itinerary, int(ticket_count))
        if confirmation:
            finish_booking(confirmation)


def build_itinerary(booking: Dict[str, Any], transport_mode: str, optimize_by: str) -> Dict[str, Any]:
    """Return the itinerary sent to the API for the selected route."""
    return {
        "type": "multidestino",
        "transport_mode": transport_mode,
        "optimize_by": optimize_by,
        "cities": st.session_state.selected_cities,
        "optimal_route": booking["route"],
        "route_type": booking["type"],
        "total_cost": booking["total_cost"],
        "total_time": None,
    }


def submit_reservations(itinerary: Dict[str, Any], ticket_count: int) -> Optional[str]:
    """Create one reservation or a batch, returning a confirmation message on success."""
    user_id = st.session_state.user_id
    if ticket_count == 1:
        with st.spinner("Creando reserva..."):
            reservation = create_reservation(user_id, itinerary)
        return f"Reserva creada correctamente. ID: {reservation['reservation_id'][:12]}..." if reservation else None
    batch_payload = [{"user_id": user_id, "itinerary": itinerary} for _ in range(ticket_count)]
    with st.spinner(f"Enviando lote de {ticket_count} reservas..."):
        batch_response = create_reservations_batch(batch_payload)
    if not batch_response:
        return None
    mark_batch_submitted()
    return f"Lote creado correctamente ({batch_response['count']} reservas en cola)"


def finish_booking(confirmation: str) -> None:
    """Confirm the booking, clear the planner and go to the reservations page."""
    st.toast(confirmation)
    st.balloons()
    st.session_state.clear_selected_cities = True
    reset_route_results()
    request_page(RESERVATIONS_PAGE)
    st.rerun()
