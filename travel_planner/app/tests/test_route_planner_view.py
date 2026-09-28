"""Streamlit AppTest of the route planner page with the API mocked: criterion selector, cost and time."""

from unittest.mock import MagicMock

import pytest

from app.ui.route_matrices import RouteMatrices

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

CITIES = ["Madrid", "París", "Roma"]
FAKE_MATRICES = RouteMatrices(
    cities=CITIES,
    cost=[[0.0, 150.0, 200.0], [150.0, 0.0, 170.0], [200.0, 170.0, 0.0]],
    time=[[0.0, 2.5, 3.0], [2.5, 0.0, 2.8], [3.0, 2.8, 0.0]],
    legs=[[0, 1, 2], [1, 0, 1], [2, 1, 0]],
    tolls=[[0.0, 40.0, 55.0], [40.0, 0.0, 45.0], [55.0, 45.0, 0.0]],
)
TSP_ANSWER = {
    "algorithm": "held_karp",
    "optimal_route": ["Madrid", "París", "Roma", "Madrid"],
    "total_cost": 8.3,
    "elapsed_ms": 1.0,
    "history": None,
    "cached": False,
}
tsp_solver = MagicMock(return_value=TSP_ANSWER)


def fake_shortest_route(origin, destination, _transport, _optimize_by):
    return {
        "path": [origin, destination],
        "total_cost": FAKE_MATRICES.cost_between(origin, destination),
        "total_hours": FAKE_MATRICES.time_between(origin, destination),
        "cached": False,
    }


def render_planner_with_mocked_api():
    from unittest.mock import patch

    from app.tests.test_route_planner_view import FAKE_MATRICES, fake_shortest_route, tsp_solver
    from app.ui.state import init_state
    from app.ui.views import route_planner

    init_state()
    with patch.object(route_planner, "load_route_matrices", return_value=FAKE_MATRICES), \
            patch.object(route_planner, "calculate_shortest_route", side_effect=fake_shortest_route), \
            patch.object(route_planner, "optimize_multi_destination", tsp_solver), \
            patch.object(route_planner, "compare_transports", return_value=None), \
            patch.object(route_planner, "show_map"):
        route_planner.render_route_planner()


def planner_with(cities, optimize_by, transport="avión"):
    app_test = AppTest.from_function(render_planner_with_mocked_api, default_timeout=30).run()
    app_test.selectbox(key="transport_mode").set_value(transport)
    app_test.selectbox(key="optimize_by").set_value(optimize_by)
    app_test.multiselect(key="selected_cities").set_value(cities)
    app_test.run()
    app_test.button(key="compute_routes").click().run()
    return app_test


def metric_values(app_test, label):
    return [metric.value for metric in app_test.metric if metric.label == label]


def test_criterion_selector_offers_price_and_time():
    app_test = AppTest.from_function(render_planner_with_mocked_api, default_timeout=30).run()

    selector = app_test.selectbox(key="optimize_by")
    assert selector.label == "Optimizar por"
    assert selector.options == ["Precio", "Tiempo"]


def test_two_city_route_shows_cost_and_time():
    app_test = planner_with(["Madrid", "Roma"], "cost")

    assert not app_test.exception
    assert metric_values(app_test, "Costo en €") == ["400.00 €", "400.00 €"]
    assert metric_values(app_test, "Tiempo total") == ["6.0 h", "6.0 h"]


def test_multi_city_route_optimized_by_time_uses_the_time_matrix():
    tsp_solver.reset_mock()

    app_test = planner_with(CITIES, "time")

    assert not app_test.exception
    submitted_matrix = tsp_solver.call_args.args[1]
    assert submitted_matrix == FAKE_MATRICES.submatrix(CITIES, "time")
    assert metric_values(app_test, "Tu Ruta") == ["8.3 h"]
    assert metric_values(app_test, "Tiempo total") == ["8.3 h", "8.3 h"]
    assert metric_values(app_test, "Costo en €") == ["520.00 €", "520.00 €"]


def test_car_route_shows_the_tolls_included_in_its_cost():
    app_test = planner_with(["Madrid", "Roma"], "cost", transport="auto")

    assert not app_test.exception
    assert [caption.value for caption in app_test.caption if caption.value.startswith("Peajes")] == [
        "Peajes: 110.00 € (incluidos en el costo)",
        "Peajes: 110.00 € (incluidos en el costo)",
    ]
