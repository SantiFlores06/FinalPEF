"""Tests for the hints comparing the chosen transport with the other ones."""

from app.ui.transport_hints import comparison_rows, describe_against, transport_hints

CAR = {"transport": "auto", "path": ["Madrid", "Barcelona"], "total_cost": 90, "total_hours": 6.2, "toll_cost": 35}
TRAIN = {"transport": "tren", "path": ["Madrid", "Zaragoza", "Barcelona"], "total_cost": 110, "total_hours": 4.2}
PLANE = {"transport": "avión", "path": ["Madrid", "Barcelona"], "total_cost": 130, "total_hours": 1.2}
COMPARISON = {
    "origin": "Madrid",
    "destination": "Barcelona",
    "optimize_by": "cost",
    "options": [CAR, TRAIN, PLANE],
    "cheapest": "auto",
    "fastest": "avión",
}


def test_cheaper_option_leads_with_the_saving():
    assert describe_against(CAR, TRAIN) == "En auto: 20 € más barato y 2 h más lento"


def test_faster_option_leads_with_the_time_gain():
    assert describe_against(PLANE, TRAIN) == "En avión: 3 h más rápido y 20 € más caro"


def test_hints_list_only_transports_beating_the_chosen_one():
    assert transport_hints(COMPARISON, "auto") == [
        "En tren: 2 h más rápido y 20 € más caro",
        "En avión: 5 h más rápido y 40 € más caro",
    ]


def test_no_hints_when_the_chosen_transport_cannot_link_the_cities():
    assert transport_hints(COMPARISON, "barco") == []


def test_rows_mark_cheapest_and_fastest_and_count_legs():
    rows = {row["Transporte"]: row for row in comparison_rows(COMPARISON)}

    assert rows["Auto"]["Destacado"] == "Más barato"
    assert rows["Avión"]["Destacado"] == "Más rápido"
    assert rows["Tren"]["Destacado"] == ""
    assert rows["Tren"]["Tramos"] == 2


def test_rows_show_tolls_only_for_the_car():
    rows = {row["Transporte"]: row for row in comparison_rows(COMPARISON)}

    assert rows["Auto"]["Peajes (€)"] == 35
    assert rows["Tren"]["Peajes (€)"] is None


def render_comparison_page():
    from app.tests.test_transport_hints import COMPARISON
    from app.ui.views.route_planner import render_transport_comparison

    render_transport_comparison(COMPARISON, "tren")


def test_route_planner_renders_transport_comparison_with_hints():
    from streamlit.testing.v1 import AppTest

    app_test = AppTest.from_function(render_comparison_page, default_timeout=30).run()

    assert not app_test.exception
    assert [info.value for info in app_test.info] == [
        "En auto: 20 € más barato y 2 h más lento",
        "En avión: 3 h más rápido y 20 € más caro",
    ]
    assert len(app_test.dataframe) == 1


def test_api_client_returns_none_when_the_comparison_fails():
    from unittest.mock import Mock, patch

    from app.ui import api_client

    with patch.object(api_client.requests, "get", return_value=Mock(status_code=404)):
        assert api_client.compare_transports("Madrid", "Atlántida", "cost") is None
    with patch.object(api_client.requests, "get", return_value=Mock(status_code=200, json=lambda: COMPARISON)):
        assert api_client.compare_transports("Madrid", "Barcelona", "cost") == COMPARISON
