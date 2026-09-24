"""Tests for the Gemini recommendations: caching and parallel fetching, with the SDK mocked."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from app.ai import gemini_recommendations
from app.ai.gemini_recommendations import (
    MAX_OUTPUT_TOKENS,
    generate_city_recommendations,
    generate_recommendations_for_cities,
    recommendation_cache,
)


def response_with_text(text):
    return MagicMock(text=text)


@pytest.fixture(autouse=True)
def empty_cache():
    recommendation_cache.clear()
    yield
    recommendation_cache.clear()


@pytest.fixture
def client():
    fake_client = MagicMock()
    fake_client.models.generate_content.return_value = response_with_text("  - Louvre - Museo  ")
    with patch.object(gemini_recommendations, "get_client", return_value=fake_client):
        yield fake_client


def test_returns_the_stripped_text_of_the_model(client):
    assert generate_city_recommendations("Paris") == "- Louvre - Museo"


def test_limits_the_output_tokens(client):
    generate_city_recommendations("Paris")

    config = client.models.generate_content.call_args.kwargs["config"]
    assert config.max_output_tokens == MAX_OUTPUT_TOKENS


def test_cache_hit_avoids_a_second_call(client):
    first = generate_city_recommendations("Paris")
    second = generate_city_recommendations("Paris")

    assert first == second
    assert client.models.generate_content.call_count == 1


def test_failures_are_not_cached(client):
    client.models.generate_content.side_effect = [RuntimeError("quota"), response_with_text("- Prado")]

    assert generate_city_recommendations("Madrid") is None
    assert generate_city_recommendations("Madrid") == "- Prado"
    assert client.models.generate_content.call_count == 2


def test_empty_answers_are_not_cached(client):
    client.models.generate_content.side_effect = [response_with_text(""), response_with_text("- Prado")]

    assert generate_city_recommendations("Madrid") is None
    assert generate_city_recommendations("Madrid") == "- Prado"


def test_returns_none_without_client():
    with patch.object(gemini_recommendations, "get_client", return_value=None):
        assert generate_city_recommendations("Paris") is None


def test_parallel_helper_returns_one_entry_per_unique_city(client):
    recommendations = generate_recommendations_for_cities(["Paris", "Roma", "Paris", "Berlin"])

    assert list(recommendations) == ["Paris", "Roma", "Berlin"]
    assert all(recommendations.values())
    assert client.models.generate_content.call_count == 3


def test_parallel_helper_runs_requests_concurrently(client):
    cities = ["Paris", "Roma", "Berlin"]
    all_started = threading.Barrier(len(cities), timeout=5)

    def wait_for_every_request(**kwargs):
        all_started.wait()
        return response_with_text("- Lugar")

    client.models.generate_content.side_effect = wait_for_every_request

    recommendations = generate_recommendations_for_cities(cities)

    assert recommendations == {city: "- Lugar" for city in cities}


def test_parallel_helper_keeps_failed_cities_as_none(client):
    client.models.generate_content.side_effect = lambda **kwargs: (
        response_with_text(None) if "Roma" in kwargs["contents"] else response_with_text("- Lugar")
    )

    recommendations = generate_recommendations_for_cities(["Paris", "Roma"])

    assert recommendations == {"Paris": "- Lugar", "Roma": None}


def test_parallel_helper_with_no_cities_returns_empty_dict():
    assert generate_recommendations_for_cities([]) == {}
