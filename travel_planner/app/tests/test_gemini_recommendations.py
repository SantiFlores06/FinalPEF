"""Tests for the Gemini recommendations: single structured call, caching and failures, with the SDK mocked."""

import json
from unittest.mock import MagicMock, patch

import pytest

from app.ai import gemini_recommendations
from app.ai.gemini_recommendations import (
    JSON_MIME_TYPE,
    OUTPUT_TOKENS_PER_CITY,
    TIPS_PER_CITY,
    generate_recommendations_for_cities,
    recommendation_cache,
)


class RateLimitError(Exception):
    code = 429


def response_with_text(text):
    return MagicMock(text=text)


def tips_answer(*cities):
    return response_with_text(json.dumps([{"city": city, "tips": [f"{city} 1", f"{city} 2", f"{city} 3"]} for city in cities]))


def answer_for_requested_cities(**kwargs):
    requested = kwargs["contents"].split("Ciudades: ")[1].split(".\n")[0].split("; ")
    return tips_answer(*requested)


@pytest.fixture(autouse=True)
def empty_cache():
    recommendation_cache.clear()
    yield
    recommendation_cache.clear()


@pytest.fixture
def client():
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = answer_for_requested_cities
    with patch.object(gemini_recommendations, "get_client", return_value=fake_client), \
            patch.object(gemini_recommendations.time, "sleep"):
        yield fake_client


def requested_prompts(client):
    return [call.kwargs["contents"] for call in client.models.generate_content.call_args_list]


def test_single_call_returns_tips_for_every_city(client):
    recommendations = generate_recommendations_for_cities(["París", "Roma", "París", "Atenas"])

    assert list(recommendations) == ["París", "Roma", "Atenas"]
    assert recommendations["Atenas"] == ["Atenas 1", "Atenas 2", "Atenas 3"]
    assert client.models.generate_content.call_count == 1


def test_asks_for_structured_json_with_a_budget_per_city(client):
    generate_recommendations_for_cities(["París", "Roma"])

    config = client.models.generate_content.call_args.kwargs["config"]
    assert config.response_mime_type == JSON_MIME_TYPE
    assert config.response_schema is not None
    assert config.max_output_tokens == 2 * OUTPUT_TOKENS_PER_CITY


def test_prompt_includes_the_interest(client):
    generate_recommendations_for_cities(["Roma"], interest="Gastronomía")

    assert "platos típicos" in requested_prompts(client)[0]


def test_cache_is_kept_per_interest(client):
    generate_recommendations_for_cities(["Roma"], interest="Cultura")
    generate_recommendations_for_cities(["Roma"], interest="Cultura")
    generate_recommendations_for_cities(["Roma"], interest="Compras")

    assert client.models.generate_content.call_count == 2


def test_partial_cache_hit_requests_only_missing_cities(client):
    generate_recommendations_for_cities(["París", "Roma"])

    recommendations = generate_recommendations_for_cities(["París", "Roma", "Berlín"])

    assert all(recommendations.values())
    last_prompt = requested_prompts(client)[-1]
    assert "Berlín" in last_prompt
    assert "París" not in last_prompt


def test_malformed_json_leaves_cities_without_tips_and_uncached(client):
    client.models.generate_content.side_effect = [response_with_text("not json"), tips_answer("Madrid")]

    assert generate_recommendations_for_cities(["Madrid"]) == {"Madrid": None}
    assert generate_recommendations_for_cities(["Madrid"]) == {"Madrid": ["Madrid 1", "Madrid 2", "Madrid 3"]}


def test_cities_missing_from_the_answer_stay_none(client):
    client.models.generate_content.side_effect = None
    client.models.generate_content.return_value = tips_answer("París")

    assert generate_recommendations_for_cities(["París", "Roma"]) == {
        "París": ["París 1", "París 2", "París 3"],
        "Roma": None,
    }


def test_tips_are_trimmed_to_the_expected_amount(client):
    client.models.generate_content.side_effect = None
    client.models.generate_content.return_value = response_with_text(
        json.dumps([{"city": "roma ", "tips": [" a ", "", "b", "c", "d"]}])
    )

    tips = generate_recommendations_for_cities(["Roma"])["Roma"]

    assert tips == ["a", "b", "c"]
    assert len(tips) == TIPS_PER_CITY


def test_rate_limit_is_retried_once(client):
    client.models.generate_content.side_effect = [RateLimitError("quota"), tips_answer("Madrid")]

    assert generate_recommendations_for_cities(["Madrid"])["Madrid"]
    assert client.models.generate_content.call_count == 2


def test_persistent_failure_returns_none_without_caching(client):
    client.models.generate_content.side_effect = RateLimitError("quota")

    assert generate_recommendations_for_cities(["Madrid"]) == {"Madrid": None}
    assert recommendation_cache.get("Madrid", "Variado") is None


def test_non_retryable_error_is_not_retried(client):
    client.models.generate_content.side_effect = ValueError("bad request")

    assert generate_recommendations_for_cities(["Madrid"]) == {"Madrid": None}
    assert client.models.generate_content.call_count == 1


def test_returns_none_without_client():
    with patch.object(gemini_recommendations, "get_client", return_value=None):
        assert generate_recommendations_for_cities(["París"]) == {"París": None}


def test_no_cities_returns_empty_dict():
    assert generate_recommendations_for_cities([]) == {}
