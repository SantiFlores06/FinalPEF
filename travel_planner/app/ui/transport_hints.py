"""Hints comparing the chosen transport with the other ones between two cities."""

from typing import Any, Dict, List, Optional

from app.ui.formatting import format_transport

Option = Dict[str, Any]
SAME_HOURS_TOLERANCE = 0.05
CHEAPEST_LABEL = "Más barato"
FASTEST_LABEL = "Más rápido"


def format_hours(hours: float) -> str:
    """Return hours with one decimal, dropping a trailing '.0'."""
    return f"{hours:.1f}".removesuffix(".0")


def cost_phrase(cost_difference: float) -> str:
    """Describe a cost difference against the chosen transport."""
    if cost_difference < 0:
        return f"{abs(cost_difference):.0f} € más barato"
    if cost_difference > 0:
        return f"{cost_difference:.0f} € más caro"
    return "mismo precio"


def hours_phrase(hours_difference: float) -> str:
    """Describe a travel time difference against the chosen transport."""
    if hours_difference < -SAME_HOURS_TOLERANCE:
        return f"{format_hours(abs(hours_difference))} h más rápido"
    if hours_difference > SAME_HOURS_TOLERANCE:
        return f"{format_hours(hours_difference)} h más lento"
    return "mismo tiempo"


def describe_against(option: Option, reference: Option) -> str:
    """Return a hint like 'En tren: 45 € más barato y 2 h más lento', leading with the advantage."""
    cost_difference = option["total_cost"] - reference["total_cost"]
    hours_difference = option["total_hours"] - reference["total_hours"]
    phrases = [cost_phrase(cost_difference), hours_phrase(hours_difference)]
    if cost_difference >= 0 and hours_difference < 0:
        phrases.reverse()
    return f"En {format_transport(option['transport']).lower()}: {' y '.join(phrases)}"


def is_better_somehow(option: Option, reference: Option) -> bool:
    """Return whether an option is cheaper or faster than the reference."""
    return (
        option["total_cost"] < reference["total_cost"]
        or option["total_hours"] < reference["total_hours"] - SAME_HOURS_TOLERANCE
    )


def find_option(options: List[Option], transport: str) -> Optional[Option]:
    """Return the option of a transport, or None when it cannot link the cities."""
    return next((option for option in options if option["transport"] == transport), None)


def transport_hints(comparison: Dict[str, Any], selected_transport: str) -> List[str]:
    """Return one hint per transport that is cheaper or faster than the selected one."""
    reference = find_option(comparison["options"], selected_transport)
    if reference is None:
        return []
    return [
        describe_against(option, reference)
        for option in comparison["options"]
        if option is not reference and is_better_somehow(option, reference)
    ]


def highlight_labels(comparison: Dict[str, Any], transport: str) -> str:
    """Return whether a transport is the cheapest and/or the fastest option."""
    labels = []
    if transport == comparison["cheapest"]:
        labels.append(CHEAPEST_LABEL)
    if transport == comparison["fastest"]:
        labels.append(FASTEST_LABEL)
    return " · ".join(labels)


def comparison_rows(comparison: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return one table row per transport with its total cost, tolls (None without them), hours, legs and highlights."""
    return [
        {
            "Transporte": format_transport(option["transport"]),
            "Costo (€)": option["total_cost"],
            "Peajes (€)": option.get("toll_cost"),
            "Tiempo (h)": option["total_hours"],
            "Tramos": len(option["path"]) - 1,
            "Destacado": highlight_labels(comparison, option["transport"]),
        }
        for option in comparison["options"]
    ]
