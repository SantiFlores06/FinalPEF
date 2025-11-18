"""
test_validator.py - Tests para las reglas de validación de itinerarios.
"""
import pytest
from datetime import datetime
from app.core.itinerary_validator import (
    ItineraryValidator, RouteSegment, TransportType, ItineraryConstraints
)

def test_validate_budget_success(basic_constraints):
    """Un itinerario dentro del presupuesto debe ser válido."""
    #  Usamos Madrid y Barcelona porque 'basic_constraints' las exige.
    validator = ItineraryValidator(basic_constraints)
    
    segments = [
        RouteSegment("Madrid", "Valencia", TransportType.TREN, cost=100, duration_hours=2),
        RouteSegment("Valencia", "Barcelona", TransportType.TREN, cost=200, duration_hours=2)
    ]
    # Total 300 < 1000. Visitamos Madrid y Barcelona -> VÁLIDO.
    is_valid, errors = validator.validate_full_itinerary(segments)
    assert is_valid is True
    assert len(errors) == 0

def test_validate_budget_exceeded(basic_constraints):
    """Un itinerario muy caro debe fallar."""
    # CORRECCIÓN: Quitamos el requisito de ciudades para probar SOLO el presupuesto
    basic_constraints.required_cities = []
    validator = ItineraryValidator(basic_constraints)
    
    segments = [
        RouteSegment("A", "B", TransportType.AVION, cost=1500, duration_hours=1)
    ]
    # Total 1500 > 1000 -> FALLA
    is_valid, errors = validator.validate_full_itinerary(segments)
    assert is_valid is False
    # Buscamos el mensaje específico en cualquier parte de la lista de errores
    assert any("Presupuesto excedido" in e for e in errors)

def test_validate_continuity_error(basic_constraints):
    """Debe detectar si hay un 'teletransporte' (discontinuidad)."""
    # CORRECCIÓN: Quitamos requisito de ciudades
    basic_constraints.required_cities = [] 
    validator = ItineraryValidator(basic_constraints)
    
    segments = [
        RouteSegment("Madrid", "Barcelona", TransportType.TREN, 100, 2),
        # El siguiente segmento sale de Valencia, pero llegamos a Barcelona!
        RouteSegment("Valencia", "Sevilla", TransportType.BUS, 50, 4)
    ]
    
    is_valid, errors = validator.validate_full_itinerary(segments)
    assert is_valid is False
    assert any("Discontinuidad" in e for e in errors)

def test_forbidden_city(basic_constraints):
    """No se debe permitir visitar ciudades prohibidas."""
    # CORRECCIÓN: Quitamos requisito de ciudades para aislar este test
    basic_constraints.required_cities = []
    # Prohibir "París"
    basic_constraints.forbidden_cities = ["París"]
    validator = ItineraryValidator(basic_constraints)
    
    segments = [
        RouteSegment("Madrid", "París", TransportType.AVION, 100, 2)
    ]
    
    is_valid, errors = validator.validate_full_itinerary(segments)
    assert is_valid is False
    assert any("Ciudades prohibidas visitadas" in e for e in errors)