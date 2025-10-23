"""
itinerary_validator.py - Validaciones y reglas de negocio para itinerarios.
Valida restricciones de tiempo, presupuesto, conexiones y lógica del sistema.
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum


class TransportType(Enum):
    """Tipos de transporte disponibles."""
    AVION = "avión"
    TREN = "tren"
    BUS = "bus"
    AUTO = "auto"
    BARCO = "barco"


class ValidationError(Exception):
    """Excepción personalizada para errores de validación."""
    pass


@dataclass
class TimeWindow:
    """Ventana de tiempo para restricciones."""
    start: datetime
    end: datetime
    
    def duration_hours(self) -> float:
        """Calcula la duración en horas."""
        return (self.end - self.start).total_seconds() / 3600
    
    def overlaps_with(self, other: 'TimeWindow') -> bool:
        """Verifica si hay solapamiento con otra ventana."""
        return self.start < other.end and other.start < self.end


@dataclass
class RouteSegment:
    """Representa un segmento individual del itinerario."""
    origin: str
    destination: str
    transport_type: TransportType
    cost: float
    duration_hours: float
    departure_time: Optional[datetime] = None
    arrival_time: Optional[datetime] = None
    distance_km: Optional[float] = None
    
    def is_valid_timing(self) -> bool:
        """Valida que los tiempos sean consistentes."""
        if self.departure_time and self.arrival_time:
            actual_duration = (self.arrival_time - self.departure_time).total_seconds() / 3600
            return abs(actual_duration - self.duration_hours) < 0.1
        return True


@dataclass
class ItineraryConstraints:
    """Restricciones del itinerario."""
    max_budget: float
    max_duration_hours: float
    max_segments: int
    required_cities: List[str]
    forbidden_cities: List[str] = None
    allowed_transports: List[TransportType] = None
    min_layover_hours: float = 1.0
    max_layover_hours: float = 24.0
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    def __post_init__(self) -> None:
        """Inicializa valores por defecto."""
        if self.forbidden_cities is None:
            self.forbidden_cities = []
        if self.allowed_transports is None:
            self.allowed_transports = list(TransportType)


class ItineraryValidator:
    """
    Valida itinerarios completos según reglas de negocio.
    
    Verifica restricciones de presupuesto, tiempo, conexiones lógicas,
    y otras reglas específicas del dominio de viajes.
    """
    
    def __init__(self, constraints: ItineraryConstraints) -> None:
        """
        Inicializa el validador con restricciones.
        
        Args:
            constraints: Conjunto de restricciones a aplicar.
        """
        self.constraints = constraints
        self.validation_errors: List[str] = []
    
    def validate_full_itinerary(
        self, 
        segments: List[RouteSegment]
    ) -> Tuple[bool, List[str]]:
        """
        Valida un itinerario completo.
        
        Args:
            segments: Lista de segmentos del itinerario en orden.
        
        Returns:
            Tupla (es_válido, lista_de_errores).
        """
        self.validation_errors.clear()
        
        # Ejecutar todas las validaciones
        self._validate_budget(segments)
        self._validate_duration(segments)
        self._validate_segment_count(segments)
        self._validate_continuity(segments)
        self._validate_required_cities(segments)
        self._validate_forbidden_cities(segments)
        self._validate_transport_types(segments)
        self._validate_layovers(segments)
        self._validate_timing_consistency(segments)
        self._validate_date_range(segments)
        
        return len(self.validation_errors) == 0, self.validation_errors
    
    def _validate_budget(self, segments: List[RouteSegment]) -> None:
        """Valida que el costo total no exceda el presupuesto."""
        total_cost = sum(seg.cost for seg in segments)
        if total_cost > self.constraints.max_budget:
            self.validation_errors.append(
                f"Presupuesto excedido: {total_cost:.2f} > {self.constraints.max_budget:.2f}"
            )
    
    def _validate_duration(self, segments: List[RouteSegment]) -> None:
        """Valida que la duración total no exceda el máximo."""
        total_hours = sum(seg.duration_hours for seg in segments)
        if total_hours > self.constraints.max_duration_hours:
            self.validation_errors.append(
                f"Duración excedida: {total_hours:.1f}h > {self.constraints.max_duration_hours:.1f}h"
            )
    
    def _validate_segment_count(self, segments: List[RouteSegment]) -> None:
        """Valida que no haya demasiados segmentos."""
        if len(segments) > self.constraints.max_segments:
            self.validation_errors.append(
                f"Demasiados segmentos: {len(segments)} > {self.constraints.max_segments}"
            )
    
    def _validate_continuity(self, segments: List[RouteSegment]) -> None:
        """Valida que el destino de un segmento sea el origen del siguiente."""
        for i in range(len(segments) - 1):
            if segments[i].destination != segments[i + 1].origin:
                self.validation_errors.append(
                    f"Discontinuidad en segmento {i}: "
                    f"{segments[i].destination} != {segments[i + 1].origin}"
                )
    
    def _validate_required_cities(self, segments: List[RouteSegment]) -> None:
        """Valida que se visiten todas las ciudades requeridas."""
        visited_cities = set()
        for seg in segments:
            visited_cities.add(seg.origin)
            visited_cities.add(seg.destination)
        
        missing_cities = set(self.constraints.required_cities) - visited_cities
        if missing_cities:
            self.validation_errors.append(
                f"Ciudades requeridas no visitadas: {', '.join(missing_cities)}"
            )
    
    def _validate_forbidden_cities(self, segments: List[RouteSegment]) -> None:
        """Valida que no se visiten ciudades prohibidas."""
        visited_cities = set()
        for seg in segments:
            visited_cities.add(seg.origin)
            visited_cities.add(seg.destination)
        
        forbidden_visited = visited_cities & set(self.constraints.forbidden_cities)
        if forbidden_visited:
            self.validation_errors.append(
                f"Ciudades prohibidas visitadas: {', '.join(forbidden_visited)}"
            )
    
    def _validate_transport_types(self, segments: List[RouteSegment]) -> None:
        """Valida que solo se usen transportes permitidos."""
        for i, seg in enumerate(segments):
            if seg.transport_type not in self.constraints.allowed_transports:
                self.validation_errors.append(
                    f"Transporte no permitido en segmento {i}: {seg.transport_type.value}"
                )
    
    def _validate_layovers(self, segments: List[RouteSegment]) -> None:
        """Valida que los tiempos de escala sean razonables."""
        if not all(seg.arrival_time and seg.departure_time for seg in segments):
            return  # No se puede validar sin tiempos
        
        for i in range(len(segments) - 1):
            arrival = segments[i].arrival_time
            next_departure = segments[i + 1].departure_time
            
            if arrival and next_departure:
                layover_hours = (next_departure - arrival).total_seconds() / 3600
                
                if layover_hours < self.constraints.min_layover_hours:
                    self.validation_errors.append(
                        f"Escala muy corta en {segments[i].destination}: {layover_hours:.1f}h"
                    )
                elif layover_hours > self.constraints.max_layover_hours:
                    self.validation_errors.append(
                        f"Escala muy larga en {segments[i].destination}: {layover_hours:.1f}h"
                    )
    
    def _validate_timing_consistency(self, segments: List[RouteSegment]) -> None:
        """Valida consistencia de tiempos en cada segmento."""
        for i, seg in enumerate(segments):
            if not seg.is_valid_timing():
                self.validation_errors.append(
                    f"Inconsistencia de tiempos en segmento {i}: "
                    f"{seg.origin} -> {seg.destination}"
                )
    
    def _validate_date_range(self, segments: List[RouteSegment]) -> None:
        """Valida que el viaje esté dentro del rango de fechas permitido."""
        if not self.constraints.start_date or not self.constraints.end_date:
            return
        
        if not segments or not segments[0].departure_time or not segments[-1].arrival_time:
            return
        
        trip_start = segments[0].departure_time
        trip_end = segments[-1].arrival_time
        
        if trip_start < self.constraints.start_date:
            self.validation_errors.append(
                f"Viaje inicia antes de lo permitido: {trip_start} < {self.constraints.start_date}"
            )
        
        if trip_end > self.constraints.end_date:
            self.validation_errors.append(
                f"Viaje termina después de lo permitido: {trip_end} > {self.constraints.end_date}"
            )
    
    def get_itinerary_summary(self, segments: List[RouteSegment]) -> Dict:
        """
        Genera un resumen del itinerario.
        
        Args:
            segments: Lista de segmentos.
        
        Returns:
            Diccionario con estadísticas del itinerario.
        """
        if not segments:
            return {}
        
        visited_cities = set()
        for seg in segments:
            visited_cities.add(seg.origin)
            visited_cities.add(seg.destination)
        
        transport_counts: Dict[str, int] = {}
        for seg in segments:
            transport = seg.transport_type.value
            transport_counts[transport] = transport_counts.get(transport, 0) + 1
        
        total_cost = sum(seg.cost for seg in segments)
        total_duration = sum(seg.duration_hours for seg in segments)
        total_distance = sum(seg.distance_km for seg in segments if seg.distance_km)
        
        return {
            'total_segments': len(segments),
            'cities_visited': len(visited_cities),
            'city_list': sorted(visited_cities),
            'total_cost': total_cost,
            'total_duration_hours': total_duration,
            'total_distance_km': total_distance,
            'transport_breakdown': transport_counts,
            'origin': segments[0].origin,
            'final_destination': segments[-1].destination
        }


# Ejemplo de uso
if __name__ == "__main__":
    # Crear restricciones
    constraints = ItineraryConstraints(
        max_budget=500,
        max_duration_hours=24,
        max_segments=5,
        required_cities=["Madrid", "Barcelona", "París"],
        forbidden_cities=["Londres"],
        min_layover_hours=2.0,
        max_layover_hours=12.0
    )
    
    # Crear segmentos de ejemplo
    segments = [
        RouteSegment(
            origin="Madrid",
            destination="Barcelona",
            transport_type=TransportType.TREN,
            cost=50,
            duration_hours=3,
            departure_time=datetime(2025, 11, 1, 8, 0),
            arrival_time=datetime(2025, 11, 1, 11, 0),
            distance_km=620
        ),
        RouteSegment(
            origin="Barcelona",
            destination="París",
            transport_type=TransportType.AVION,
            cost=100,
            duration_hours=2,
            departure_time=datetime(2025, 11, 1, 14, 0),
            arrival_time=datetime(2025, 11, 1, 16, 0),
            distance_km=1030
        )
    ]
    
    # Validar
    validator = ItineraryValidator(constraints)
    is_valid, errors = validator.validate_full_itinerary(segments)
    
    print("=" * 60)
    print("Validación de Itinerario")
    print("=" * 60)
    
    if is_valid:
        print("✓ Itinerario VÁLIDO")
    else:
        print("✗ Itinerario INVÁLIDO")
        print("\nErrores encontrados:")
        for error in errors:
            print(f"  - {error}")
    
    # Mostrar resumen
    summary = validator.get_itinerary_summary(segments)
    print("\nResumen del Itinerario:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
