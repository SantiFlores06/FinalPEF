"""
models.py - Modelos de datos para el sistema de recomendaciones.
Usa Pydantic para validación automática.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum

class SeasonType(str, Enum):
    """Estaciones del año."""
    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"
    WINTER = "winter"

class DestinationType(str, Enum):
    """Tipos de destino."""
    CITY = "city"
    BEACH = "beach"
    MOUNTAIN = "mountain"
    COUNTRYSIDE = "countryside"
    ISLAND = "island"

class DestinationFeatures(BaseModel):
    """Características de un destino (0-1)."""
    culture: float = Field(default=0.5, ge=0.0, le=1.0)
    beach: float = Field(default=0.5, ge=0.0, le=1.0)
    mountains: float = Field(default=0.5, ge=0.0, le=1.0)
    nightlife: float = Field(default=0.5, ge=0.0, le=1.0)
    historical: float = Field(default=0.5, ge=0.0, le=1.0)
    modern: float = Field(default=0.5, ge=0.0, le=1.0)
    family_friendly: float = Field(default=0.5, ge=0.0, le=1.0)
    adventure: float = Field(default=0.5, ge=0.0, le=1.0)
    relaxation: float = Field(default=0.5, ge=0.0, le=1.0)
    shopping: float = Field(default=0.5, ge=0.0, le=1.0)

class Destination(BaseModel):
    """Modelo de un destino turístico."""
    id: str
    name: str
    country: str
    type: DestinationType
    features: DestinationFeatures
    cost_level: int = Field(ge=1, le=5)
    best_season: List[SeasonType]
    tags: List[str]
    description: str
    avg_temp_summer: Optional[float] = None
    avg_temp_winter: Optional[float] = None

class UserRating(BaseModel):
    """Rating de un usuario para un destino."""
    user_id: str
    destination_id: str
    rating: float = Field(ge=1.0, le=5.0)
    timestamp: datetime
    review: Optional[str] = None

    @field_validator('timestamp', mode='before')
    @classmethod
    def parse_timestamp(cls, v):
        """Validador para convertir string ISO a objeto datetime."""
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v)
            except ValueError:
                raise ValueError(f"Formato de timestamp inválido: {v}")
        return v

class UserProfile(BaseModel):
    """Perfil de preferencias de un usuario."""
    user_id: str
    name: str
    preferences: DestinationFeatures
    budget_level: int = Field(ge=1, le=5, default=3)
    preferred_seasons: List[SeasonType] = []