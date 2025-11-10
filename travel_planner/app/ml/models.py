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
    culture: float = Field(ge=0.0, le=1.0)
    beach: float = Field(ge=0.0, le=1.0)
    mountains: float = Field(ge=0.0, le=1.0)
    nightlife: float = Field(ge=0.0, le=1.0)
    historical: float = Field(ge=0.0, le=1.0)
    modern: float = Field(ge=0.0, le=1.0)
    family_friendly: float = Field(ge=0.0, le=1.0, default=0.5)
    adventure: float = Field(ge=0.0, le=1.0, default=0.5)
    relaxation: float = Field(ge=0.0, le=1.0, default=0.5)
    shopping: float = Field(ge=0.0, le=1.0, default=0.5)


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
    
    def to_feature_vector(self) -> List[float]:
        """Convierte características a vector numpy-compatible."""
        return [
            self.features.culture,
            self.features.beach,
            self.features.mountains,
            self.features.nightlife,
            self.features.historical,
            self.features.modern,
            self.features.family_friendly,
            self.features.adventure,
            self.features.relaxation,
            self.features.shopping
        ]


class UserRating(BaseModel):
    """Rating de un usuario para un destino."""
    user_id: str
    destination_id: str
    rating: float = Field(ge=1.0, le=5.0)
    timestamp: datetime
    review: Optional[str] = None
    
    @field_validator('timestamp', mode='before')
    def parse_timestamp(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v)
        return v


class UserProfile(BaseModel):
    """Perfil de preferencias de un usuario."""
    user_id: str
    name: Optional[str] = None
    preferences: DestinationFeatures
    budget_level: int = Field(ge=1, le=5, default=3)
    preferred_seasons: List[SeasonType] = []
    
    def to_preference_vector(self) -> List[float]:
        """Convierte preferencias a vector."""
        return [
            self.preferences.culture,
            self.preferences.beach,
            self.preferences.mountains,
            self.preferences.nightlife,
            self.preferences.historical,
            self.preferences.modern,
            self.preferences.family_friendly,
            self.preferences.adventure,
            self.preferences.relaxation,
            self.preferences.shopping
        ]
