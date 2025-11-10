"""
data_loader.py - Carga y gestión de datos desde archivos externos.
Separa datos de lógica para mejor mantenibilidad.
"""

import json
from pathlib import Path
from typing import Dict, List
import logging
from .models import Destination, UserRating, UserProfile

logger = logging.getLogger(__name__)


class DataLoader:
    """Carga datos desde archivos JSON."""
    
    def __init__(self, data_dir: str = "app/data") -> None:
        """
        Inicializa el cargador de datos.
        
        Args:
            data_dir: Directorio donde están los archivos de datos.
        """
        self.data_dir = Path(data_dir)
        self.destinations: Dict[str, Destination] = {}
        self.ratings: List[UserRating] = []
        self.users: Dict[str, UserProfile] = {}
        
        # Cargar datos automáticamente
        self._load_all_data()
    
    def _load_all_data(self) -> None:
        """Carga todos los datos desde archivos."""
        try:
            self._load_destinations()
            self._load_ratings()
            self._load_users()
            logger.info(f"Datos cargados: {len(self.destinations)} destinos, {len(self.ratings)} ratings, {len(self.users)} usuarios")
        except FileNotFoundError as e:
            logger.warning(f"Archivos de datos no encontrados: {e}")
            logger.warning("Usando datos de ejemplo en memoria")
            self._load_sample_data()
    
    def _load_destinations(self) -> None:
        """Carga destinos desde destinations.json."""
        file_path = self.data_dir / "destinations.json"
        
        if not file_path.exists():
            raise FileNotFoundError(f"No se encuentra {file_path}")
        
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        
        for dest_data in data['destinations']:
            destination = Destination(**dest_data)
            self.destinations[destination.id] = destination
    
    def _load_ratings(self) -> None:
        """Carga ratings desde ratings.json."""
        file_path = self.data_dir / "ratings.json"
        
        if not file_path.exists():
            logger.warning(f"No se encuentra {file_path}, usando ratings vacíos")
            return
        
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        
        for rating_data in data['ratings']:
            rating = UserRating(**rating_data)
            self.ratings.append(rating)
    
    def _load_users(self) -> None:
        """Carga usuarios desde users.json."""
        file_path = self.data_dir / "users.json"
        
        if not file_path.exists():
            logger.warning(f"No se encuentra {file_path}, usando usuarios vacíos")
            return
        
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        
        for user_data in data['users']:
            user = UserProfile(**user_data)
            self.users[user.user_id] = user
    
    def _load_sample_data(self) -> None:
        """Carga datos de ejemplo si los archivos no existen."""
        # Datos de ejemplo mínimos (fallback)
        sample_destinations = [
            {
                "id": "madrid",
                "name": "Madrid",
                "country": "España",
                "type": "city",
                "features": {
                    "culture": 0.9, "beach": 0.1, "mountains": 0.2,
                    "nightlife": 0.8, "historical": 0.9, "modern": 0.7,
                    "family_friendly": 0.8, "adventure": 0.3,
                    "relaxation": 0.5, "shopping": 0.8
                },
                "cost_level": 3,
                "best_season": ["spring", "fall"],
                "tags": ["capital", "museums", "tapas"],
                "description": "capital ciudad histórica museo arte cultura"
            }
        ]
        
        for dest_data in sample_destinations:
            destination = Destination(**dest_data)
            self.destinations[destination.id] = destination
    
    def get_destination(self, destination_id: str) -> Destination:
        """Obtiene un destino por ID."""
        return self.destinations.get(destination_id)
    
    def get_all_destinations(self) -> List[Destination]:
        """Obtiene todos los destinos."""
        return list(self.destinations.values())
    
    def get_user_ratings_dict(self) -> Dict[str, Dict[str, float]]:
        """
        Convierte ratings a formato dict para el recommender.
        
        Returns:
            Dict[user_id, Dict[destination_id, rating]]
        """
        user_ratings = {}
        for rating in self.ratings:
            if rating.user_id not in user_ratings:
                user_ratings[rating.user_id] = {}
            user_ratings[rating.user_id][rating.destination_id] = rating.rating
        
        return user_ratings
    
    def get_destination_features_dict(self) -> Dict[str, Dict[str, any]]:
        """
        Convierte destinos a formato dict para el recommender.
        
        Returns:
            Dict[destination_name, Dict[feature, value]]
        """
        dest_features = {}
        for dest_id, dest in self.destinations.items():
            dest_features[dest.name] = {
                "type": dest.type,
                "culture": dest.features.culture,
                "beach": dest.features.beach,
                "mountains": dest.features.mountains,
                "nightlife": dest.features.nightlife,
                "historical": dest.features.historical,
                "modern": dest.features.modern,
                "cost_level": dest.cost_level,
                "description": dest.description
            }
        
        return dest_features
    
    def add_rating(self, rating: UserRating) -> None:
        """Agrega un nuevo rating."""
        self.ratings.append(rating)
    
    def save_ratings(self) -> None:
        """Guarda ratings actualizados a archivo."""
        file_path = self.data_dir / "ratings.json"
        
        ratings_data = {
            'ratings': [
                {
                    'user_id': r.user_id,
                    'destination_id': r.destination_id,
                    'rating': r.rating,
                    'timestamp': r.timestamp.isoformat(),
                    'review': r.review
                }
                for r in self.ratings
            ]
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(ratings_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Ratings guardados en {file_path}")
