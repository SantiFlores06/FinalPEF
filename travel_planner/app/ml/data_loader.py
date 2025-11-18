"""
data_loader.py - Carga y gestión de datos desde archivos externos.
Separa datos de lógica para mejor mantenibilidad.
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from .models import Destination, UserRating, UserProfile
from collections import defaultdict

logger = logging.getLogger(__name__)

class DataLoader:
    """Carga datos desde archivos JSON."""
    
    def __init__(self, data_dir: str = "app/data") -> None:
        self.data_dir = Path(data_dir)
        self.destinations: Dict[str, Destination] = {}
        self.ratings: List[UserRating] = []
        self.users: Dict[str, UserProfile] = {}
        
        self._load_all_data()

    def _load_all_data(self) -> None:
        """Carga todos los datos desde archivos."""
        try:
            self._load_destinations()
            self._load_ratings()
            self._load_users()
            logger.info(f"Datos cargados: {len(self.destinations)} destinos, {len(self.ratings)} ratings, {len(self.users)} usuarios")
        except FileNotFoundError as e:
            logger.warning(f"Archivos de datos no encontrados: {e}. El sistema de IA usará datos limitados.")
            pass # Permite que el sistema funcione sin datos si es necesario

    def _load_destinations(self) -> None:
        """Carga destinos desde destinations.json."""
        file_path = self.data_dir / "destinations.json"
        if not file_path.exists():
            raise FileNotFoundError(f"No se encuentra {file_path}")
        
        # CORREGIDO: Usar 'utf-8-sig' para manejar el error BOM (Byte Order Mark)
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        
        for dest_data in data.get('destinations', []):
            try:
                destination = Destination(**dest_data)
                self.destinations[destination.id] = destination
            except Exception as e:
                logger.error(f"Error cargando destino: {dest_data.get('id')}. Error: {e}")

    def _load_ratings(self) -> None:
        """Carga ratings desde ratings.json."""
        file_path = self.data_dir / "ratings.json"
        if not file_path.exists():
             logger.warning(f"No se encuentra {file_path}, usando ratings vacíos")
             return
        
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        
        for rating_data in data.get('ratings', []):
            try:
                rating = UserRating(**rating_data)
                self.ratings.append(rating)
            except Exception as e:
                logger.error(f"Error cargando rating: {rating_data}. Error: {e}")

    def _load_users(self) -> None:
        """Carga usuarios desde users.json."""
        file_path = self.data_dir / "users.json"
        if not file_path.exists():
             logger.warning(f"No se encuentra {file_path}, usando usuarios vacíos")
             return

        with open(file_path, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)

        for user_data in data.get('users', []):
            try:
                user = UserProfile(**user_data)
                self.users[user.user_id] = user
            except Exception as e:
                logger.error(f"Error cargando usuario: {user_data.get('user_id')}. Error: {e}")

    def get_destination(self, destination_id: str) -> Optional[Destination]:
        """Obtiene un destino por ID."""
        return self.destinations.get(destination_id)

    def get_all_destinations(self) -> List[Destination]:
        """Retorna lista de todos los objetos de destino."""
        return list(self.destinations.values())

    def get_user_ratings_dict(self) -> Dict[str, Dict[str, float]]:
        """Convierte la lista de ratings a un dict anidado."""
        user_ratings: Dict[str, Dict[str, float]] = defaultdict(dict)
        for rating in self.ratings:
            user_ratings[rating.user_id][rating.destination_id] = rating.rating
        return dict(user_ratings)

    def get_destination_features_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convierte los objetos de destino a un dict simple para ML."""
        dest_features: Dict[str, Dict[str, Any]] = {}
        for dest_id, dest in self.destinations.items():
            # Usar .model_dump() para convertir el submodelo Pydantic en un dict
            features = dest.features.model_dump() 
            features['cost_level'] = dest.cost_level
            features['description'] = dest.description
            dest_features[dest_id] = features
        return dest_features
    
    def save_users(self) -> None:
        """Guarda los perfiles de usuario actualizados en users.json."""
        file_path = self.data_dir / "users.json"
        
        # Convertir objetos UserProfile a diccionarios
        users_data = [user.model_dump() for user in self.users.values()]
        
        try:
            with open(file_path, 'w', encoding='utf-8-sig') as f:
                json.dump({"users": users_data}, f, indent=2, ensure_ascii=False)
            logger.info(f"Perfiles de usuario guardados correctamente en {file_path}")
        except Exception as e:
            logger.error(f"Error guardando usuarios: {e}")