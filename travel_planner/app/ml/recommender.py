"""
recommender.py - Sistema de recomendaciones con Machine Learning.
Implementa collaborative filtering, content-based y text-based filtering.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Tuple, Any, Optional
import logging
from .data_loader import DataLoader
from .models import DestinationFeatures # Importar el modelo Pydantic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TravelRecommender:
    """
    Sistema de recomendaciones para itinerarios de viaje.
    Combina tres enfoques:
    1. Collaborative Filtering: Basado en usuarios similares
    2. Content-Based Filtering: Basado en características de destinos
    3. Text-Based Filtering: Basado en búsquedas textuales (TF-IDF)
    """

    def __init__(self, data_loader: Optional[DataLoader] = None) -> None:
        """
        Inicializa el sistema de recomendaciones.
        """
        # Cargar datos
        self.data_loader = data_loader if data_loader else DataLoader()
        
        # Convertir datos a formato interno
        self.user_ratings = self.data_loader.get_user_ratings_dict()
        self.destination_features = self.data_loader.get_destination_features_dict()
        self.destinations_list = list(self.destination_features.keys())
        
        # Modelos
        self.knn_model: Optional[NearestNeighbors] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        
        # Usar las llaves del modelo Pydantic para asegurar el orden
        self.feature_keys = list(DestinationFeatures.model_fields.keys())
        
        # Pre-entrenar modelo TF-IDF
        self._train_tfidf()
        
        logger.info(f"Recommender inicializado: {len(self.destination_features)} destinos, {len(self.user_ratings)} usuarios")

    def get_collaborative_recommendations(
        self,
        user_id: str,
        n_recommendations: int = 5
    ) -> List[Tuple[str, float]]:
        """Genera recomendaciones usando Collaborative Filtering."""
        
        users = list(self.user_ratings.keys())
        destinations = self.destinations_list
        
        # Evitar error si no hay usuarios o destinos
        if not users or not destinations:
            return []

        rating_matrix = np.zeros((len(users), len(destinations)))
        for i, user in enumerate(users):
            for j, dest in enumerate(destinations):
                rating_matrix[i, j] = self.user_ratings[user].get(dest, 0)
        
        # Evitar error si la matriz está vacía
        if rating_matrix.shape[0] == 0:
            return []
            
        user_similarity = cosine_similarity(rating_matrix)

        if user_id not in users:
            logger.warning(f"Usuario {user_id} no encontrado, usando promedio global")
            avg_ratings = np.mean(rating_matrix, axis=0)
            recommendations = [
                (destinations[i], avg_ratings[i])
                for i in range(len(destinations))
                if avg_ratings[i] > 0
            ]
        else:
            user_idx = users.index(user_id)
            visited_destinations = set(self.user_ratings[user_id].keys())
            unvisited_indices = [
                i for i, dest in enumerate(destinations)
                if dest not in visited_destinations
            ]
            
            predictions = []
            for dest_idx in unvisited_indices:
                weighted_sum = 0
                similarity_sum = 0
                for other_user_idx in range(len(users)):
                    if other_user_idx != user_idx:
                        similarity = user_similarity[user_idx][other_user_idx]
                        rating = rating_matrix[other_user_idx][dest_idx]
                        if rating > 0:
                            weighted_sum += similarity * rating
                            similarity_sum += abs(similarity)
                
                if similarity_sum > 0:
                    predicted_rating = weighted_sum / similarity_sum
                    predictions.append((destinations[dest_idx], predicted_rating))
            
            recommendations = predictions
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]

    def get_content_based_recommendations(
        self,
        user_preferences: Dict[str, float],
        n_recommendations: int = 5
    ) -> List[Tuple[str, float]]:
        """Genera recomendaciones usando Content-Based Filtering."""
        
        user_vector = np.array([user_preferences.get(key, 0.5) for key in self.feature_keys])
        
        dest_matrix = np.array([
            [self.destination_features[dest].get(key, 0) for key in self.feature_keys]
            for dest in self.destinations_list
        ])
        
        # Evitar error si la matriz de destinos está vacía
        if dest_matrix.shape[0] == 0:
            return []
            
        user_vector_2d = user_vector.reshape(1, -1)
        similarities = cosine_similarity(user_vector_2d, dest_matrix)[0]
        
        recommendations = [
            (self.destinations_list[i], similarities[i])
            for i in range(len(self.destinations_list))
        ]
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]

    def _train_tfidf(self):
        """Entrena el modelo TF-IDF con las descripciones."""
        # CORRECCIÓN: Lista de stop words en español [cite: 4206-4209]
        spanish_stop_words = [
            'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'ser', 'se', 'no',
            'haber', 'por', 'con', 'su', 'para', 'como', 'estar', 'tener'
        ]
        
        descriptions = [
            self.destination_features[dest].get("description", "")
            for dest in self.destinations_list
        ]
        
        if not descriptions:
            logger.warning("No hay descripciones para entrenar TF-IDF.")
            return

        self.tfidf_vectorizer = TfidfVectorizer(stop_words=spanish_stop_words)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(descriptions)

    def get_text_based_recommendations(
        self,
        query: str,
        n_recommendations: int = 5
    ) -> List[Tuple[str, float]]:
        """Genera recomendaciones basadas en texto (TF-IDF)."""
        if not self.tfidf_vectorizer:
            logger.warning("Modelo TF-IDF no entrenado.")
            return []
            
        query_vector = self.tfidf_vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        recommendations = [
            (self.destinations_list[i], similarities[i])
            for i in range(len(self.destinations_list))
            if similarities[i] > 0
        ]
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]

    def get_hybrid_recommendations(
        self,
        user_id: str,
        user_preferences: Dict[str, float] = None,
        query: str = None,
        n_recommendations: int = 5,
        weights: Tuple[float, float, float] = (0.4, 0.4, 0.2)
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """Sistema híbrido que combina múltiples enfoques."""
        
        collaborative_weight, content_weight, text_weight = weights
        all_destinations = self.destinations_list
        
        # 1. Collaborative Filtering
        collab_recs = self.get_collaborative_recommendations(user_id, n_recommendations * 2)
        collab_scores = {dest: score for dest, score in collab_recs}
        
        # 2. Content-Based Filtering
        content_scores = {}
        if user_preferences:
            content_recs = self.get_content_based_recommendations(user_preferences, n_recommendations * 2)
            content_scores = {dest: score for dest, score in content_recs}
            
        # 3. Text-Based Filtering
        text_scores = {}
        if query:
            text_recs = self.get_text_based_recommendations(query, n_recommendations * 2)
            text_scores = {dest: score for dest, score in text_recs}
            
        # 4. Combinar scores
        hybrid_scores = []
        for dest in all_destinations:
            # Normalizar score colaborativo (ratings 1-5)
            collab_score = collab_scores.get(dest, 0) / 5.0 
            content_score = content_scores.get(dest, 0)
            text_score = text_scores.get(dest, 0)
            
            final_score = (
                collaborative_weight * collab_score +
                content_weight * content_score +
                text_weight * text_score
            )
            
            detail = {
                'collaborative': collab_score,
                'content': content_score,
                'text': text_score
            }
            hybrid_scores.append((dest, final_score, detail))
            
        hybrid_scores.sort(key=lambda x: x[1], reverse=True)
        return hybrid_scores[:n_recommendations]

    def train_knn_model(self, n_neighbors: int = 5) -> None:
        """Entrena modelo KNN para recomendaciones rápidas."""
        
        feature_matrix = np.array([
            [self.destination_features[dest].get(key, 0) for key in self.feature_keys]
            for dest in self.destinations_list
        ])
        
        if feature_matrix.shape[0] == 0:
            logger.warning("No hay destinos para entrenar modelo KNN.")
            return

        # Ajustar n_neighbors si hay menos destinos que vecinos 
        max_neighbors = min(n_neighbors, len(self.destinations_list))
        
        if max_neighbors == 0:
             logger.warning("No hay suficientes vecinos para entrenar KNN.")
             return
             
        self.knn_model = NearestNeighbors(n_neighbors=max_neighbors, metric='cosine')
        self.knn_model.fit(feature_matrix)
        logger.info(f"Modelo KNN entrenado con {len(self.destinations_list)} destinos (max neighbors: {max_neighbors})")

    def get_similar_destinations(
        self,
        destination: str, # <-- CORREGIDO: (era destination_id)
        n_similar: int = 3
    ) -> List[Tuple[str, float]]:
        """Encuentra destinos similares usando KNN."""
        
        if self.knn_model is None:
            logger.warning("Modelo KNN no entrenado. Entrenando ahora...")
            self.train_knn_model()
            if self.knn_model is None: # Si sigue sin entrenarse
                return []
            
        if destination not in self.destinations_list:
            logger.warning(f"Destino {destination} no encontrado en get_similar_destinations")
            return []
            
        dest_idx = self.destinations_list.index(destination)
        
        dest_vector = np.array([
            [self.destination_features[destination].get(key, 0) for key in self.feature_keys]
        ])
        
        # Asegurar que n_neighbors no sea mayor que las muestras
        n_neighbors_query = min(n_similar + 1, len(self.destinations_list)) # +1 para incluirse a sí mismo
        
        distances, indices = self.knn_model.kneighbors(dest_vector, n_neighbors=n_neighbors_query)
        
        similar = [
            (self.destinations_list[idx], 1 - distances[0][i]) # Convertir distancia a similaridad
            for i, idx in enumerate(indices[0])
            if idx != dest_idx # Excluir el destino original
        ]
        
        return similar[:n_similar]

    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas del sistema."""
        return {
            'total_users': len(self.user_ratings),
            'total_destinations': len(self.destination_features),
            'total_ratings': sum(len(ratings) for ratings in self.user_ratings.values()),
            'avg_ratings_per_user': np.mean([len(ratings) for ratings in self.user_ratings.values()]) if self.user_ratings else 0,
            'model_trained': self.knn_model is not None
        }
    
    def learn_from_reservation(self, user_id: str, destination_ids: List[str], learning_rate: float = 0.1) -> None:
        """
        Actualiza las preferencias del usuario basándose en una reserva realizada.
        
        Args:
            user_id: El usuario que hizo la reserva.
            destination_ids: Lista de destinos reservados (IDs).
            learning_rate: Qué tanto influye la nueva experiencia (0.0 a 1.0).
                           0.1 significa que la reserva cambia las preferencias un 10%.
        """
        if user_id not in self.data_loader.users:
            logger.warning(f"Usuario {user_id} no encontrado para aprendizaje. Omitiendo.")
            return

        user_profile = self.data_loader.users[user_id]
        
        # Recorremos los destinos reservados para aprender de ellos
        for dest_id in destination_ids:
            dest_id_lower = dest_id.lower()
            
            # Verificar si tenemos datos de este destino
            if dest_id_lower in self.destination_features:
                dest_features = self.destination_features[dest_id_lower]
                
                # ACTUALIZACIÓN MATEMÁTICA (Promedio Ponderado)
                # Nuevo_Gusto = (Gusto_Actual * 0.9) + (Característica_Destino * 0.1)
                
                for feature, value in dest_features.items():
                    # Solo actualizamos características numéricas que existen en el perfil (cultura, playa, etc.)
                    if hasattr(user_profile.preferences, feature) and isinstance(value, (int, float)):
                        current_pref = getattr(user_profile.preferences, feature)
                        
                        # Fórmula de aprendizaje
                        new_pref = (current_pref * (1.0 - learning_rate)) + (value * learning_rate)
                        
                        # Actualizar el perfil
                        setattr(user_profile.preferences, feature, new_pref)
                
                logger.info(f"🧠 IA: El usuario {user_id} aprendió de su viaje a {dest_id}")

        # Guardar los cambios en el disco
        self.data_loader.save_users()

# Ejemplo de uso
""" 
if __name__ == "__main__":
    
    print("=" * 60)
    print("Sistema de Recomendaciones de Viajes con ML")
    print("=" * 60)
    
    # Cargar datos
    data_loader = DataLoader(data_dir="app/data")
    recommender = TravelRecommender(data_loader)

    # 1. Collaborative Filtering
    print("\n1. Recomendaciones Collaborative Filtering (user_1):")
    collab_recs = recommender.get_collaborative_recommendations("user_1", n_recommendations=3)
    for dest, score in collab_recs:
        print(f"  - {dest}: {score:.3f}")

    # 2. Content-Based Filtering
    print("\n2. Recomendaciones Content-Based:")
    print("   Preferencias: cultura alta, playa media")
    preferences = {"culture": 0.9, "beach": 0.6, "nightlife": 0.7}
    content_recs = recommender.get_content_based_recommendations(preferences, n_recommendations=3)
    for dest, score in content_recs:
        print(f"  - {dest}: {score:.3f}")

    # 3. Text-Based Filtering
    print("\n3. Recomendaciones Text-Based:")
    print("   Query: 'playa y arquitectura moderna'")
    text_recs = recommender.get_text_based_recommendations("playa arquitectura moderna", n_recommendations=3)
    for dest, score in text_recs:
        print(f"  - {dest}: {score:.3f}")

    # 4. Hybrid System
    print("\n4. Recomendaciones Híbridas (user_1):")
    hybrid_recs = recommender.get_hybrid_recommendations(
        user_id="user_1",
        user_preferences=preferences,
        query="playa cultura",
        n_recommendations=3
    )
    for dest, score, detail in hybrid_recs:
        print(f"  - {dest}: {score:.3f}")
        print(f"    (Collab: {detail['collaborative']:.2f}, Content: {detail['content']:.2f}, Text: {detail['text']:.2f})")

    # 5. Similar Destinations (KNN)
    print("\n5. Destinos Similares:")
    dest_list = recommender.destinations_list
    if dest_list:
        sample_dest = dest_list[0] # Usar 'madrid'
        print(f"   Similares a {sample_dest}:")
        similar = recommender.get_similar_destinations(sample_dest, n_similar=2)
        for dest, similarity in similar:
            print(f"  - {dest}: {similarity:.3f}")

    # 6. Estadísticas
    print("\n6. Estadísticas:")
    stats = recommender.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")


"""
