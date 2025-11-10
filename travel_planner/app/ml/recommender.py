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
        
        Args:
            data_loader: Instancia de DataLoader. Si es None, crea una nueva.
        """
        # Cargar datos
        self.data_loader = data_loader if data_loader else DataLoader()
        
        # Convertir datos a formato interno
        self.user_ratings = self.data_loader.get_user_ratings_dict()
        self.destination_features = self.data_loader.get_destination_features_dict()
        
        # Modelos
        self.knn_model = None
        
        logger.info(f"Recommender inicializado: {len(self.destination_features)} destinos, {len(self.user_ratings)} usuarios")
    
    def add_user_rating(self, user_id: str, destination: str, rating: float) -> None:
        """
        Registra el rating de un usuario para un destino.
        
        Args:
            user_id: ID del usuario.
            destination: Nombre del destino.
            rating: Calificación (1-5).
        """
        if user_id not in self.user_ratings:
            self.user_ratings[user_id] = {}
        
        self.user_ratings[user_id][destination] = rating
        logger.info(f"Rating agregado: {user_id} → {destination}: {rating}")
    
    def get_collaborative_recommendations(
        self,
        user_id: str,
        n_recommendations: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Genera recomendaciones usando Collaborative Filtering.
        
        Args:
            user_id: ID del usuario.
            n_recommendations: Número de recomendaciones.
        
        Returns:
            Lista de tuplas (destino, score_predicho).
        """
        users = list(self.user_ratings.keys())
        destinations = list(self.destination_features.keys())
        
        # Matriz de ratings (usuarios × destinos)
        rating_matrix = np.zeros((len(users), len(destinations)))
        
        for i, user in enumerate(users):
            for j, dest in enumerate(destinations):
                rating_matrix[i, j] = self.user_ratings[user].get(dest, 0)
        
        # Calcular similitud entre usuarios
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
        """
        Genera recomendaciones usando Content-Based Filtering.
        
        Args:
            user_preferences: Dict con preferencias del usuario.
            n_recommendations: Número de recomendaciones.
        
        Returns:
            Lista de tuplas (destino, score_similaridad).
        """
        feature_keys = ["culture", "beach", "mountains", "nightlife", "historical", "modern"]
        user_vector = np.array([user_preferences.get(key, 0.5) for key in feature_keys])
        
        dest_names = list(self.destination_features.keys())
        dest_matrix = np.array([
            [self.destination_features[dest].get(key, 0) for key in feature_keys]
            for dest in dest_names
        ])
        
        user_vector_2d = user_vector.reshape(1, -1)
        similarities = cosine_similarity(user_vector_2d, dest_matrix)[0]
        
        recommendations = [
            (dest_names[i], similarities[i])
            for i in range(len(dest_names))
        ]
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]
    
    def get_text_based_recommendations(
        self,
        query: str,
        n_recommendations: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Genera recomendaciones basadas en texto (TF-IDF).
        
        Args:
            query: Texto descriptivo.
            n_recommendations: Número de recomendaciones.
        
        Returns:
            Lista de tuplas (destino, score_relevancia).
        """
        dest_names = list(self.destination_features.keys())
        descriptions = [
            self.destination_features[dest]["description"]
            for dest in dest_names
        ]
        
        # Stop words en español (lista personalizada básica)
        spanish_stop_words = [
            'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'ser', 'se', 'no', 
            'haber', 'por', 'con', 'su', 'para', 'como', 'estar', 'tener', 
            'le', 'lo', 'todo', 'pero', 'más', 'hacer', 'o', 'poder', 'decir',
            'este', 'ir', 'otro', 'ese', 'si', 'me', 'ya', 'ver', 'porque',
            'dar', 'cuando', 'muy', 'sin', 'vez', 'mucho', 'saber', 'sobre',
            'también', 'hasta', 'año', 'dos', 'querer', 'entre', 'así', 'desde',
            'ni', 'nos', 'día', 'uno', 'bien', 'poco', 'entonces', 'tan', 'ahora',
            'después', 'siempre', 'solo', 'algo', 'cada', 'menos', 'nuevo'
        ]
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(stop_words=spanish_stop_words)
        tfidf_matrix = vectorizer.fit_transform(descriptions)
        query_vector = vectorizer.transform([query])
        
        similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
        
        recommendations = [
            (dest_names[i], similarities[i])
            for i in range(len(dest_names))
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
        """
        Sistema híbrido que combina múltiples enfoques.
        
        Args:
            user_id: ID del usuario.
            user_preferences: Preferencias explícitas.
            query: Búsqueda textual.
            n_recommendations: Número de recomendaciones.
            weights: Pesos (collaborative, content, text).
        
        Returns:
            Lista de tuplas (destino, score_final, scores_detalle).
        """
        collaborative_weight, content_weight, text_weight = weights
        
        collab_recs = self.get_collaborative_recommendations(user_id, n_recommendations * 2)
        collab_scores = {dest: score for dest, score in collab_recs}
        
        content_scores = {}
        if user_preferences:
            content_recs = self.get_content_based_recommendations(user_preferences, n_recommendations * 2)
            content_scores = {dest: score for dest, score in content_recs}
        
        text_scores = {}
        if query:
            text_recs = self.get_text_based_recommendations(query, n_recommendations * 2)
            text_scores = {dest: score for dest, score in text_recs}
        
        all_destinations = set(collab_scores.keys()) | set(content_scores.keys()) | set(text_scores.keys())
        
        hybrid_scores = []
        for dest in all_destinations:
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
        """Entrena modelo KNN."""
        dest_names = list(self.destination_features.keys())
        feature_keys = ["culture", "beach", "mountains", "nightlife", "historical", "modern"]
        
        feature_matrix = np.array([
            [self.destination_features[dest].get(key, 0) for key in feature_keys]
            for dest in dest_names
        ])
        
        # Ajustar n_neighbors si hay pocos destinos
        total_destinations = len(dest_names)
        actual_neighbors = min(n_neighbors, total_destinations)
        
        self.knn_model = NearestNeighbors(n_neighbors=actual_neighbors, metric='cosine')
        self.knn_model.fit(feature_matrix)
        
        logger.info(f"Modelo KNN entrenado con {len(dest_names)} destinos (max neighbors: {actual_neighbors})")

    def get_similar_destinations(
        self,
        destination: str,
        n_similar: int = 3
    ) -> List[Tuple[str, float]]:
        """Encuentra destinos similares usando KNN."""
        if self.knn_model is None:
            self.train_knn_model()
        
        dest_names = list(self.destination_features.keys())
        
        if destination not in dest_names:
            return []
        
        feature_keys = ["culture", "beach", "mountains", "nightlife", "historical", "modern"]
        dest_vector = np.array([
            [self.destination_features[destination].get(key, 0) for key in feature_keys]
        ])
        
        distances, indices = self.knn_model.kneighbors(dest_vector, n_neighbors=n_similar + 1)
        
        dest_idx = dest_names.index(destination)
        similar = [
            (dest_names[idx], 1 - distances[0][i])
            for i, idx in enumerate(indices[0])
            if idx != dest_idx
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


# Ejemplo de uso
if __name__ == "__main__":
    print("=" * 60)
    print("Sistema de Recomendaciones de Viajes con ML")
    print("=" * 60)
    
    recommender = TravelRecommender()
    
    print("\n1. Recomendaciones Collaborative Filtering (user_1):")
    collab_recs = recommender.get_collaborative_recommendations("user_1", n_recommendations=3)
    if collab_recs:
        for dest, score in collab_recs:
            print(f"   {dest}: {score:.3f}")
    else:
        print("   (No hay suficientes datos)")
    
    print("\n2. Recomendaciones Content-Based:")
    print("   Preferencias: cultura alta, playa media, vida nocturna alta")
    preferences = {"culture": 0.9, "beach": 0.6, "nightlife": 0.7}
    content_recs = recommender.get_content_based_recommendations(preferences, n_recommendations=3)
    for dest, score in content_recs:
        print(f"   {dest}: {score:.3f}")
    
    print("\n3. Recomendaciones Text-Based:")
    print("   Query: 'playa arquitectura moderna'")
    text_recs = recommender.get_text_based_recommendations("playa arquitectura moderna", n_recommendations=3)
    for dest, score in text_recs:
        print(f"   {dest}: {score:.3f}")
    
    print("\n4. Recomendaciones Híbridas (user_1):")
    hybrid_recs = recommender.get_hybrid_recommendations(
        user_id="user_1",
        user_preferences=preferences,
        query="playa cultura",
        n_recommendations=3
    )
    for dest, score, detail in hybrid_recs:
        print(f"   {dest}: {score:.3f}")
        print(f"      Collab: {detail['collaborative']:.2f}, Content: {detail['content']:.2f}, Text: {detail['text']:.2f}")
    
    print("\n5. Destinos Similares:")
    dest_list = list(recommender.destination_features.keys())
    if len(dest_list) > 1:
        sample_dest = dest_list[0]
        print(f"   Similares a {sample_dest}:")
        # Pedir máximo (total_destinos - 1) similares
        n_similar = min(2, len(dest_list) - 1)
        similar = recommender.get_similar_destinations(sample_dest, n_similar=n_similar)
        for dest, similarity in similar:
            print(f"      {dest}: {similarity:.3f}")
    else:
        print("   (No hay suficientes destinos)")
    
    stats = recommender.get_statistics()
    print("\n6. Estadísticas:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
