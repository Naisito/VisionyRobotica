import numpy as np
from collections import deque, Counter
from typing import List, Dict, Tuple

class WasteStabilizer:
    """
    Estabiliza las clasificaciones en tiempo real usando suavizado temporal.
    Rastrea objetos entre frames y usa un historial de predicciones.
    """
    
    def __init__(self, history_size: int = 10, max_distance: int = 50, missing_threshold: int = 5):
        """
        Args:
            history_size: Número de frames para el voto por mayoría.
            max_distance: Distancia máxima (px) para considerar que es el mismo objeto.
            missing_threshold: Frames que un objeto puede desaparecer antes de borrarlo.
        """
        self.history_size = history_size
        self.max_distance = max_distance
        self.missing_threshold = missing_threshold
        
        # Estructura de tracks:
        # {
        #   track_id: {
        #       'history': deque(maxlen=history_size),
        #       'center': (x, y),
        #       'missing_count': 0,
        #       'last_seen_class': str
        #   }
        # }
        self.tracks = {}
        self.next_track_id = 1
        
    def update(self, detections: List[dict]) -> List[dict]:
        """
        Actualiza los tracks con las nuevas detecciones y devuelve resultados estabilizados.
        
        Args:
            detections: Lista de dicts con keys 'center', 'class', 'confidence', etc.
            
        Returns:
            Lista de detecciones estabilizadas (con 'class' suavizada).
        """
        # 1. Predecir/Marcar todos como missing temporalmente
        for track_id in self.tracks:
            self.tracks[track_id]['missing_count'] += 1
            
        # 2. Asociar detecciones a tracks existentes (Greedy nearest neighbor)
        # Matriz de distancias
        if not detections:
            pass # No hay detecciones, todos incrementan missing_count
        else:
            # Copia para no modificar la lista original mientras iteramos
            unmatched_detections = []
            for i, det in enumerate(detections):
                unmatched_detections.append((i, det))
            
            # Intentar asociar cada track existente con la detección más cercana
            for track_id, track_data in self.tracks.items():
                if not unmatched_detections:
                    break
                    
                track_center = track_data['center']
                best_dist = float('inf')
                best_idx = -1
                
                for idx, (orig_idx, det) in enumerate(unmatched_detections):
                    det_center = det['center']
                    dist = np.sqrt((track_center[0] - det_center[0])**2 + (track_center[1] - det_center[1])**2)
                    
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = idx
                
                # Si encontramos una coincidencia válida
                if best_idx != -1 and best_dist < self.max_distance:
                    # Actualizar track
                    orig_idx, det = unmatched_detections.pop(best_idx)
                    
                    self.tracks[track_id]['missing_count'] = 0
                    self.tracks[track_id]['center'] = det['center']
                    self.tracks[track_id]['history'].append(det['class'])
                    
                    # Guardar referencia a la detección original para devolver info extra (contour, etc)
                    self.tracks[track_id]['current_det'] = det
        
            # 3. Crear nuevos tracks para detecciones no asociadas
            for _, det in unmatched_detections:
                self.tracks[self.next_track_id] = {
                    'history': deque([det['class']], maxlen=self.history_size),
                    'center': det['center'],
                    'missing_count': 0,
                    'current_det': det
                }
                self.next_track_id += 1
        
        # 4. Limpiar tracks perdidos
        tracks_to_remove = []
        for track_id, track_data in self.tracks.items():
            if track_data['missing_count'] > self.missing_threshold:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
            
        # 5. Generar resultados estabilizados
        stabilized_results = []
        for track_id, track_data in self.tracks.items():
            # Solo devolver si el objeto fue visto en este frame (missing_count == 0)
            # Opcional: podríamos devolver objetos "perdidos" brevemente (ghosts), 
            # pero para visualización suele ser mejor solo mostrar lo que se ve.
            if track_data['missing_count'] == 0:
                # Voto por mayoría
                counts = Counter(track_data['history'])
                most_common_class, _ = counts.most_common(1)[0]
                
                # Construir resultado mezclando info original con clase estabilizada
                original_det = track_data['current_det']
                result = original_det.copy()
                result['class'] = most_common_class
                result['track_id'] = track_id # Útil para debug
                
                # Opcional: recalcular confianza basada en estabilidad
                # result['confidence'] = counts[most_common_class] / len(track_data['history'])
                
                stabilized_results.append(result)
                
        return stabilized_results
