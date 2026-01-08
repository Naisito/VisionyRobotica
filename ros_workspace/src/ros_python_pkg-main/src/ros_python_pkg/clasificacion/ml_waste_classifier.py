import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple

class MLWasteClassifier:
    """
    Clasificador de residuos basado en Machine Learning.
    Usa un modelo entrenado (KNN, SVC, RF) para clasificar.
    """
    
    CLASS_UNKNOWN = "DESCONOCIDO"

    def __init__(self, model_dir='models', confidence_threshold: float = 0.35):
        """
        Args:
            model_dir: Directorio donde están los archivos .pkl
            confidence_threshold: Umbral de confianza
        """
        self.confidence_threshold = confidence_threshold
        self.model_dir = model_dir

        # Try to resolve model directory (project root 'models' or package 'clasificacion/models')
        def _dir_has_required(d: str) -> bool:
            req = ['waste_classifier_model.pkl', 'scaler.pkl', 'label_encoder.pkl', 'feature_columns.pkl']
            return all(os.path.exists(os.path.join(d, r)) for r in req)

        resolved_dir = model_dir
        if not _dir_has_required(resolved_dir):
            pkg_dir = os.path.join(os.path.dirname(__file__), 'models')
            if _dir_has_required(pkg_dir):
                resolved_dir = pkg_dir
        self.model_dir = resolved_dir

        try:
            self.model = joblib.load(os.path.join(self.model_dir, 'waste_classifier_model.pkl'))
            self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
            self.encoder = joblib.load(os.path.join(self.model_dir, 'label_encoder.pkl'))
            self.feature_columns = joblib.load(os.path.join(self.model_dir, 'feature_columns.pkl'))
            self.classes = self.encoder.classes_
        except Exception as e:
            self.model = None
            self.scaler = None
            self.encoder = None
            self.feature_columns = None
            self.classes = []

    def classify(self, features: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        """
        Clasifica un objeto a partir de sus características.

        Devuelve:
            (clase_predicha, confianza [0,1], diccionario_scores_por_clase)
        """
        if self.model is None:
            return self.CLASS_UNKNOWN, 0.0, {}

        # Filtrar SOLO features ML
        ml_features = {k: v for k, v in features.items() if k.startswith('ml_')}
        
        # Preparar features para el modelo
        # El modelo espera un DataFrame o array con las mismas columnas que en el entrenamiento
        # Convertimos el dict a DataFrame de una fila
        df = pd.DataFrame([ml_features])
        
        # Asegurar que las columnas numéricas sean float y rellenar nulos
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Asegurar orden de columnas
        if hasattr(self, 'feature_columns'):
            # Reindexar para tener las mismas columnas que en el entrenamiento
            # Rellenar con 0 si falta alguna (aunque no debería)
            df = df.reindex(columns=self.feature_columns, fill_value=0)
        
        # Escalar
        try:
            X_scaled = self.scaler.transform(df)
        except Exception as e:
            print(f"Error al escalar features: {e}")
            return self.CLASS_UNKNOWN, 0.0, {}

        # Predecir probabilidades
        try:
            probs = self.model.predict_proba(X_scaled)[0]
        except Exception as e:
            print(f"Error al predecir: {e}")
            return self.CLASS_UNKNOWN, 0.0, {}
            
        # Obtener clase con mayor probabilidad
        max_prob_idx = np.argmax(probs)
        confidence = probs[max_prob_idx]
        predicted_class = self.classes[max_prob_idx]
        
        # Crear diccionario de scores
        scores = {cls: prob for cls, prob in zip(self.classes, probs)}
        
        # --- HEURISTIC OVERRIDE ---
        # Si el ML dice lata pero parece de plástico (no metálico y color plástico/blanco)
        # forzamos a botella. Esto corrige la confusión común con botellas blancas redondas.
        is_metallic = features.get('is_metallic_color', 0.0) > 0.5
        is_plastic = features.get('is_transparent_color', 0.0) > 0.5
        metallic_score = features.get('metallic_score', 0.0)
        
        if predicted_class == "lata":
            # Si NO es color metálico Y (es plástico O score metálico muy bajo)
            if not is_metallic and (is_plastic or metallic_score < 0.4):
                print(f"  [OVERRIDE] lata -> botella (No metallic color, plastic/low metallic score)")
                predicted_class = "botella"
                confidence = 0.95 # Confianza artificial alta tras corrección
        
        # --- NUEVO HEURISTIC: Detectar latas que el ML pierde ---
        # El Objeto 5 (lata) tenía metallic_score=0.60 y gradient=106, pero ML dijo botella.
        # Las latas con muchos gráficos tienen gradiente alto y score metálico decente.
        elif predicted_class != "lata":
            gradient_mean = features.get('gradient_mean', 0.0)
            if metallic_score > 0.55 and gradient_mean > 80.0:
                 print(f"  [OVERRIDE] {predicted_class} -> lata (High metallic score {metallic_score:.2f} & gradient {gradient_mean:.1f})")
                 predicted_class = "lata"
                 confidence = 0.90

        # Verificar umbral
        if confidence < self.confidence_threshold:
            print(f"  [DEBUG] Low confidence: {confidence:.2f} < {self.confidence_threshold} (Best: {predicted_class})")
            print(f"  [DEBUG] Scores: {scores}")
            return self.CLASS_UNKNOWN, confidence, scores
            
        return predicted_class, confidence, scores

    def get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """Color BGR para dibujar cada clase."""
        # Reutilizamos los colores estándar, asumiendo nombres estándar
        colors = {
            "lata":       (0, 255, 255),   # Amarillo
            "botella":    (255, 0, 0),     # Azul
            "carton":     (0, 165, 255),   # Naranja
            self.CLASS_UNKNOWN:   (128, 128, 128)  # Gris
        }
        return colors.get(class_name, (255, 255, 255))

    def print_classification(
        self,
        class_name: str,
        confidence: float,
        scores: Dict[str, float],
        indent: int = 0,
    ):
        prefix = "  " * indent
        print(f"{prefix}=== ML CLASSIFICATION RESULT ===")
        print(f"{prefix}Predicted Class: {class_name}")
        print(f"{prefix}Confidence: {confidence:.2%}")
        print(f"{prefix}Scores:")
        for cls, s in scores.items():
            print(f"{prefix}  {cls}: {s:.2f}")
