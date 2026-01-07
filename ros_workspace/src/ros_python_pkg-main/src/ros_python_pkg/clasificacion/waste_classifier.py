import numpy as np
from typing import Dict, Tuple


class WasteClassifier:
    """
    Clasificador de residuos basado en reglas sencillas.

    Clases:
        - LATA
        - BOTELLA
        - CARTON

    La idea es:
        1) Separar objetos METÁLICOS de NO METÁLICOS.
        2) En los NO METÁLICOS, distinguir BOTELLA (alargada / tapa circular)
           de CARTON (rectangular tipo brick).
    """

    CLASS_CAN = "LATA"
    CLASS_BOTTLE = "BOTELLA"
    CLASS_CARDBOARD = "CARTON"
    CLASS_UNKNOWN = "DESCONOCIDO"

    def __init__(self, confidence_threshold: float = 0.5):
        """
        Args:
            confidence_threshold: mínimo de confianza para aceptar una clase.
        """
        self.confidence_threshold = confidence_threshold

    def classify(self, features: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        """
        Clasifica un objeto a partir de sus características.

        Devuelve:
            (clase_predicha, confianza [0,1], diccionario_scores_por_clase)
        """
        scores = {
            self.CLASS_CAN:        self._score_can(features),
            self.CLASS_BOTTLE:     self._score_bottle(features),
            self.CLASS_CARDBOARD:  self._score_cardboard(features),
        }

        # Elegir mejor clase
        best_class = max(scores, key=scores.get)
        raw_vals = np.array(list(scores.values()), dtype=float)

        # Normalizamos a [0,1] para sacar algo tipo "probabilidad"
        raw_vals -= raw_vals.min()
        total = raw_vals.sum()
        if total <= 0:
            confidence = 0.0
        else:
            probs = raw_vals / total
            confidence = float(probs[list(scores.keys()).index(best_class)])

        # Si la confianza es baja, devolvemos DESCONOCIDO
        if confidence < self.confidence_threshold:
            return self.CLASS_UNKNOWN, confidence, scores

        return best_class, confidence, scores

       # ================================
    # LATA
    # ================================
    def _score_can(self, f: Dict[str, float]) -> float:
        metallic   = f.get("metallic_score", 0.0)
        is_met_col = f.get("is_metallic_color", 0.0) > 0.5
        circ       = f.get("circularity", 0.0)
        elong      = f.get("elongation_ratio", 1.0)
        aspect     = f.get("aspect_ratio", 1.0)
        edge_top   = f.get("edge_density_top", 0.0)
        spec_top   = f.get("specular_ratio_top", 0.0)

        score = 0.0

        # 1) metal → base de lata
        if metallic > 0.55:
            score += 2.0
        if is_met_col:
            score += 1.0

        # 2) lata rectangular (Smints, Aquarius tumbada):
        #    metálica y algo alargada pero NO exageradamente
        if 1.4 <= elong <= 2.4:
            score += 3.0

        # 3) lata "disco" (Aquarius desde arriba):
        #    muy circular, con tapa con bordes
        if circ > 0.75 and edge_top > 0.08 and spec_top > 0.15:
            score += 3.0

        # 4) penalizar cosas demasiado largas → parecen botella tumbada
        if elong > 2.6:
            score -= 3.0

        return score

    # ================================
    # BOTELLA
    # ================================
    def _score_bottle(self, f: Dict[str, float]) -> float:
        metallic = f.get("metallic_score", 0.0)
        circ     = f.get("circularity", 0.0)
        elong    = f.get("elongation_ratio", 1.0)
        aspect   = f.get("aspect_ratio", 1.0)
        edge_top = f.get("edge_density_top", 0.0)
        spec_top = f.get("specular_ratio_top", 0.0)

        score = 0.0

        # 1) botella tumbada: muy alargada
        if elong > 2.6 or aspect > 2.6 or aspect < 1.0 / 2.6:
            score += 6.0
        elif elong > 2.0 or aspect > 2.0 or aspect < 0.5:
            score += 3.0

        # 2) botella de pie: casi cuadrada, brilla poco arriba y pocos bordes
        is_square = 0.70 <= aspect <= 1.30 and elong < 1.8
        if is_square and spec_top < 0.2 and edge_top < 0.06:
            score += 3.0

        # 3) ligera preferencia por botellas menos metálicas
        if metallic < 0.6:
            score += 1.0

        return score

    # ================================
    # CARTON
    # ================================
    def _score_cardboard(self, f: Dict[str, float]) -> float:
        metallic   = f.get("metallic_score", 0.0)
        is_met_col = f.get("is_metallic_color", 0.0) > 0.5
        circ       = f.get("circularity", 0.0)
        elong      = f.get("elongation_ratio", 1.0)
        aspect     = f.get("aspect_ratio", 1.0)
        spec_top   = f.get("specular_ratio_top", 0.0)

        score = 0.0

        is_square = 0.70 <= aspect <= 1.30 and elong < 1.8

        # 1) forma cuadrada típica de brick
        if is_square:
            score += 3.0

            # Cartón visto de lado (como tu objeto 1 en esta foto):
            # cuadrado, circ > 0.70, poca luz en la parte superior
            if circ > 0.70 and spec_top < 0.10:
                score += 2.0

            # Cartón visto desde arriba: muchísima luz en la tapa
            if spec_top > 0.6:
                score += 3.0

        # 2) si es muy circular y metálico, es más probable lata que cartón
        if circ > 0.85 and metallic > 0.6:
            score -= 3.0

        # 3) pequeña penalización si es claramente metálico y de color metálico
        if metallic > 0.65 and is_met_col:
            score -= 1.0

        return score


    # ------------------------------------------------------------------ #
    # Utilidades visualización
    # ------------------------------------------------------------------ #
    def get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """Color BGR para dibujar cada clase."""
        colors = {
            self.CLASS_CAN:       (0, 255, 255),   # Amarillo
            self.CLASS_BOTTLE:    (255, 0, 0),     # Azul
            self.CLASS_CARDBOARD: (0, 165, 255),   # Naranja
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
        print(f"{prefix}=== CLASSIFICATION RESULT ===")
        print(f"{prefix}Predicted Class: {class_name}")
        print(f"{prefix}Confidence: {confidence:.2%}")
        print(f"{prefix}Scores:")
        for cls, s in scores.items():
            print(f"{prefix}  {cls}: {s:.2f}")
