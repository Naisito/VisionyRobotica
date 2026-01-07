import cv2
import numpy as np
from typing import Dict, Tuple, List


class FeatureExtractor:
    """
    Extrae características de objetos detectados para clasificación.
    Características:
        - Forma, circularidad, color, metalicidad, textura, tamaño (para reglas)
        - Y además: vector de features tipo train_features.py (para RandomForest)
    """

    def __init__(self):
        pass

    def extract_features(self, image: np.ndarray, contour: np.ndarray,
                         mask: np.ndarray = None) -> Dict[str, float]:
        """
        Extrae todas las características de un objeto dado su contorno.

        Args:
            image: Imagen original en BGR
            contour: Contorno del objeto
            mask: Máscara binaria del objeto (opcional)

        Returns:
            Diccionario con todas las características extraídas
        """
        features: Dict[str, float] = {}

        # --- Máscara del contorno (una sola vez) ---
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)

        # === FEATURES PARA CLASIFICADOR POR REGLAS (ya existentes) ===
        features.update(self._extract_shape_features(contour))
        features.update(self._extract_color_features(image, contour, mask))
        features.update(self._extract_metallic_features(image, contour, mask))
        features.update(self._extract_texture_features(image, contour, mask))

        # === NUEVO: FEATURES TIPO train_features.py (para RandomForest) ===
        ml_feats, ml_vec = self._extract_ml_features(image, contour, mask)
        features.update(ml_feats)
        # Guardamos el vector en el mismo orden que en el entrenamiento
        features["ml_feature_vector"] = ml_vec

        return features

    # ------------------------------------------------------------------
    # FEATURES ORIGINALES (REGLAS)
    # ------------------------------------------------------------------
    def _extract_shape_features(self, contour: np.ndarray) -> Dict[str, float]:
        """Extrae características de forma del contorno."""
        features = {}

        # Área y perímetro
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        features['area'] = area
        features['perimeter'] = perimeter

        # Bounding box axis-alineado
        x, y, w, h = cv2.boundingRect(contour)
        features['bbox_width'] = w
        features['bbox_height'] = h
        features['bbox_area'] = w * h

        # Aspect ratio (relación de aspecto w/h)
        if h > 0:
            features['aspect_ratio'] = float(w) / h
        else:
            features['aspect_ratio'] = 0.0

        # Bounding box rotado: elongación y orientación
        rot_rect = cv2.minAreaRect(contour)  # ((cx,cy), (w_rot, h_rot), angle)
        (cx_rect, cy_rect), (w_rot, h_rot), angle = rot_rect

        w_rot = float(max(w_rot, 1.0))
        h_rot = float(max(h_rot, 1.0))
        long_side = max(w_rot, h_rot)
        short_side = min(w_rot, h_rot)
        elongation = long_side / short_side

        features['rotated_width'] = w_rot
        features['rotated_height'] = h_rot
        features['elongation_ratio'] = elongation
        features['orientation_angle'] = float(angle)

        # Extent
        if w * h > 0:
            features['extent'] = area / (w * h)
        else:
            features['extent'] = 0.0

        # Circularidad
        if perimeter > 0:
            features['circularity'] = (4 * np.pi * area) / (perimeter ** 2)
        else:
            features['circularity'] = 0.0

        # Compacidad
        if perimeter > 0:
            features['compactness'] = area / perimeter
        else:
            features['compactness'] = 0.0

        # Aproximación poligonal
        epsilon = 0.02 * perimeter if perimeter > 0 else 0
        approx = cv2.approxPolyDP(contour, epsilon, True)
        features['num_vertices'] = len(approx)

        # Convex hull
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            features['solidity'] = area / hull_area
        else:
            features['solidity'] = 0.0

        # Momentos / centroide
        moments = cv2.moments(contour)
        if moments['m00'] > 0:
            cx = moments['m10'] / moments['m00']
            cy = moments['m01'] / moments['m00']
            features['centroid_x'] = cx
            features['centroid_y'] = cy
        else:
            features['centroid_x'] = 0.0
            features['centroid_y'] = 0.0

        return features

    def _extract_color_features(self, image: np.ndarray,
                                contour: np.ndarray,
                                mask: np.ndarray = None) -> Dict[str, float]:
        """Extrae características de color del objeto."""
        features = {}

        # Máscara
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Medias HSV
        mean_hsv = cv2.mean(hsv, mask=mask)
        features['hue_mean'] = mean_hsv[0]
        features['saturation_mean'] = mean_hsv[1]
        features['value_mean'] = mean_hsv[2]

        # Desviaciones estándar
        hsv_pixels = hsv[mask > 0]
        if len(hsv_pixels) > 0:
            std_hsv = np.std(hsv_pixels, axis=0)
            features['hue_std'] = std_hsv[0]
            features['saturation_std'] = std_hsv[1]
            features['value_std'] = std_hsv[2]
        else:
            features['hue_std'] = 0.0
            features['saturation_std'] = 0.0
            features['value_std'] = 0.0

        # Histograma de tono
        hist_h = cv2.calcHist([hsv], [0], mask, [180], [0, 180])
        features['dominant_hue'] = int(np.argmax(hist_h))

        # Heurísticas de color
        is_metallic_color = (mean_hsv[1] < 50 and mean_hsv[2] > 150) or \
                            (15 <= mean_hsv[0] <= 45 and mean_hsv[1] > 100)
        features['is_metallic_color'] = 1.0 if is_metallic_color else 0.0

        # Modificado: Detectar plástico blanco o transparente
        # Antes exigía V < 100, lo que excluía botellas blancas.
        # Ahora buscamos baja saturación en general.
        is_transparent = mean_hsv[1] < 60 
        features['is_transparent_color'] = 1.0 if is_transparent else 0.0

        is_brown = (10 <= mean_hsv[0] <= 30 and 20 <= mean_hsv[1] <= 150)
        features['is_brown_color'] = 1.0 if is_brown else 0.0

        return features

    def _extract_metallic_features(self, image: np.ndarray,
                                   contour: np.ndarray,
                                   mask: np.ndarray = None) -> Dict[str, float]:
        """Extrae características de metalicidad (brillo especular, reflexiones)."""
        features = {}

        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        bright_threshold = 200
        bright_pixels = np.sum((gray > bright_threshold) & (mask > 0))
        total_pixels = np.sum(mask > 0)

        if total_pixels > 0:
            features['specular_ratio'] = bright_pixels / total_pixels
        else:
            features['specular_ratio'] = 0.0

        # Parte superior (tapa)
        x, y, w, h = cv2.boundingRect(contour)
        if w > 0 and h > 0:
            top_h = max(1, int(h * 0.25))
            top_mask = np.zeros_like(mask)
            top_mask[y:y + top_h, x:x + w] = mask[y:y + top_h, x:x + w]

            bright_pixels_top = np.sum((gray > bright_threshold) & (top_mask > 0))
            total_pixels_top = np.sum(top_mask > 0)

            if total_pixels_top > 0:
                features['specular_ratio_top'] = bright_pixels_top / total_pixels_top
            else:
                features['specular_ratio_top'] = 0.0
        else:
            features['specular_ratio_top'] = 0.0

        # Gradiente
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

        masked_gradient = gradient_magnitude[mask > 0]
        if len(masked_gradient) > 0:
            features['gradient_mean'] = float(np.mean(masked_gradient))
            features['gradient_std'] = float(np.std(masked_gradient))
        else:
            features['gradient_mean'] = 0.0
            features['gradient_std'] = 0.0

        # Rango de intensidad
        masked_gray = gray[mask > 0]
        if len(masked_gray) > 0:
            features['intensity_range'] = float(np.max(masked_gray) - np.min(masked_gray))
            features['intensity_mean'] = float(np.mean(masked_gray))
            features['intensity_std'] = float(np.std(masked_gray))
        else:
            features['intensity_range'] = 0.0
            features['intensity_mean'] = 0.0
            features['intensity_std'] = 0.0

        metallic_score = (
            features['specular_ratio'] * 0.4 +
            min(features['gradient_mean'] / 100.0, 1.0) * 0.3 +
            min(features['intensity_range'] / 255.0, 1.0) * 0.3
        )
        features['metallic_score'] = metallic_score

        return features

    def _extract_texture_features(self, image: np.ndarray,
                                  contour: np.ndarray,
                                  mask: np.ndarray = None) -> Dict[str, float]:
        """Extrae características de textura del objeto."""
        features = {}

        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Varianza de intensidad
        masked_gray = gray[mask > 0]
        if len(masked_gray) > 0:
            features['texture_variance'] = float(np.var(masked_gray))
        else:
            features['texture_variance'] = 0.0

        # Bordes internos
        edges = cv2.Canny(gray, 50, 150)
        internal_edges = edges & mask
        edge_pixels = np.sum(internal_edges > 0)
        total_pixels = np.sum(mask > 0)

        if total_pixels > 0:
            features['edge_density'] = edge_pixels / total_pixels
        else:
            features['edge_density'] = 0.0

        # Parte superior vs cuerpo
        x, y, w, h = cv2.boundingRect(contour)
        if w > 0 and h > 0:
            top_h = max(1, int(h * 0.25))
            top_edges = internal_edges[y:y + top_h, x:x + w]
            body_edges = internal_edges[y + top_h:y + h, x:x + w]

            top_pixels = top_edges.size
            body_pixels = body_edges.size

            if top_pixels > 0:
                features['edge_density_top'] = float(np.sum(top_edges > 0) / top_pixels)
            else:
                features['edge_density_top'] = 0.0

            if body_pixels > 0:
                features['edge_density_body'] = float(np.sum(body_edges > 0) / body_pixels)
            else:
                features['edge_density_body'] = 0.0

            # === NUEVO: densidad de bordes del tapón (banda muy superior) ===
            cap_h = max(1, int(h * 0.15))         # 15% de la altura: zona del tapón
            cap_edges = internal_edges[y:y + cap_h, x:x + w]
            cap_pixels = cap_edges.size

            if cap_pixels > 0:
                features['cap_edge_density'] = float(np.sum(cap_edges > 0) / cap_pixels)
            else:
                features['cap_edge_density'] = 0.0
        else:
            features['edge_density_top'] = 0.0
            features['edge_density_body'] = 0.0
            features['cap_edge_density'] = 0.0


        # Homogeneidad
        if len(masked_gray) > 0:
            std = np.std(masked_gray)
            features['homogeneity'] = 1.0 / (1.0 + std / 50.0)
        else:
            features['homogeneity'] = 0.0

        return features

    # ------------------------------------------------------------------
    # NUEVO: FEATURES TIPO train_features.py
    # ------------------------------------------------------------------
    def _extract_ml_features(self,
                             image: np.ndarray,
                             contour: np.ndarray,
                             mask: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
        """
        Replica (en lo posible) las features de train_features.extract_features,
        pero usando la máscara del contorno ya segmentado.

        Devuelve:
            - diccionario con claves ml_*
            - vector np.array en el mismo orden que en el training original
        """
        ml = {}

        # Copia de la imagen para posible corrección de luz
        img_proc = image.copy()

        # 0. Low Light Enhancement
        hsv_check = cv2.cvtColor(img_proc, cv2.COLOR_BGR2HSV)
        avg_brightness = np.mean(hsv_check[:, :, 2])

        if avg_brightness < 80:
            gamma = 1.5
            invGamma = 1.0 / gamma
            table = np.array(
                [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)],
                dtype="uint8"
            )
            img_proc = cv2.LUT(img_proc, table)

        # 1. Shape features (sobre el contorno, como en el script)
        #    [aspect_ratio, solidity, extent, circularity, rectangularity]
        aspect_ratio = 0.0
        solidity = 0.0
        extent = 0.0
        circularity = 0.0
        rectangularity = 0.0

        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        if area > 0:
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            if h_rect > 0:
                aspect_ratio = float(w_rect) / h_rect

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0.0

            rect_area = w_rect * h_rect
            extent = float(area) / rect_area if rect_area > 0 else 0.0

            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter * perimeter)
            else:
                circularity = 0.0

            min_rect = cv2.minAreaRect(contour)
            min_rect_area = float(min_rect[1][0]) * float(min_rect[1][1])
            rectangularity = float(area) / min_rect_area if min_rect_area > 0 else 0.0

        ml['ml_aspect_ratio'] = aspect_ratio
        ml['ml_solidity'] = solidity
        ml['ml_extent'] = extent
        ml['ml_circularity'] = circularity
        ml['ml_rectangularity'] = rectangularity

        # 2. Color features (Mean Sin/Cos Hue, CircVar, stats de S y V)
        #    Reescalamos a 160x160 y la máscara igual (como en el script)
        img_resized = cv2.resize(img_proc, (160, 160))
        mask_resized = cv2.resize(mask, (160, 160), interpolation=cv2.INTER_NEAREST)

        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)

        if cv2.countNonZero(mask_resized) > 0:
            mask_bool = mask_resized > 0

            h_channel = hsv[:, :, 0].astype(float)
            h_rad = h_channel * (2 * np.pi / 180.0)

            sin_h = np.sin(h_rad)
            cos_h = np.cos(h_rad)

            mean_sin_h = float(np.mean(sin_h[mask_bool]))
            mean_cos_h = float(np.mean(cos_h[mask_bool]))

            R = np.sqrt(mean_sin_h ** 2 + mean_cos_h ** 2)
            circ_var_h = float(1 - R)

            s_channel = hsv[:, :, 1].astype(float)
            v_channel = hsv[:, :, 2].astype(float)

            s_vals = s_channel[mask_bool]
            v_vals = v_channel[mask_bool]

            mean_s = float(np.mean(s_vals))
            mean_v = float(np.mean(v_vals))
            std_s = float(np.std(s_vals))
            std_v = float(np.std(v_vals))
            max_s = float(np.max(s_vals))
            max_v = float(np.max(v_vals))
            skew_s = float(self._safe_skew(s_vals))
            skew_v = float(self._safe_skew(v_vals))
        else:
            mean_sin_h = 0.0
            mean_cos_h = 0.0
            circ_var_h = 1.0
            mean_s = 0.0
            mean_v = 0.0
            std_s = 0.0
            std_v = 0.0
            max_s = 0.0
            max_v = 0.0
            skew_s = 0.0
            skew_v = 0.0

        ml['ml_mean_sin_h'] = mean_sin_h
        ml['ml_mean_cos_h'] = mean_cos_h
        ml['ml_circ_var_h'] = circ_var_h
        ml['ml_mean_s'] = mean_s
        ml['ml_mean_v'] = mean_v
        ml['ml_std_s'] = std_s
        ml['ml_std_v'] = std_v
        ml['ml_max_s'] = max_s
        ml['ml_max_v'] = max_v
        ml['ml_skew_s'] = skew_s
        ml['ml_skew_v'] = skew_v

        # 3. Texture: Edge Density
        gray_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_resized, 100, 200)
        edges_masked = cv2.bitwise_and(edges, edges, mask=mask_resized)
        edge_density = float(np.sum(edges_masked) / (edges.shape[0] * edges.shape[1]))
        ml['ml_edge_density'] = edge_density

        # Vector en el mismo orden que en train_features.py
        ml_vec = np.array([
            ml['ml_aspect_ratio'],
            ml['ml_solidity'],
            ml['ml_extent'],
            ml['ml_circularity'],
            ml['ml_rectangularity'],
            ml['ml_mean_sin_h'],
            ml['ml_mean_cos_h'],
            ml['ml_circ_var_h'],
            ml['ml_mean_s'],
            ml['ml_mean_v'],
            ml['ml_std_s'],
            ml['ml_std_v'],
            ml['ml_max_s'],
            ml['ml_max_v'],
            ml['ml_skew_s'],
            ml['ml_skew_v'],
            ml['ml_edge_density'],
        ], dtype=float)

        return ml, ml_vec

    @staticmethod
    def _safe_skew(arr: np.ndarray) -> float:
        """Calcula un skew sencillo sin depender de scipy."""
        arr = np.asarray(arr, dtype=float)
        n = arr.size
        if n < 2:
            return 0.0
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return 0.0
        z = (arr - mean) / std
        return float(np.mean(z ** 3))

    # ------------------------------------------------------------------
    # (opcional) método para imprimir como antes
    # ------------------------------------------------------------------
    def print_features(self, features: Dict[str, float], indent: int = 0):
        """Imprime las características de forma legible."""
        prefix = "  " * indent

        print(f"{prefix}=== SHAPE FEATURES ===")
        print(f"{prefix}Area: {features['area']:.1f}")
        print(f"{prefix}Perimeter: {features['perimeter']:.1f}")
        print(f"{prefix}Aspect Ratio: {features['aspect_ratio']:.2f}")
        print(f"{prefix}Elongation Ratio: {features.get('elongation_ratio', 0.0):.2f}")
        print(f"{prefix}Circularity: {features['circularity']:.3f}")
        print(f"{prefix}Compactness: {features['compactness']:.2f}")
        print(f"{prefix}Vertices: {features['num_vertices']}")
        print(f"{prefix}Solidity: {features['solidity']:.3f}")

        print(f"\n{prefix}=== COLOR FEATURES ===")
        print(f"{prefix}Hue Mean: {features['hue_mean']:.1f}")
        print(f"{prefix}Saturation Mean: {features['saturation_mean']:.1f}")
        print(f"{prefix}Value Mean: {features['value_mean']:.1f}")
        print(f"{prefix}Is Metallic Color: {features['is_metallic_color']}")
        print(f"{prefix}Is Transparent Color: {features['is_transparent_color']}")
        print(f"{prefix}Is Brown Color: {features['is_brown_color']}")

        print(f"\n{prefix}=== METALLIC FEATURES ===")
        print(f"{prefix}Specular Ratio: {features['specular_ratio']:.3f}")
        print(f"{prefix}Specular Ratio Top: {features.get('specular_ratio_top', 0.0):.3f}")
        print(f"{prefix}Gradient Mean: {features['gradient_mean']:.2f}")
        print(f"{prefix}Intensity Range: {features['intensity_range']:.1f}")
        print(f"{prefix}Metallic Score: {features['metallic_score']:.3f}")

        print(f"\n{prefix}=== TEXTURE FEATURES ===")
        print(f"{prefix}Texture Variance: {features['texture_variance']:.2f}")
        print(f"{prefix}Edge Density: {features['edge_density']:.3f}")
        print(f"{prefix}Edge Density Top: {features.get('edge_density_top', 0.0):.3f}")
        print(f"{prefix}Edge Density Body: {features.get('edge_density_body', 0.0):.3f}")
        print(f"{prefix}Homogeneity: {features['homogeneity']:.3f}")
