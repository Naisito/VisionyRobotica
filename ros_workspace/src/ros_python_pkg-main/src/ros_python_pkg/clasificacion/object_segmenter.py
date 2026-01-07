import cv2
import numpy as np
from typing import List, Tuple, Optional


class ObjectSegmenter:
    """
    Segmenta objetos en imágenes usando estrategia multi-modal:
    - Segmentación por color
    - Detección de bordes
    - Umbral adaptativo
    Basado en test_new_segmentation.py pero mejorado y modularizado.
    """
    
    def __init__(self, 
                 min_area: int = 1000,
                 brightness_threshold: int = 80,
                 gamma_correction: float = 1.5):
        """
        Args:
            min_area: Área mínima para considerar un contorno válido
            brightness_threshold: Umbral de brillo para aplicar corrección gamma
            gamma_correction: Factor de corrección gamma para imágenes oscuras
        """
        self.min_area = min_area
        self.brightness_threshold = brightness_threshold
        self.gamma_correction = gamma_correction
    
    def segment_objects(self, 
                       image: np.ndarray,
                       roi_mask: Optional[np.ndarray] = None,
                       debug: bool = False) -> Tuple[List[np.ndarray], dict]:
        """
        Segmenta objetos en la imagen.
        
        Args:
            image: Imagen de entrada en BGR
            roi_mask: Máscara ROI opcional (255 = válido, 0 = excluido)
            debug: Si True, retorna imágenes de debug
            
        Returns:
            Tupla (contours, debug_info) donde:
            - contours: Lista de contornos válidos
            - debug_info: Diccionario con información y imágenes de debug
        """
        debug_info = {}
        
        # Preprocesamiento: corrección de brillo si es necesario
        processed_img = self._preprocess_image(image)
        debug_info['preprocessed'] = processed_img
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        debug_info['gray'] = gray
        
        # Estrategia 1: Segmentación por color
        mask_colored = self._color_segmentation(processed_img)
        debug_info['color_mask'] = mask_colored
        
        # Estrategia 2: Detección de bordes
        mask_edges = self._edge_detection(gray)
        debug_info['edge_mask'] = mask_edges
        
        # Estrategia 3: Umbral adaptativo
        mask_adaptive = self._adaptive_threshold(gray)
        debug_info['adaptive_mask'] = mask_adaptive
        
        # SALVAGUARDA: Verificar si las máscaras auxiliares cubren demasiada área (fondo detectado)
        total_pixels = image.shape[0] * image.shape[1]
        
        # Verificar mask_edges
        edges_area = cv2.countNonZero(mask_edges)
        if edges_area > total_pixels * 0.3:  # Si más del 30% son bordes, es ruido
            if debug: print(f"WARN: Edge mask descartada (Área {edges_area/total_pixels:.1%})")
            mask_edges = np.zeros_like(mask_edges)
            
        # Verificar mask_adaptive
        adaptive_area = cv2.countNonZero(mask_adaptive)
        if adaptive_area > total_pixels * 0.4:  # Si más del 40% es objeto, es fondo
            if debug: print(f"WARN: Adaptive mask descartada (Área {adaptive_area/total_pixels:.1%})")
            mask_adaptive = np.zeros_like(mask_adaptive)
        
        # Combinar estrategias de forma inteligente
        # 1. La máscara de color es la más fiable para latas y objetos con color
        binary = mask_colored.copy()
        
        # 2. Para objetos blancos/difíciles, usar intersección de bordes y adaptive
        # Esto elimina el ruido de adaptive (que no tiene bordes)
        # Dilatar bordes para asegurar solapamiento
        kernel_dilate = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(mask_edges, kernel_dilate, iterations=1)
        
        # Solo considerar adaptive si está cerca de bordes
        structure_mask = cv2.bitwise_and(mask_adaptive, edges_dilated)
        
        # Limpiar ruido de esta máscara estructural
        structure_mask = cv2.morphologyEx(structure_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        
        # Añadir a la máscara principal
        binary = cv2.bitwise_or(binary, structure_mask)
        
        debug_info['combined_mask'] = binary
        
        # Limpieza morfológica
        # Reducimos las iteraciones para evitar unir objetos cercanos
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        # Closing muy suave o nulo para no pegar objetos
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        debug_info['cleaned_mask'] = binary
        
        # Aplicar ROI mask si se proporciona
        if roi_mask is not None:
            binary = cv2.bitwise_and(binary, roi_mask)
            debug_info['roi_applied'] = binary

        # --- WATERSHED PARA SEPARAR OBJETOS ---
        # 1. Sure background (dilatar)
        sure_bg = cv2.dilate(binary, kernel, iterations=2)
        
        # 2. Sure foreground (distance transform)
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        # Umbral para sure foreground: 0.4 * max_dist parece razonable para separar
        ret, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # 3. Unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # 4. Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # 5. Watershed
        # Necesitamos imagen de 3 canales para watershed
        img_for_watershed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(img_for_watershed, markers)
        
        # 6. Crear máscara final basada en marcadores (excluyendo bordes -1 y fondo 1)
        # Los objetos serán > 1
        final_mask = np.zeros_like(binary)
        final_mask[markers > 1] = 255
        
        debug_info['watershed_markers'] = (markers * 10).astype(np.uint8) # Visualización
        debug_info['final_mask'] = final_mask
        
        # Encontrar contornos en la máscara final separada
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrar por área mínima y máxima
        h, w = image.shape[:2]
        max_area = (h * w) * 0.5
        
        valid_contours = []
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            
            # Filtrar por área
            if area < self.min_area or area > max_area:
                if debug: print(f"  Contorno {i}: Rechazado por área ({area:.0f})")
                continue
            
            # Filtrar contornos que tocan los bordes
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            touches_border = (x <= 2 or y <= 2 or 
                            x + w_box >= w - 2 or 
                            y + h_box >= h - 2)
            
            if touches_border:
                if debug: print(f"  Contorno {i}: Rechazado por tocar bordes")
                continue
            
            # Filtrar objetos que son claramente fondo oscuro
            # ... (código de filtrado de fondo igual) ...
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            obj_pixels = hsv[mask > 0]
            
            if len(obj_pixels) > 0:
                mean_value = np.mean(obj_pixels[:, 2])
                mean_saturation = np.mean(obj_pixels[:, 1])
                std_value = np.std(obj_pixels[:, 2])
                std_saturation = np.std(obj_pixels[:, 1])
                
                # Criterios relajados
                is_very_dark = mean_value < 60 # Antes 70
                is_very_uniform = std_value < 15 # Antes 20
                is_low_saturation = mean_saturation < 25 # Antes 30
                is_uniform_saturation = std_saturation < 10 # Antes 15
                
                gray_obj = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray_obj, 50, 150)
                edge_pixels = np.sum((edges > 0) & (mask > 0))
                total_pixels = np.sum(mask > 0)
                edge_density = edge_pixels / total_pixels if total_pixels > 0 else 0
                is_low_edge_density = edge_density < 0.03 # Antes 0.05
                
                background_score = 0
                if is_very_dark: background_score += 2
                if is_very_uniform: background_score += 1
                if is_low_saturation: background_score += 1
                if is_uniform_saturation: background_score += 1
                if is_low_edge_density: background_score += 1
                
                # SALVAGUARDA 1: Si tiene algún píxel con saturación alta, NO es fondo (es colorido)
                max_saturation = np.max(obj_pixels[:, 1])
                if max_saturation > 80:
                    background_score = 0
                    if debug: print(f"  Contorno {i}: Salvado por color (Max Sat: {max_saturation})")

                # SALVAGUARDA 2: Si es muy brillante (promedio V alto), NO es fondo (el fondo es negro)
                # El tapete negro tiene V bajo (< 60-70). Objetos blancos tienen V alto (> 100).
                if mean_value > 80:
                    background_score = 0
                    if debug: print(f"  Contorno {i}: Salvado por brillo (Mean V: {mean_value:.1f})")

                if debug:
                    print(f"  Contorno {i} (Area {area:.0f}): Score {background_score}")
                    # ... prints detallados ...

                # Aumentamos el umbral de rechazo de 3 a 4 para ser más permisivos
                if background_score >= 4:
                    if debug: print(f"  Contorno {i}: Rechazado por fondo (Score {background_score})")
                    continue
            
            valid_contours.append(cnt)
        
        debug_info['num_contours_total'] = len(contours)
        debug_info['num_contours_valid'] = len(valid_contours)
        
        return valid_contours, debug_info

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesa la imagen aplicando corrección gamma si es necesario.
        """
        # Verificar condiciones de baja luz
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        avg_brightness = np.mean(hsv[:, :, 2])
        
        if avg_brightness < self.brightness_threshold:
            # Aplicar corrección gamma
            inv_gamma = 1.0 / self.gamma_correction
            table = np.array([((i / 255.0) ** inv_gamma) * 255 
                            for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(image, table)
        
        return image.copy()
        return mask
    def _color_segmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Segmentación basada en color (objetos con saturación significativa).
        Excluye el fondo oscuro y uniforme.
        También detecta objetos blancos/claros.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Estrategia 1: Detectar píxeles con saturación > 40 Y valor > 80 (excluir fondo oscuro)
        # Aumentado de 30/60 a 40/80 para reducir ruido del tapete
        mask1 = cv2.inRange(hsv[:, :, 1], 40, 255)  # Saturación
        mask_value = cv2.inRange(hsv[:, :, 2], 80, 255)  # Valor (brillo)
        mask = cv2.bitwise_and(mask1, mask_value)
        
        
        # Estrategia 2: También detectar objetos metálicos (baja saturación pero alto brillo)
        mask_metallic = cv2.inRange(hsv[:, :, 1], 0, 50)  # Baja saturación
        mask_bright = cv2.inRange(hsv[:, :, 2], 120, 255)  # Alto brillo
        mask_metal = cv2.bitwise_and(mask_metallic, mask_bright)
        
        # Estrategia 3: Detectar objetos blancos/claros (botellas blancas, cartón blanco)
        # Alto valor (brillo) pero baja saturación
        mask_white = cv2.inRange(hsv[:, :, 2], 180, 255)  # Muy brillante
        mask_low_sat = cv2.inRange(hsv[:, :, 1], 0, 60)  # Baja saturación
        mask_white_obj = cv2.bitwise_and(mask_white, mask_low_sat)
        
        # Combinar todas las estrategias
        mask = cv2.bitwise_or(mask, mask_metal)
        mask = cv2.bitwise_or(mask, mask_white_obj)
        
        # Excluir regiones muy oscuras y uniformes (fondo típico)
        # Detectar áreas con valor muy bajo (oscuras)
        very_dark = cv2.inRange(hsv[:, :, 2], 0, 50) # Aumentado de 40 a 50
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(very_dark))
        
        return mask
    
    def _edge_detection(self, gray: np.ndarray) -> np.ndarray:
        """
        Detección de bordes con Canny.
        """
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        
        # Dilatar para conectar bordes cercanos
        kernel = np.ones((3, 3), np.uint8)
        # Reducido de 2 a 1 para evitar unir objetos muy cercanos
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        return edges_dilated

    def _adaptive_threshold(self, gray: np.ndarray) -> np.ndarray:
        """
        Umbral adaptativo para detectar objetos en diferentes condiciones de iluminación.
        """
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Usar THRESH_BINARY_INV para que el fondo uniforme sea negro
        # (pixel > mean - C) -> 0 (negro)
        adaptive = cv2.adaptiveThreshold(
            blurred, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 
            51, 5
        )
        
        # Verificar si la máscara está invertida (fondo blanco)
        # A veces con iluminación extraña puede invertirse
        h, w = adaptive.shape
        borders = np.concatenate([
            adaptive[0, :], adaptive[h-1, :],
            adaptive[:, 0], adaptive[:, w-1]
        ])
        white_pixels = np.sum(borders == 255)
        if white_pixels > len(borders) * 0.5:
            adaptive = cv2.bitwise_not(adaptive)
            
        return adaptive
    
    def _morphological_cleanup(self, binary: np.ndarray) -> np.ndarray:
        """
        Limpieza morfológica para eliminar ruido y rellenar huecos.
        """
        kernel = np.ones((3, 3), np.uint8)
        
        # Opening para eliminar ruido pequeño
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Closing para rellenar huecos pequeños
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        return cleaned
    
    def visualize_segmentation(self, 
                              image: np.ndarray,
                              contours: List[np.ndarray],
                              debug_info: dict = None) -> np.ndarray:
        """
        Crea una visualización de la segmentación.
        
        Args:
            image: Imagen original
            contours: Contornos detectados
            debug_info: Información de debug opcional
            
        Returns:
            Imagen con visualización
        """
        vis = image.copy()
        
        # Dibujar contornos
        cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
        
        # Dibujar bounding boxes y numeración
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Añadir número
            cv2.putText(vis, f"#{i+1}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Añadir información de texto
        if debug_info:
            text = f"Objects detected: {debug_info.get('num_contours_valid', len(contours))}"
            cv2.putText(vis, text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return vis
    
    def create_debug_visualization(self, debug_info: dict) -> np.ndarray:
        """
        Crea una visualización de todas las etapas de segmentación.
        
        Args:
            debug_info: Diccionario con imágenes de debug
            
        Returns:
            Imagen con comparación lado a lado
        """
        images_to_show = []
        labels = []
        
        # Seleccionar imágenes clave para mostrar
        if 'color_mask' in debug_info:
            images_to_show.append(debug_info['color_mask'])
            labels.append('Color')
        
        if 'edge_mask' in debug_info:
            images_to_show.append(debug_info['edge_mask'])
            labels.append('Edges')
        
        if 'adaptive_mask' in debug_info:
            images_to_show.append(debug_info['adaptive_mask'])
            labels.append('Adaptive')
        
        if 'combined_mask' in debug_info:
            images_to_show.append(debug_info['combined_mask'])
            labels.append('Combined')
        
        if 'final_mask' in debug_info:
            images_to_show.append(debug_info['final_mask'])
            labels.append('Final')
        
        # Convertir a BGR para visualización
        bgr_images = []
        for img, label in zip(images_to_show, labels):
            if len(img.shape) == 2:
                bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                bgr = img.copy()
            
            # Añadir etiqueta
            cv2.putText(bgr, label, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            bgr_images.append(bgr)
        
        # Concatenar horizontalmente
        if bgr_images:
            # Redimensionar todas las imágenes al mismo tamaño
            h = bgr_images[0].shape[0]
            w = bgr_images[0].shape[1]
            
            resized = []
            for img in bgr_images:
                if img.shape[:2] != (h, w):
                    img = cv2.resize(img, (w, h))
                resized.append(img)
            
            # Concatenar
            vis = np.hstack(resized)
            return vis
        
        return np.zeros((100, 100, 3), dtype=np.uint8)
