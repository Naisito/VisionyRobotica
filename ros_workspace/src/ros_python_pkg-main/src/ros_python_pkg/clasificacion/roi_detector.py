import cv2
import numpy as np
from typing import Tuple, List, Optional


class ROIDetector:
    """
    Genera una máscara ROI válida para detección de objetos:
    - Parte interior de la cartulina negra (recortada con margen)
    - Excluyendo caja de la izquierda
    - Excluyendo marcadores ArUco
    """
    
    def __init__(self, margin: int = 10, board_inner_margin: int = 0):
        """
        Args:
            margin: Margen adicional alrededor de áreas excluidas (píxeles)
            board_inner_margin: Cuánto recortar hacia dentro el tablero negro
                                por cada lado, en píxeles. Controla la zona naranja.
        """
        self.margin = margin
        self.board_inner_margin = board_inner_margin
        
        # Inicializar detector de ArUco - probar múltiples diccionarios
        self.aruco_dicts = [
            cv2.aruco.DICT_4X4_50,
            cv2.aruco.DICT_4X4_100,
            cv2.aruco.DICT_4X4_250,
            cv2.aruco.DICT_5X5_50,
            cv2.aruco.DICT_6X6_50,
            cv2.aruco.DICT_6X6_250
        ]
    
    def create_roi_mask(self, image: np.ndarray, 
                        detect_aruco: bool = True,
                        detect_box: bool = True,
                        detect_board: bool = True,
                        debug: bool = False) -> Tuple[np.ndarray, dict]:
        """
        Crea una máscara ROI excluyendo áreas no deseadas.
        
        Args:
            image: Imagen de entrada en BGR
            detect_aruco: Si True, detecta y excluye marcadores ArUco
            detect_box: Si True, detecta y excluye la caja de la izquierda
            detect_board: Si True, detecta el tablero negro (Otsu) y excluye lo de fuera
            debug: Si True, imprime avisos
        
        Returns:
            Tupla (mask, info) donde:
            - mask: Máscara binaria (255 = ROI válido, 0 = excluido)
            - info: Diccionario con información de detección
        """
        h, w = image.shape[:2]
        
        # Inicializar máscara
        if detect_board:
            mask = np.zeros((h, w), dtype=np.uint8)
        else:
            mask = np.ones((h, w), dtype=np.uint8) * 255
        
        info = {
            'aruco_markers': [],
            'box_region': None,
            'board_contour': None,
            'inner_board_rect': None,
            'excluded_areas': 0
        }
        
        # 1. Detectar tablero (zona válida base) usando Otsu
        if detect_board:
            board_cnt = self._detect_background_board(image)
            if board_cnt is not None:
                # Dibujar el tablero detectado
                cv2.drawContours(mask, [board_cnt], -1, 255, -1)
                info['board_contour'] = board_cnt

                # --- NUEVO: recortar a un rectángulo interior (zona naranja) ---
                # --- NUEVO: recortar a un rectángulo interior (zona naranja) ---
                # Siempre calculamos esto para visualizar el ROI
                x, y, w_board, h_board = cv2.boundingRect(board_cnt)

                # coordenadas del rectángulo interior
                # Usamos el mismo margen para todos los lados
                bottom_margin = self.board_inner_margin 
                x_in = max(0, x + self.board_inner_margin)
                y_in = max(0, y + self.board_inner_margin)
                w_in = max(0, w_board - 2 * self.board_inner_margin)
                h_in = max(0, h_board - self.board_inner_margin - bottom_margin)

                # máscara con solo el rectángulo interior
                inner_mask = np.zeros_like(mask)
                cv2.rectangle(inner_mask,
                              (x_in, y_in),
                              (x_in + w_in, y_in + h_in),
                              255, -1)

                # ROI final del tablero = intersección tablero ∩ rectángulo interior
                mask = cv2.bitwise_and(mask, inner_mask)

                info['inner_board_rect'] = (x_in, y_in, w_in, h_in)
            else:
                # Fallback
                mask[:] = 255
                if debug:
                    print("WARN: No se detectó tablero, usando imagen completa")

        # 2. Detectar y excluir ArUco markers
        if detect_aruco:
            aruco_markers = self._detect_aruco_markers(image)
            info['aruco_markers'] = aruco_markers
            
            for corners in aruco_markers:
                # Expandir el área del marcador con margen
                expanded_corners = self._expand_polygon(corners, self.margin)
                cv2.fillPoly(mask, [expanded_corners], 0)
                info['excluded_areas'] += 1

                # --- NUEVO: Excluir todo lo que esté a la DERECHA del ArUco ---
                # Encontrar la coordenada X más a la derecha del marcador (incluyendo margen)
                max_x = np.max(expanded_corners[:, 0])
                
                # Dibujar rectángulo negro desde max_x hasta el final de la imagen
                if max_x < w:
                    cv2.rectangle(mask, 
                                 (max_x, 0), 
                                 (w, h), 
                                 0, -1)
                    info['excluded_areas'] += 1

            # --- NUEVO: Excluir arriba y abajo de los ArUcos ---
            if len(aruco_markers) > 0:
                # Recopilar todos los puntos de todos los marcadores
                all_points = np.vstack(aruco_markers).reshape(-1, 2)
                
                # Encontrar límites verticales
                min_y = np.min(all_points[:, 1])
                max_y = np.max(all_points[:, 1])
                
                # Excluir zona SUPERIOR (desde 0 hasta min_y)
                if min_y > 0:
                    cv2.rectangle(mask, (0, 0), (w, int(min_y)), 0, -1)
                    info['excluded_areas'] += 1
                    
                # Excluir zona INFERIOR (desde max_y hasta h)
                if max_y < h:
                    cv2.rectangle(mask, (0, int(max_y)), (w, h), 0, -1)
                    info['excluded_areas'] += 1
        
        # 3. Detectar y excluir caja de la izquierda
        if detect_box:
            #box_region = self._detect_left_box(image)
            box_region = None # Desactivado por petición del usuario
            if box_region is not None:
                x, y, w_box, h_box = box_region
                info['box_region'] = box_region

                # Desplazar un poco la zona de la caja hacia la izquierda
                shift = int(w_box * 0.05)
                x = max(0, x - shift)

                # Guardar la caja para visualización
                info['box_region'] = (x, y, w_box, h_box)

                # Aplicar margen alrededor de la caja
                x_roi = max(0, x - self.margin)
                y_roi = max(0, y - self.margin)
                w_roi = min(w - x_roi, w_box + 3 * self.margin)
                h_roi = min(h - y_roi, h_box + 3 * self.margin)

                mask[y_roi:y_roi + h_roi, x_roi:x_roi + w_roi] = 0

                # Excluir toda la franja vertical donde está la caja (incluyendo arriba y abajo)
                # para que el ROI no empiece hasta después de la caja.
                exclusion_right_x = x + w_box + self.margin
                
                # Dibujar rectángulo negro desde el inicio (0,0) hasta (exclusion_right_x, height)
                cv2.rectangle(mask, 
                             (0, 0), 
                             (exclusion_right_x, h), 
                             0, -1)
                
                info['excluded_areas'] += 1
        
        return mask, info

    def _detect_background_board(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detecta el tablero negro de fondo usando Otsu thresholding en el canal V (HSV),
        y luego ajusta un rectángulo mínimo al contorno para evitar entrantes producidos
        por objetos (botellas, etc.) que toquen el borde.
        """
        # Convertir a HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        
        # Blur para reducir ruido
        blurred = cv2.GaussianBlur(v_channel, (5, 5), 0)
        
        # Otsu: tablero negro => blanco en la binaria (INV)
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        
        # Limpieza morfológica
        kernel = np.ones((7, 7), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Contornos externos
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        # Contorno más grande (tablero)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_cnt = contours[0]

        # Comprobar área mínima (30% de la imagen)
        total_area = image.shape[0] * image.shape[1]
        if cv2.contourArea(largest_cnt) < total_area * 0.3:
            return None

        # quitar entrantes usando convex hull + rectángulo mínimo ---
        hull = cv2.convexHull(largest_cnt)          # elimina concavidades (botella, etc.)
        rect = cv2.minAreaRect(hull)                # rectángulo mínimo que encierra el hull
        box = cv2.boxPoints(rect)                   # 4 puntos del rectángulo
        box = np.int32(box).reshape(-1, 1, 2)       # formato contorno (N,1,2) para drawContours
        
        return box

    
    def _detect_aruco_markers(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detecta marcadores ArUco en la imagen.
        Prueba múltiples diccionarios para asegurar detección.
        
        Returns:
            Lista de esquinas de marcadores detectados
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        all_markers = []
        
        # Probar múltiples diccionarios
        for dict_type in self.aruco_dicts:
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
            aruco_params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
            
            corners, ids, rejected = detector.detectMarkers(gray)
            
            if corners is not None and len(corners) > 0:
                for corner in corners:
                    # corner tiene forma (1, 4, 2), convertir a (4, 2)
                    corner_points = corner.reshape(-1, 2).astype(np.int32)
                    
                    # Evitar duplicados
                    is_duplicate = False
                    for existing in all_markers:
                        if np.allclose(corner_points, existing, atol=5):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        all_markers.append(corner_points)
        
        return all_markers
    
    def _detect_left_box(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detecta la caja de colores en el lado izquierdo de la imagen.

        Estrategia:
        1. Tomar un ROI del lado izquierdo (por ejemplo 40% del ancho).
        2. Buscar píxeles con alta saturación (colores vivos de la caja).
        3. Quedarse con todos los contornos suficientemente grandes.
        4. Unir todos esos contornos en UN solo bounding box (min x, min y, max x, max y).
        Returns:
            Tupla (x, y, w, h) del bounding box de la caja, o None si no se detecta.
        """
        h, w = image.shape[:2]

        # Región de interés: 10% izquierdo de la imagen
        left_region_width = int(w * 0.10)
        roi = image[:, :left_region_width].copy()

        # Convertir a HSV para detección de colores
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Detectar píxeles con alta saturación (colores vivos de la caja)
        # Ajusta estos umbrales si hace falta
        s_channel = hsv[:, :, 1]
        mask_saturated = cv2.inRange(s_channel, 70, 255)

        # Operaciones morfológicas para limpiar
        kernel = np.ones((5, 5), np.uint8)
        mask_saturated = cv2.morphologyEx(mask_saturated, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask_saturated = cv2.morphologyEx(mask_saturated, cv2.MORPH_OPEN, kernel, iterations=2)

        # Encontrar contornos
        contours, _ = cv2.findContours(mask_saturated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Filtrar contornos por área y unirlos en un solo bounding box
        min_area = (h * w) * 0.005   # al menos 0.5% del área total
        xs, ys, xes, yes = [], [], [], []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            x, y, w_box, h_box = cv2.boundingRect(cnt)

            # solo consideramos cosas claramente en el lado izquierdo
            if x + w_box > left_region_width:
                continue

            xs.append(x)
            ys.append(y)
            xes.append(x + w_box)
            yes.append(y + h_box)

        if not xs:
            return None

        # Bounding box combinado de todos los fragmentos de la caja
        x_min = min(xs)
        y_min = min(ys)
        x_max = max(xes)
        y_max = max(yes)

        w_box = x_max - x_min
        h_box = y_max - y_min

        return (x_min, y_min, w_box, h_box)

    
    def _expand_polygon(self, points: np.ndarray, margin: int) -> np.ndarray:
        """
        Expande un polígono añadiendo un margen.
        
        Args:
            points: Puntos del polígono (N, 2)
            margin: Margen en píxeles
            
        Returns:
            Puntos expandidos
        """
        # Calcular centroide
        centroid = np.mean(points, axis=0)
        
        # Expandir cada punto alejándolo del centroide
        expanded = []
        for point in points:
            direction = point - centroid
            distance = np.linalg.norm(direction)
            if distance > 0:
                direction = direction / distance
                new_point = point + direction * margin
                expanded.append(new_point)
            else:
                expanded.append(point)
        
        return np.array(expanded, dtype=np.int32)
    
    def visualize_roi(self, image: np.ndarray, mask: np.ndarray, 
                      info: dict) -> np.ndarray:
        """
        Crea una visualización del ROI y áreas excluidas.
        
        Args:
            image: Imagen original
            mask: Máscara ROI (255 = válido, 0 = excluido)
            info: Información de detección
            
        Returns:
            Imagen con visualización
        """
        vis = image.copy()
        
        # Crear overlay
        overlay = vis.copy()
        
        # 1. Pintar zona válida (mask == 255) de VERDE
        overlay[mask == 255] = [0, 255, 0]
        
        # 2. Pintar zona excluida (mask == 0) de ROJO
        overlay[mask == 0] = [0, 0, 255]
        
        # Mezclar con transparencia
        alpha = 0.3
        vis = cv2.addWeighted(vis, 1 - alpha, overlay, alpha, 0)
        
        # Dibujar contornos de ArUco markers
        for corners in info['aruco_markers']:
            cv2.polylines(vis, [corners], True, (0, 255, 0), 2)
            center = np.mean(corners, axis=0).astype(int)
            cv2.putText(vis, "ArUco", tuple(center), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Dibujar caja
        if info['box_region'] is not None:
            x, y, w_box, h_box = info['box_region']
            cv2.rectangle(vis, (x, y), (x + w_box, y + h_box), (255, 0, 0), 2)
            cv2.putText(vis, "Box", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Dibujar contorno del tablero si existe
        '''if info.get('board_contour') is not None:
            cv2.drawContours(vis, [info['board_contour']], -1, (255, 255, 0), 2)'''

        # Dibujar rectángulo interior (zona naranja deseada)
        if info.get('inner_board_rect') is not None:
            x_in, y_in, w_in, h_in = info['inner_board_rect']
            cv2.rectangle(vis, (x_in, y_in), (x_in + w_in, y_in + h_in),
                          (0, 140, 255), 3)  # naranja
        

        return vis
    
    def filter_contours_by_roi(self, contours: List[np.ndarray], 
                               mask: np.ndarray,
                               min_overlap: float = 0.5) -> List[np.ndarray]:
        """
        Filtra contornos que están fuera del ROI válido.
        
        Args:
            contours: Lista de contornos
            mask: Máscara ROI (255 = válido, 0 = excluido)
            min_overlap: Proporción mínima del contorno que debe estar en ROI
            
        Returns:
            Lista de contornos válidos
        """
        valid_contours = []
        
        for cnt in contours:
            # Crear máscara del contorno
            cnt_mask = np.zeros_like(mask)
            cv2.drawContours(cnt_mask, [cnt], -1, 255, -1)
            
            # Calcular overlap con ROI
            overlap = cv2.bitwise_and(cnt_mask, mask)
            overlap_area = np.sum(overlap > 0)
            cnt_area = np.sum(cnt_mask > 0)
            
            if cnt_area > 0:
                overlap_ratio = overlap_area / cnt_area
                if overlap_ratio >= min_overlap:
                    valid_contours.append(cnt)
        
        return valid_contours
