#!/usr/bin/env python3
"""
Clasificador de Residuos - Script Principal

Detecta y clasifica objetos (latas, botellas, cartón) en imágenes.
Excluye marcadores ArUco y cajas usando ROI.

Uso:
    python clasificador_main.py --input <imagen> --output <salida> [opciones]
    python clasificador_main.py --batch <carpeta> --output <carpeta_salida>
"""

import cv2
import numpy as np
import time
import argparse
import os
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

# Handle both module and standalone script execution
import sys
_THIS_DIR = os.path.dirname(__file__)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

try:
    from .feature_extractor import FeatureExtractor
    from .roi_detector import ROIDetector
    from .object_segmenter import ObjectSegmenter
    from .waste_classifier import WasteClassifier
    from .ml_waste_classifier import MLWasteClassifier
    from .stabilizer import WasteStabilizer
    from .camera_calibration import CameraCalibration
except ImportError:
    from feature_extractor import FeatureExtractor
    from roi_detector import ROIDetector
    from object_segmenter import ObjectSegmenter
    from waste_classifier import WasteClassifier
    from ml_waste_classifier import MLWasteClassifier
    from stabilizer import WasteStabilizer
    from camera_calibration import CameraCalibration



class WasteClassificationSystem:
    """Sistema completo de clasificación de residuos."""
    
    def __init__(self, 
                 min_area: int = 1000,
                 roi_margin: int = 10,
                 confidence_threshold: float = 0.4,
                 detect_aruco: bool = True,
                 detect_box: bool = True,
                 use_ml: bool = False,
                 filter_class: str = None,
                 board_margin: int = 0,
                 calibration_file: str = None):
        """
        Args:
            min_area: Área mínima para detectar objetos
            roi_margin: Margen alrededor de áreas excluidas
            confidence_threshold: Umbral de confianza para clasificación
            detect_aruco: Si True, detecta y excluye ArUco markers
            detect_box: Si True, detecta y excluye caja de la izquierda
            use_ml: Si True, usa el clasificador ML
            filter_class: Filtrar por tipo ('botella', 'carton', 'lata', None=todos)
        """
        self.feature_extractor = FeatureExtractor()
        self.roi_detector = ROIDetector(margin=roi_margin, board_inner_margin=board_margin)
        self.segmenter = ObjectSegmenter(min_area=min_area)

        self.calibration = None
        if calibration_file is not None:
            ext = os.path.splitext(calibration_file)[1].lower()
            try:
                if ext in ['.yaml', '.yml']:
                    self.calibration = CameraCalibration.from_yaml(calibration_file)
                elif ext == '.txt':
                    self.calibration = CameraCalibration.from_ost_txt(calibration_file)
                elif ext == '.pkl':
                    self.calibration = CameraCalibration.from_pickle(calibration_file)
            except Exception as e:
                self.calibration = None
        
        if use_ml:
            self.classifier = MLWasteClassifier(confidence_threshold=confidence_threshold)
            self.rule_classifier = WasteClassifier(confidence_threshold=max(0.3, confidence_threshold - 0.1))
        else:
            self.classifier = WasteClassifier(confidence_threshold=confidence_threshold)
            self.rule_classifier = None
        
        self.detect_aruco = detect_aruco
        self.detect_box = detect_box
        
        # Mapeo de filtro de clase (normalizar a mayúsculas)
        self.filter_class = filter_class
        self.class_map = {
            'botella': 'botella',
            'carton': 'carton', 
            'lata': 'lata'
        }
        self.filter_class_name = self.class_map.get(filter_class.lower()) if filter_class else None
        
        if self.filter_class_name:
            print(f"Filtrando solo objetos de tipo: {self.filter_class_name}")

    def _adapt_calibration(self, target_w: int, target_h: int):
        """
        Adapta la matriz de calibración (K) a una nueva resolución.
        Útil si calibramos en 640x480 pero capturamos en 1280x720.
        """
        if self.calibration is None:
            return

        current_w, current_h = self.calibration.image_size
        
        # Si ya coincide, no hacer nada
        if current_w == target_w and current_h == target_h:
            return

        print(f"Adaptando calibración de {current_w}x{current_h} a {target_w}x{target_h}...")
        
        # ASUNCIÓN: La cámara mantiene el FOV horizontal y recorta verticalmente para 16:9.
        # Por tanto, usamos el factor de escala horizontal para ambos ejes (pixels cuadrados).
        scale = target_w / current_w
        
        # Escalar matriz K
        # K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        self.calibration.K[0, 0] *= scale # fx
        self.calibration.K[1, 1] *= scale # fy
        
        # Recentrar punto principal
        self.calibration.K[0, 2] = target_w / 2.0 # cx
        self.calibration.K[1, 2] = target_h / 2.0 # cy
        
        # Actualizar tamaño en objeto calibración
        self.calibration.image_size = (target_w, target_h)
        
        # Forzar recalculo de mapas
        self.calibration.map1 = None
        self.calibration.map2 = None
        print(f"Calibración adaptada (scale={scale:.2f}).")
    
    def process_image(self, 
                     image_path: str,
                     output_path: str = None,
                     show_roi: bool = False,
                     show_debug: bool = False,
                     verbose: bool = True) -> dict:
        """
        Procesa una imagen completa: detecta, segmenta y clasifica objetos.
        
        Args:
            image_path: Ruta de la imagen de entrada
            output_path: Ruta de la imagen de salida (opcional)
            show_roi: Si True, visualiza el ROI
            show_debug: Si True, guarda imágenes de debug
            verbose: Si True, imprime información detallada
            
        Returns:
            Diccionario con resultados de clasificación
        """
        # Cargar imagen
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        # 0. Calibrar y Escalar
        image = self._maybe_undistort(image)
        
        # 1. Crear ROI mask
        roi_mask, roi_info = self.roi_detector.create_roi_mask(
            image, 
            detect_aruco=self.detect_aruco,
            detect_box=self.detect_box
        )
        
        # 2. Segmentar objetos
        contours, seg_info = self.segmenter.segment_objects(
            image, 
            roi_mask=roi_mask,
            debug=show_debug
        )
        
        # 3. Extraer características y clasificar cada objeto
        results = []
        for i, contour in enumerate(contours):
            # Calcular centro
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                x, y, w, h = cv2.boundingRect(contour)
                cx = x + w // 2
                cy = y + h // 2
            
            # Extraer características
            features = self.feature_extractor.extract_features(image, contour)
            
            # Clasificar
            class_name, confidence, scores = self.classifier.classify(features)
            if class_name == 'DESCONOCIDO' and self.rule_classifier is not None:
                rb_class, rb_conf, rb_scores = self.rule_classifier.classify(features)
                if rb_class != 'DESCONOCIDO' and rb_conf >= 0.3:
                    class_name, confidence, scores = rb_class, rb_conf, rb_scores
            
            # Filtrar por clase si está especificado
            if self.filter_class_name and class_name != self.filter_class_name:
                continue
            
            # Guardar resultado
            result = {
                'id': i + 1,
                'contour': contour,
                'features': features,
                'class': class_name,
                'confidence': confidence,
                'scores': scores,
                'center': (cx, cy),
            }
            results.append(result)
        
        # 4. Crear visualización
        vis_image = self._create_visualization(image, results)
        
        # 5. Guardar resultados
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            cv2.imwrite(output_path, vis_image)
        
        return {
            'image_path': image_path,
            'output_path': output_path,
            'num_objects': len(results),
            'results': results,
            'roi_info': roi_info,
            'visualization': vis_image
        }
    
    def _create_visualization(self, image: np.ndarray, results: List[dict]) -> np.ndarray:
        """Crea visualización con bounding boxes, etiquetas y centro de cada objeto."""
        vis = image.copy()
        
        for result in results:
            contour = result['contour']
            class_name = result['class']
            confidence = result['confidence']
            obj_id = result['id']
            
            # Color según la clase
            color = self.classifier.get_class_color(class_name)
            
            # Dibujar contorno
            cv2.drawContours(vis, [contour], -1, color, 2)
            
            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
            
            # Obtener centro (si viene en results, si no, recalcular)
            if 'center' in result:
                cx, cy = result['center']
            else:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx = x + w // 2
                    cy = y + h // 2
            
            # Punto rojo en el centro (BGR: (0, 0, 255))
            cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
            
            # Etiqueta
            label = f"#{obj_id} {class_name} ({confidence:.0%})"
            
            # Fondo para el texto
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis, (x, y-text_h-10), (x+text_w, y), color, -1)
            
            # Texto
            cv2.putText(vis, label, (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Contador total
        total_text = f"Total objetos: {len(results)}"
        cv2.putText(vis, total_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        return vis
    
    def process_batch(self, 
                     input_dir: str,
                     output_dir: str,
                     pattern: str = "*.png",
                     **kwargs) -> dict:
        """
        Procesa un lote de imágenes.
        
        Args:
            input_dir: Directorio de entrada
            output_dir: Directorio de salida
            pattern: Patrón de archivos (ej: "*.png", "*.jpg")
            **kwargs: Argumentos adicionales para process_image
            
        Returns:
            Diccionario con estadísticas del lote
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Encontrar todas las imágenes
        image_files = list(input_path.glob(pattern))
        
        print(f"\n{'='*60}")
        print(f"PROCESAMIENTO POR LOTES")
        print(f"Directorio de entrada: {input_dir}")
        print(f"Directorio de salida: {output_dir}")
        print(f"Imágenes encontradas: {len(image_files)}")
        print(f"{'='*60}\n")
        
        all_results = []
        class_totals = {}
        
        for i, img_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Procesando: {img_file.name}")
            
            output_file = output_path / img_file.name
            
            try:
                result = self.process_image(
                    str(img_file),
                    str(output_file),
                    verbose=False,
                    **kwargs
                )
                
                all_results.append(result)
                
                # Actualizar totales
                for obj_result in result['results']:
                    cls = obj_result['class']
                    class_totals[cls] = class_totals.get(cls, 0) + 1
                
                print(f"Objetos detectados: {result['num_objects']}")
                
            except Exception as e:
                print(f"Error: {e}")
        
        # Resumen final
        print(f"\n{'='*60}")
        print("RESUMEN DEL LOTE:")
        print(f"  Imágenes procesadas: {len(all_results)}/{len(image_files)}")
        print(f"  Total de objetos: {sum(class_totals.values())}")
        print(f"\n  Distribución por clase:")
        for cls, count in sorted(class_totals.items()):
            print(f"    {cls}: {count}")
        print(f"{'='*60}\n")
        
        return {
            'num_images': len(image_files),
            'num_processed': len(all_results),
            'class_totals': class_totals,
            'results': all_results
        }

    def process_realtime(self, camera_id: int = 0, record: bool = False):
        """
        Ejecuta la clasificación en tiempo real continua.
        Detecta, segmenta y clasifica en cada frame.
        """
        cap = None
        print(f"Intentando abrir cámara {camera_id} para tiempo real...")
        
        # Intentar diferentes backends como en process_capture
        for backend in (cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF):
            temp_cap = cv2.VideoCapture(camera_id, backend)
            temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if temp_cap.isOpened():
                cap = temp_cap
                break
        
        if cap is None:
            cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"Error: No se pudo abrir la cámara {camera_id}")
            return

        # Verificar resolución real obtenida
        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Resolución de cámara configurada: {int(actual_w)}x{int(actual_h)}")

        '''if int(actual_w) != 1280 or int(actual_h) != 720:
            print("ADVERTENCIA: La cámara no está capturando en 1280x720 nativo.")
            print("La imagen será reescalada/procesada a esta resolución.")'''

        # Adaptar calibración a la resolución REAL
        #self._adapt_calibration(int(actual_w), int(actual_h))

        print(f"\n{'='*60}")
        print("INICIANDO MODO TIEMPO REAL (CONTINUO)")
        print("  [q] - Salir")
        if record:
            print("  [GRABANDO VIDEO]")
        print(f"{'='*60}\n")

        # Inicializar estabilizador
        stabilizer = WasteStabilizer(history_size=10, max_distance=50)
        
        # Inicializar grabador de video
        video_writer = None
        if record:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"video_realtime_{timestamp}.avi"
            # Usar MJPG para compatibilidad
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            # Asumimos 30 fps, ajustar si es necesario
            video_writer = cv2.VideoWriter(video_filename, fourcc, 30.0, (int(actual_w), int(actual_h)))
            print(f"Grabando video en: {video_filename}")

        try:
            while True:
                # Leer frame de la cámara
                ret, frame = cap.read()
                if not ret:
                    print("Error al leer frame de la cámara")
                    break

                # 0. Calibrar y Escalar
                # Primero quitamos distorsión
                frame = self._maybe_undistort(frame)
                
                # Luego aseguramos 1280x720
                #frame = cv2.resize(frame, (1280, 720))

                # 1. Detectar ROI
                roi_mask, roi_info = self.roi_detector.create_roi_mask(
                    frame,
                    detect_aruco=self.detect_aruco,
                    detect_box=self.detect_box
                )

                # 2. Segmentar objetos
                contours, seg_info = self.segmenter.segment_objects(
                    frame,
                    roi_mask=roi_mask,
                    debug=False
                )

                # 3. Clasificar objetos detectados
                results = []
                for i, contour in enumerate(contours):
                    # Calcular centro
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                    else:
                        x, y, w, h = cv2.boundingRect(contour)
                        cx = x + w // 2
                        cy = y + h // 2

                    # Extraer características
                    features = self.feature_extractor.extract_features(frame, contour)
                    
                    # Clasificar (ML primero, luego fallback por reglas)
                    class_name, confidence, scores = self.classifier.classify(features)
                    if class_name == 'DESCONOCIDO' and self.rule_classifier is not None:
                        rb_class, rb_conf, rb_scores = self.rule_classifier.classify(features)
                        if rb_class != 'DESCONOCIDO' and rb_conf >= 0.3:
                            class_name, confidence, scores = rb_class, rb_conf, rb_scores
                    
                    # Filtrar por clase si está especificado
                    if self.filter_class_name and class_name != self.filter_class_name:
                        continue  # Saltar este objeto si no coincide con el filtro
                    
                    results.append({
                        'id': i + 1,
                        'contour': contour,
                        'class': class_name,
                        'confidence': confidence,
                        'center': (cx, cy)
                    })

                # 4. Estabilizar resultados
                stabilized_results = stabilizer.update(results)

                # 5. Visualizar
                # Primero dibujamos el ROI
                vis_frame = self.roi_detector.visualize_roi(frame, roi_mask, roi_info)
                
                # Luego dibujamos los objetos detectados (usando resultados estabilizados)
                final_vis = self._create_visualization(vis_frame, stabilized_results)

                # Grabar frame si está activado
                if video_writer is not None:
                    video_writer.write(final_vis)

                cv2.namedWindow('Clasificador de Residuos - Tiempo Real', cv2.WINDOW_NORMAL)
                cv2.imshow('Clasificador de Residuos - Tiempo Real', final_vis)

                # Control de teclado
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
                print("Video guardado y cerrado.")
            cv2.destroyAllWindows()
    
    def process_capture(self, camera_id: int = 0):
            cap = None
            print(f"Intentando abrir cámara {camera_id}...")
            for backend in (cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF):
                temp_cap = cv2.VideoCapture(camera_id, backend)
                temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) #1280×720
                temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                if temp_cap.isOpened():
                    cap = temp_cap
                    break
            
            if cap is None:
                cap = cv2.VideoCapture(camera_id)

            if not cap.isOpened():
                print(f"Error: No se pudo abrir la cámara {camera_id}")
                return None
            
            # Verificar resolución real obtenida
            actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"Resolución de captura obtenida: {int(actual_w)}x{int(actual_h)}")
            
            '''if int(actual_w) != 1280 or int(actual_h) != 720: #1280×720
                print("ADVERTENCIA: La cámara no está capturando en 1280x720 nativo.")
                print("La imagen será reescalada, pero la calidad puede verse afectada.")'''

            # Adaptar calibración si es necesario
            #self._adapt_calibration(1280, 720)
            #self._adapt_calibration(int(actual_w), int(actual_h))


            frame = None
            try:
                print(f"Estabilizando sensor cámara {camera_id}...")
                for _ in range(20):
                    cap.read()
                    time.sleep(0.01)
                
                ret, frame = cap.read()
                
                if not ret:
                    print("Error al leer frame de la cámara")
                    return None
                
                print("¡Foto capturada correctamente!")

            finally:
                # Liberar cámara inmediatamente (lógica de "sacar foto y cerrar")
                cap.release()
                cv2.destroyAllWindows()

            # --- 2. LÓGICA DE PROCESAMIENTO ---
            print("Procesando captura...")
            
            # 0. Calibrar y Escalar
            # Primero quitamos distorsión
            frame = self._maybe_undistort(frame)
            
            # Luego aseguramos 1280x720
            #frame = cv2.resize(frame, (1280, 720))
            
            # Detectar ROI
            roi_mask, roi_info = self.roi_detector.create_roi_mask(
                frame,
                detect_aruco=self.detect_aruco,
                detect_box=self.detect_box
            )
            
            # Segmentar objetos
            contours, seg_info = self.segmenter.segment_objects(
                frame,
                roi_mask=roi_mask,
                debug=False
            )
            
            # Clasificar y obtener coordenadas
            results = []
            for i, contour in enumerate(contours):
                # Calcular centro
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    x, y, w, h = cv2.boundingRect(contour)
                    cx = x + w // 2
                    cy = y + h // 2
                
                # Extraer características
                features = self.feature_extractor.extract_features(frame, contour)
                
                # Clasificar
                class_name, confidence, scores = self.classifier.classify(features)
                
                # Filtrar por clase si está especificado
                if self.filter_class_name and class_name != self.filter_class_name:
                    continue
                
                results.append({
                    'id': i + 1,
                    'class': class_name,
                    'confidence': confidence,
                    'center': (cx, cy),
                    'contour': contour
                })
            
            # Mostrar resultados en consola
            print(f"\n{'='*60}")
            print(f"OBJETOS DETECTADOS: {len(results)}")
            print(f"{'='*60}")
            
            for obj in results:
                cx, cy = obj['center']
                print(f"\nObjeto #{obj['id']}:")
                print(f"  Clase: {obj['class']}")
                print(f"  Confianza: {obj['confidence']:.0%}")
                print(f"  Coordenadas (x, y): ({cx}, {cy})")
            
            print(f"\n{'='*60}\n")
            
            # Crear visualización final
            vis_final = self._create_visualization(frame, results)
            
            filename = "imagenprocesada.png"
            cv2.imwrite(filename, vis_final)
            print(f"Captura procesada guardada en: {filename}")
            
            cv2.imshow('Captura - Resultado', vis_final)
            print("Presiona cualquier tecla para continuar...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            return results

    def _maybe_undistort(self, image: np.ndarray) -> np.ndarray:
        """
        Si hay calibración cargada, devuelve la imagen undistort,
        si no, devuelve la imagen original.
        """
        if self.calibration is None:
            return image
        return self.calibration.undistort(image)
def main():
    """Función principal con interfaz de línea de comandos."""
    parser = argparse.ArgumentParser(
        description='Clasificador de Residuos - Detecta y clasifica latas, botellas y cartón'
    )
    
    # Argumentos de entrada/salida
    parser.add_argument('--input', '-i', type=str,
                       help='Ruta de la imagen de entrada')
    parser.add_argument('--output', '-o', type=str,
                       help='Ruta de la imagen de salida')
    parser.add_argument('--batch', '-b', type=str,
                       help='Procesar todas las imágenes en un directorio')
    parser.add_argument('--pattern', type=str, default='*.png',
                       help='Patrón de archivos para modo batch (default: *.png)')
    parser.add_argument('--realtime', action='store_true',
                       help='Ejecutar en modo tiempo real con webcam')
    parser.add_argument('--capture', action='store_true',
                       help='Modo captura: cuenta atrás y captura una imagen')
    parser.add_argument('--countdown', type=int, default=3,
                       help='Segundos de cuenta atrás antes de capturar (default: 3)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Índice de la cámara para modo tiempo real/captura (default: 0)')
    parser.add_argument('--record', action='store_true',
                       help='Grabar video de la sesión en tiempo real')
    
    # Opciones de visualización
    parser.add_argument('--show-roi', action='store_true',
                       help='Mostrar visualización del ROI')
    parser.add_argument('--show-debug', action='store_true',
                       help='Guardar imágenes de debug de segmentación')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Modo silencioso (menos output)')
    
    
    # Opciones de configuración
    parser.add_argument('--min-area', type=int, default=500,
                       help='Área mínima para detectar objetos (default: 500)')
    parser.add_argument('--roi-margin', type=int, default=10,
                       help='Margen alrededor de áreas excluidas (default: 10)')
    parser.add_argument('--board-margin', type=int, default=0,
                       help='Margen interior del tablero negro (default: 0)')
    parser.add_argument('--confidence', type=float, default=0.35,
                       help='Umbral de confianza (default: 0.35)')
    parser.add_argument('--no-aruco', action='store_true',
                       help='No detectar marcadores ArUco')
    parser.add_argument('--no-box', action='store_true',
                       help='No detectar caja de la izquierda')
    parser.add_argument('--ml', action='store_true',
                       help='Usar clasificador ML (requiere haber ejecutado train_classifier.py)')
    parser.add_argument('--filter-class', type=str, choices=['botella', 'carton', 'lata'],
                       help='Filtrar por tipo de objeto: botella, carton, lata')
    
    parser.add_argument('--calib', type=str,
                       help='Fichero de calibración de cámara (ost.yaml, ost.txt o calibracion_camara.pkl)')


    args = parser.parse_args()
    
    # Validar argumentos
    if not args.input and not args.batch and not args.realtime and not args.capture:
        parser.error('Debe especificar --input, --batch, --realtime o --capture')
    
    if args.input and not args.output:
        # Generar nombre de salida automático
        args.output = args.input.replace('.png', '_classified.png')
    
    # Crear sistema de clasificación
    system = WasteClassificationSystem(
        min_area=args.min_area,
        roi_margin=args.roi_margin,
        confidence_threshold=args.confidence,
        detect_aruco=not args.no_aruco,
        detect_box=not args.no_box,
        use_ml=args.ml,
        filter_class=args.filter_class,
        board_margin=args.board_margin,
        calibration_file=args.calib
    )
    
    # Procesar
    if args.capture:
        # Modo captura con cuenta atrás
        results = system.process_capture(camera_id=args.camera)
        if results:
            print("\nCaptura completada exitosamente")
    elif args.realtime:
        # Modo tiempo real
        system.process_realtime(camera_id=args.camera, record=args.record)
    elif args.batch:
        # Modo batch
        system.process_batch(
            args.batch,
            args.output,
            pattern=args.pattern,
            show_roi=args.show_roi,
            show_debug=args.show_debug
        )
    else:
        # Modo single image
        result = system.process_image(
            args.input,
            args.output,
            show_roi=args.show_roi,
            show_debug=args.show_debug,
            verbose=not args.quiet
        )

        # Ejemplo: imprimir centros detectados
        if not args.quiet:
            for obj in result['results']:
                print(f"Objeto #{obj['id']} -> clase: {obj['class']}, centro: {obj['center']}")
                

if __name__ == '__main__':
    main()
