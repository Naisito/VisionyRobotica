import cv2
import time
import os
import argparse
import json
import sys

# Handle both direct execution and module import
try:
    from clasificador_main import WasteClassificationSystem
except ImportError:
    # If run as standalone script, add parent to path
    sys.path.insert(0, os.path.dirname(__file__))
    from clasificador_main import WasteClassificationSystem

def main():
    # 1. Configuración de Argumentos
    parser = argparse.ArgumentParser(description='Captura y clasifica residuos.')
    parser.add_argument('--filter-class', type=str, choices=['plastico', 'carton', 'lata'],
                       help='Filtrar por tipo de objeto: plastico, carton, lata')
    parser.add_argument('--image', type=str, help='Ruta de la imagen a cargar (omite captura de cámara)')
    parser.add_argument('--json-results', type=str, help='Ruta para exportar resultados en JSON')
    args = parser.parse_args()

    CAMERA_ID = 0
    RES_W, RES_H = 1280, 720
    CALIB_FILE = "calibracion_camara_cenital.pkl"
    OUTPUT_ORIGINAL = "captura_original.png"
    OUTPUT_PROCESSED = "captura_procesada.png"

    print("Iniciando sistema de captura...")
    if args.filter_class:
        print(f"Filtro activado: Solo se mostrarán objetos de tipo '{args.filter_class}'")

    # 2. Inicializar Sistema de Clasificación (Carga calibración internamente)
    # Si cargamos imagen externa, NO aplicamos calibración (asumimos que ya está o no la necesita)
    calib_to_use = None if args.image else CALIB_FILE
    
    try:
        system = WasteClassificationSystem(
            calibration_file=calib_to_use,
            min_area=1000,
            confidence_threshold=0.4,
            filter_class=args.filter_class,
            use_ml=True
        )
    except Exception as e:
        print(f"Error al inicializar el sistema: {e}")
        return

    # 3. Determinar fuente de imagen
    input_image_path = OUTPUT_ORIGINAL

    if args.image:
        # Modo Carga de Imagen
        if not os.path.exists(args.image):
            print(f"Error: La imagen {args.image} no existe.")
            return
        input_image_path = args.image
        print(f"Cargando imagen desde: {input_image_path}")
    else:
        # Modo Captura de Cámara
        cap = None
        frame = None
        
        try:
            # Intentar abrir cámara con backend preferido para Windows
            for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
                cap = cv2.VideoCapture(CAMERA_ID, backend)
                if cap.isOpened():
                    break
            
            if not cap or not cap.isOpened():
                print(f"Error: No se pudo abrir la cámara {CAMERA_ID}")
                return

            # Configurar resolución
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, RES_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RES_H)
            
            # Estabilizar sensor (darle tiempo para ajuste de luz/foco)
            print("Estabilizando cámara...")
            for _ in range(30):
                cap.read()
                time.sleep(0.05)

            # Capturar frame final
            ret, frame = cap.read()
            if not ret:
                print("Error al capturar frame.")
                return

            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Captura realizada a: {actual_w}x{actual_h}")

        finally:
            if cap:
                cap.release()

        # 4. Guardar Imagen Original
        cv2.imwrite(OUTPUT_ORIGINAL, frame)
        print(f"Imagen original guardada en: {OUTPUT_ORIGINAL}")

    # 5. Procesar Imagen
    print("Procesando imagen...")
    results_dict = system.process_image(
        image_path=input_image_path,
        output_path=OUTPUT_PROCESSED,
        verbose=True
    )

    # 6. Mostrar Coordenadas
    print("\n" + "="*40)
    print("RESULTADOS Y COORDENADAS")
    print("="*40)
    
    objects = results_dict.get('results', [])
    if not objects:
        print("No se detectaron objetos (o fueron filtrados).")
    
    coordinates = []
    for obj in objects:
        obj_id = obj['id']
        class_name = obj['class']
        confidence = obj['confidence']
        center = obj['center'] # (x, y)
        # Convert contour to list of [x,y]
        contour = obj.get('contour')
        contour_pts = None
        if contour is not None:
            try:
                pts = contour.reshape(-1, 2)
                contour_pts = [[int(px), int(py)] for (px, py) in pts]
            except Exception:
                contour_pts = None

        coordinates.append({
            "id": obj_id,
            "class": class_name,
            "confidence": float(confidence),
            "center": {"x": int(center[0]), "y": int(center[1])},
            "contour": contour_pts
        })

        print(f"Objeto #{obj_id} ({class_name}):")
        print(f"  - Confianza: {confidence:.2f}")
        print(f"  - Coordenadas Centro (X, Y): {center}")
        print("-" * 20)

    # Export JSON results if requested
    if args.json_results:
        os.makedirs(os.path.dirname(args.json_results) or '.', exist_ok=True)
        payload = {
            "image_path": input_image_path,
            "output_path": OUTPUT_PROCESSED,
            "results": coordinates
        }
        with open(args.json_results, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Resultados JSON guardados en: {args.json_results}")

    print(f"\nProceso finalizado. Imagen procesada en: {OUTPUT_PROCESSED}")
    return coordinates

if __name__ == "__main__":
    main()