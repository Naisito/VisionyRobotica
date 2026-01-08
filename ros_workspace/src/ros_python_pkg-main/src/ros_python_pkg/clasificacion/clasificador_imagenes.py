import cv2
import time
import os
import argparse
import json
import sys

try:
    from clasificador_main import WasteClassificationSystem
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from clasificador_main import WasteClassificationSystem

def main():
    parser = argparse.ArgumentParser(description='Captura y clasifica residuos.')
    parser.add_argument('--filter-class', type=str, choices=['botella', 'carton', 'lata'],
                       help='Filtrar por tipo de objeto: botella, carton, lata')
    parser.add_argument('--image', type=str, help='Ruta de la imagen a cargar')
    parser.add_argument('--json-results', type=str, help='Ruta para exportar resultados en JSON')
    args = parser.parse_args()

    CAMERA_ID = 0
    RES_W, RES_H = 1280, 720
    CALIB_FILE = "calibracion_camara_cenital.pkl"
    OUTPUT_PROCESSED = "captura_procesada.png"

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
        raise

    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: La imagen {args.image} no existe.")
            return
        input_image_path = args.image
    else:
        cap = None
        frame = None
        try:
            for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
                cap = cv2.VideoCapture(CAMERA_ID, backend)
                if cap.isOpened():
                    break
            if not cap or not cap.isOpened():
                print(f"Error: No se pudo abrir la cámara {CAMERA_ID}")
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, RES_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RES_H)
            for _ in range(30):
                cap.read()
                time.sleep(0.05)
            ret, frame = cap.read()
            if not ret:
                print("Error al capturar frame.")
                return
        finally:
            if cap:
                cap.release()
        input_image_path = "captura_original.png"
        cv2.imwrite(input_image_path, frame)

    results_dict = system.process_image(
        image_path=input_image_path,
        output_path=OUTPUT_PROCESSED,
        verbose=False
    )

    objects = results_dict.get('results', [])
    
    if objects:
        print(f"Detectados {len(objects)} objetos:")
        for obj in objects:
            print(f"  #{obj['id']} {obj['class']} (conf: {obj['confidence']:.0%}) centro: {obj['center']}")
    else:
        print("No se detectaron objetos.")
    
    coordinates = []
    for obj in objects:
        contour = obj.get('contour')
        contour_pts = None
        if contour is not None:
            try:
                pts = contour.reshape(-1, 2)
                contour_pts = [[int(px), int(py)] for (px, py) in pts]
            except Exception:
                contour_pts = None
        coordinates.append({
            "id": obj['id'],
            "class": obj['class'],
            "confidence": float(obj['confidence']),
            "center": {"x": int(obj['center'][0]), "y": int(obj['center'][1])},
            "contour": contour_pts
        })

    if args.json_results:
        os.makedirs(os.path.dirname(args.json_results) or '.', exist_ok=True)
        payload = {
            "image_path": input_image_path,
            "output_path": OUTPUT_PROCESSED,
            "results": coordinates
        }
        with open(args.json_results, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    return coordinates

if __name__ == "__main__":
    main()
