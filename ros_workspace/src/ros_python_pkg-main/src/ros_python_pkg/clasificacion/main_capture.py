import cv2
import time
import os
import argparse
from clasificador_main import WasteClassificationSystem

def main():
    # 1. Configuración de Argumentos
    parser = argparse.ArgumentParser(description='Captura y clasifica residuos.')
    parser.add_argument('--filter-class', type=str, choices=['botella', 'carton', 'lata'],
                       help='Filtrar por tipo de objeto: botella, carton, lata')
    args = parser.parse_args()

    CAMERA_ID = 0
    RES_W, RES_H = 1280, 720
    CALIB_FILE = "calibracion_camara.pkl"
    OUTPUT_ORIGINAL = "captura_original.png"
    OUTPUT_PROCESSED = "captura_procesada.png"

    print("Iniciando sistema de captura...")
    if args.filter_class:
        print(f"Filtro activado: Solo se mostrarán objetos de tipo '{args.filter_class}'")

    # 2. Inicializar Sistema de Clasificación (Carga calibración internamente)
    try:
        system = WasteClassificationSystem(
            calibration_file=CALIB_FILE,
            min_area=1000,
            confidence_threshold=0.4,
            filter_class=args.filter_class,
            use_ml=True
        )
    except Exception as e:
        print(f"Error al inicializar el sistema: {e}")
        return

    # 3. Capturar Imagen
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
        image_path=OUTPUT_ORIGINAL,
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
        
        coordinates.append(center)

        print(f"Objeto #{obj_id} ({class_name}):")
        print(f"  - Confianza: {confidence:.2f}")
        print(f"  - Coordenadas Centro (X, Y): {center}")
        print("-" * 20)

    print(f"\nProceso finalizado. Imagen procesada en: {OUTPUT_PROCESSED}")
    return coordinates

if __name__ == "__main__":
    main()
