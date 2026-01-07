import os
import sys
import subprocess
import cv2
import pickle
import numpy as np
import time
from funciones.version_final_exporta_fotos import analyze_image

# --- CONFIGURACIÓN DE RUTAS Y PARÁMETROS ---
IMAGE_PATH = os.path.join("imagenes", "captura_camara.png")
CALIB_FILE = "calibracion_camara_cenital.pkl"
ARUCO_MM = 50
ARUCO_SIZE = "AUTO"

def capturar_foto_hd_calibrada(output_path):
    """Captura a 1280x720, aplica corrección de lente y guarda."""
    
    # 1. Cargar datos de calibración
    if not os.path.exists(CALIB_FILE):
        print(f"❌ Error: No se encuentra el archivo de calibración '{CALIB_FILE}'")
        return False

    try:
        with open(CALIB_FILE, "rb") as f:
            data = pickle.load(f)
        mtx = data["camera_matrix"]
        dist = data["dist_coeff"]
    except Exception as e:
        print(f"❌ Error al leer el archivo pkl: {e}")
        return False

    # 2. Configurar Cámara
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("📸 Abriendo cámara...")
    time.sleep(2)  # Pausa para auto-enfoque y exposición

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("❌ Error: No se pudo obtener imagen de la cámara.")
        return False

    # 3. Aplicar Calibración (Undistort)
    h, w = frame.shape[:2]
    # Obtener matriz de cámara optimizada y ROI para eliminar bordes negros
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # 4. Recortar la imagen según el ROI (limpieza de bordes)
    x, y, w_roi, h_roi = roi
    dst = dst[y:y+h_roi, x:x+w_roi]

    # Asegurar que la carpeta existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 5. Guardar resultado final
    cv2.imwrite(output_path, dst)
    print(f"✅ Foto capturada (1280x720), calibrada y guardada en: {output_path}")
    return True

def run_pipeline(use_camera=True):
    print("\n" + "="*50)
    print("🚀 INICIANDO PIPELINE DE VISIÓN CALIBRADO (HD)")
    print("="*50)

    # Definir rutas de archivos
    current_image = IMAGE_PATH
    current_json = os.path.join("puntos", "captura_camara_puntos.json")

    # --- PASO 1: OBTENCIÓN DE IMAGEN ---
    if use_camera:
        if not capturar_foto_hd_calibrada(current_image):
            return
    else:
        if not os.path.exists(current_image):
            print(f"❌ Error: No existe la imagen {current_image}")
            return
        print(f"ℹ️ Usando imagen existente: {current_image}")

    # --- PASO 2: ANÁLISIS DE PUNTOS ---
    print("\n[PASO 1] Detectando puntos en imagen calibrada...")
    try:
        # Usamos analyze_image directamente sobre la foto ya corregida
        analyze_image(current_image)
    except Exception as e:
        print(f"❌ Error en detección de puntos: {e}")
        return

    # --- PASO 3: DETECCIÓN DE CONTENEDORES ---
    print("\n[PASO 2] Detección de Contenedores (detector_rectangulos_negros.py)")
    detector_script = os.path.join("funciones", "detector_rectangulos_negros.py")
    detector_save = os.path.join("imagenes", "resultado_deteccion_cajas.png")
    
    cmd_detector = [
        sys.executable, detector_script,
        "--image", current_image,
        "--save", detector_save
    ]
    
    try:
        subprocess.run(cmd_detector, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en detección de cajas: {e}")
        return

    # --- PASO 4: MEDICIÓN ARUCO Y CONVERSIÓN ---
    print("\n[PASO 3] Medición y Conversión a mm (aruco_batch_measure.py)")
    measure_script = os.path.join("funciones", "aruco_batch_measure.py")
    
    cmd_measure = [
        sys.executable, measure_script,
        "--image", current_image,
        "--points-json", current_json,
        "--aruco-mm", str(ARUCO_MM),
        "--aruco-tamano", ARUCO_SIZE
    ]
    
    try:
        subprocess.run(cmd_measure, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en medición final: {e}")
        return

    print("\n" + "="*50)
    print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    print(f"📂 Resultados listos en 'puntos/' e 'imagenes/'")
    print("="*50)

def main():
    while True:
        print("\n--- MENÚ DE CONTROL ---")
        print("1. Capturar foto HD + Calibrar + Ejecutar Pipeline")
        print("2. Ejecutar Pipeline sobre última captura (captura_camara.png)")
        print("3. Salir")
        
        opcion = input("Selecciona una opción: ").strip()
        
        if opcion == "1":
            run_pipeline(use_camera=True)
        elif opcion == "2":
            run_pipeline(use_camera=False)
        elif opcion == "3":
            print("👋 Saliendo...")
            break
        else:
            print("⚠️ Opción no válida.")

if __name__ == "__main__":
    main()