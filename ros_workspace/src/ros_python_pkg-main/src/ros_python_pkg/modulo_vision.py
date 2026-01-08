import os
import cv2
import json
import time
import pickle
from typing import List, Dict, Any

# Fase 2: Clasificación / Segmentación
from clasificacion.camera_calibration import CameraCalibration
import subprocess
import sys

# Fase 3: Detección de puntos y visualización
from deteccion.funciones.version_final_exporta_fotos import analyze_image

# Fase 4: Medición con ArUco
from deteccion.funciones.aruco_batch_measure import detect_arucos, process_items
import numpy as np

IMAGENES_DIR = "imagenes"
PUNTOS_DIR = "puntos"
CALIB_FILE = os.path.join("clasificacion", "calibracion_camara.pkl")


def ensure_dirs():
    os.makedirs(IMAGENES_DIR, exist_ok=True)
    os.makedirs(PUNTOS_DIR, exist_ok=True)


def capture_image(camera_id: int = 0, width: int = 1280, height: int = 720) -> str:
    """Fase 1: Captura una foto en 1280x720 con calibración cenital del pkl.
    Recorta según ROI para eliminar bordes negros por distorsión."""
    ensure_dirs()
    print("[Fase 1] Abriendo cámara para captura...")

    # 1. CARGAR DATOS DE CALIBRACIÓN
    if not os.path.exists(CALIB_FILE):
        raise RuntimeError(f"Error: No se encuentra el archivo {CALIB_FILE}")

    with open(CALIB_FILE, "rb") as f:
        data = pickle.load(f)

    mtx = data["camera_matrix"]
    dist = data["dist_coeff"]
    print(f"[Fase 1] Calibración cargada desde {CALIB_FILE}")

    # 2. CONFIGURAR CÁMARA (1280x720)
    cap = None
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        cap = cv2.VideoCapture(camera_id, backend)
        if cap.isOpened():
            break
    if not cap or not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara {camera_id}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # 3. PRE-CÁLCULO PARA OPTIMIZACIÓN
    print("[Fase 1] Estabilizando cámara...")
    for _ in range(30):
        cap.read()
        time.sleep(0.05)

    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("No se pudo capturar un frame de la cámara")

    h, w = frame.shape[:2]
    print(f"[Fase 1] Resolución capturada: {w}x{h}")

    # 4. CALCULAR MAPAS DE DISTORSIÓN OPTIMALES Y ROI
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)

    # 5. APLICAR UNDISTORT
    undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    print("[Fase 1] Imagen undistort aplicada (optimizada)")

    # 6. RECORTAR SEGÚN ROI PARA ELIMINAR BORDES NEGROS
    x, y, w_roi, h_roi = roi
    undistorted_cropped = undistorted[y:y+h_roi, x:x+w_roi]
    print(f"[Fase 1] Imagen recortada por ROI: {w_roi}x{h_roi} (eliminados bordes negros)")

    output_path = os.path.join(IMAGENES_DIR, "captura_cenital.png")
    cv2.imwrite(output_path, undistorted_cropped)
    print(f"[Fase 1] Imagen capturada guardada en: {output_path}")
    return output_path


def process_with_classifier_via_script(image_path: str | None, filter_class: str | None, json_out: str) -> Dict[str, Any]:
    """Fase 2: Invoca clasificacion/clasificador_imagenes.py con argumentos y recoge JSON."""
    ensure_dirs()
    clasificacion_dir = os.path.join(os.path.dirname(__file__), "clasificacion")
    
    cmd = [sys.executable, "clasificador_imagenes.py"]
    if filter_class:
        cmd += ["--filter-class", filter_class]
    if image_path:
        cmd += ["--image", os.path.abspath(image_path)]
    cmd += ["--json-results", os.path.abspath(json_out)]

    print("[Fase 2] Ejecutando clasificador_imagenes.py...")
    try:
        # Run from clasificacion/ directory so relative imports work
        result = subprocess.run(cmd, cwd=clasificacion_dir, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[ERROR] stdout: {result.stdout}")
            print(f"[ERROR] stderr: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Fallo al ejecutar clasificador_imagenes.py: {e}")

    with open(json_out, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def detect_points_and_visualize(image_path: str) -> Dict[str, str]:
    """Fase 3: Usa versión final para extraer puntos e ilustrar."""
    print("[Fase 3] Analizando imagen para extraer puntos...")
    ensure_dirs()
    analyze_image(image_path)

    base = os.path.splitext(os.path.basename(image_path))[0]
    puntos_json = os.path.join(PUNTOS_DIR, f"{base}_puntos.json")
    resultado_img = os.path.join(IMAGENES_DIR, f"resultado_{os.path.basename(image_path)}")

    print(f"[Fase 3] Puntos JSON: {puntos_json}")
    print(f"[Fase 3] Imagen ilustrada: {resultado_img}")
    return {"puntos_json": puntos_json, "visual_img": resultado_img}


def _prompt_residue_type() -> str:
    """Ask user for residue type: lata, carton, botella."""
    print("Seleccione tipo de residuo: 'lata', 'carton' o 'botella'")
    t = input("Residuo: ").strip().lower()
    valid = {"lata", "carton", "botella"}
    if t not in valid:
        print("[Aviso] Tipo no válido, usando 'botella' por defecto.")
        t = "botella"
    return t


def _map_to_classifier_filter(residue: str) -> str:
    """Map user residue to WasteClassificationSystem filter_class.
    'botella' -> 'botella' (mapping used internally), else self.
    """
    return "botella" if residue == "botella" else residue


def filter_points_by_classification(points_json_path: str, class_results: Dict[str, Any], desired_class_name: str) -> str:
    """Filter puntos JSON to only those whose central point lies inside any contour
    of classified objects of desired_class_name. Returns path of filtered JSON.
    """
    with open(points_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Collect contours for desired class ('lata', 'carton', 'botella') from classification
    contours = []
    for r in class_results.get("results", []):
        if r.get("class") == desired_class_name:
            c = r.get("contour")
            if c is not None:
                # Convert list to numpy array if necessary
                if isinstance(c, list):
                    c = np.array(c, dtype=np.int32)
                contours.append(c)

    # Filter objetos by point-in-polygon (punto_central)
    filtered_objs = []
    for obj in data.get("objetos", []):
        pc = obj.get("punto_central") or obj.get("center")
        if not pc:
            continue
        x, y = int(pc.get("x")), int(pc.get("y"))
        inside = False
        for contour in contours:
            # Ensure proper shape for pointPolygonTest
            if isinstance(contour, np.ndarray):
                if len(contour.shape) == 3 and contour.shape[1] == 1:
                    cnt_pts = contour
                else:
                    cnt_pts = contour.reshape((-1, 1, 2)) if contour.size > 0 else contour
            else:
                # Already a list, convert to array
                contour = np.array(contour, dtype=np.int32)
                cnt_pts = contour.reshape((-1, 1, 2)) if contour.size > 0 else contour
            
            if cnt_pts.size > 0 and cv2.pointPolygonTest(cnt_pts, (x, y), False) >= 0:
                inside = True
                break
        if inside:
            filtered_objs.append(obj)

    filtered = {
        "imagen": data.get("imagen"),
        "objetos": filtered_objs,
        "contenedores": []
    }

    base = os.path.splitext(os.path.basename(points_json_path))[0]
    out_path = os.path.join(PUNTOS_DIR, f"{base}_filtrado.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    # Print only the points of requested object
    print("\nPUNTOS DEL OBJETO PEDIDO:")
    for i, obj in enumerate(filtered_objs, 1):
        pc = obj.get("punto_central") or obj.get("center")
        p1 = obj.get("punto_1")
        p2 = obj.get("punto_2")
        print(f"- Objeto #{i}:")
        if pc:
            print(f"  punto_central: (x={pc['x']}, y={pc['y']})")
        if p1:
            print(f"  punto_1: (x={p1['x']}, y={p1['y']})")
        if p2:
            print(f"  punto_2: (x={p2['x']}, y={p2['y']})")

    return out_path


def measure_with_aruco(image_path: str, puntos_json_path: str, aruco_mm: float = 50.0, dict_name: str = "AUTO") -> Dict[str, Any]:
    """Fase 4: Aplica ArUco para convertir los puntos a mm y exporta resultados."""
    print("[Fase 4] Detectando ArUco y midiendo en mm...")
    ensure_dirs()

    frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"No se pudo leer la imagen: {image_path}")

    with open(puntos_json_path, "r", encoding="utf-8") as f:
        points_data = json.load(f)

    markers = detect_arucos(frame, dict_name)
    if not markers:
        raise RuntimeError("No se detectaron marcadores ArUco en la imagen.")

    # Calcular mm/px promedio
    scales = [float(aruco_mm) / max(m["W_px"], 1e-6) for m in markers]
    mm_per_px = float(np.mean(scales))

    # Usar el marcador más grande como referencia
    main_marker = max(markers, key=lambda m: m["W_px"])

    results = {
        "imagen": image_path,
        "aruco": {
            "ids": [m["id"] for m in markers],
            "reference_id": main_marker["id"],
            "center_px": main_marker["center_px"],
            "avg_mm_per_px": mm_per_px
        },
        "objetos": process_items(points_data.get("objetos", []), main_marker, mm_per_px),
        "contenedores": process_items(points_data.get("contenedores", []), main_marker, mm_per_px)
    }

    output_path = os.path.join(PUNTOS_DIR, "resultados_mm.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[Fase 4] Resultados mm guardados en: {output_path}")
    
    # Crear archivo con coordenadas mm basado en el nombre del archivo de entrada
    base_name = os.path.splitext(os.path.basename(puntos_json_path))[0]
    mm_version_path = os.path.join(PUNTOS_DIR, f"{base_name}_mm.json")
    mm_version_data = {
        "imagen": points_data.get("imagen"),
        "objetos": results.get("objetos", []),
        "contenedores": results.get("contenedores", [])
    }
    with open(mm_version_path, "w", encoding="utf-8") as f:
        json.dump(mm_version_data, f, indent=2, ensure_ascii=False)
    print(f"[Fase 4] Versión con mm guardada en: {mm_version_path}")

    # Visualización
    vis_img = frame.copy()
    for m in markers:
        cx, cy = int(m["center_px"]["x"]), int(m["center_px"]["y"])
        cv2.circle(vis_img, (cx, cy), 8, (0, 255, 255), -1)
        cv2.putText(vis_img, f"ID={m['id']}", (cx+15, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        corners = np.array(m["corners"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis_img, [corners], True, (0, 255, 255), 2)

    # Dibujar objetos/contendedores originales
    for category in ["objetos", "contenedores"]:
        color_circle = (0, 255, 0) if category == "objetos" else (0, 165, 255)
        for obj_data in points_data.get(category, []):
            obj_id = obj_data.get("id")
            # Central
            if "punto_central" in obj_data:
                pc = obj_data["punto_central"]
                cv2.circle(vis_img, (pc["x"], pc["y"]), 6, color_circle, -1)
                label = f"ID{obj_id}" if category == "objetos" else f"C{obj_id}"
                cv2.putText(vis_img, label, (pc["x"]+10, pc["y"]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_circle, 2, cv2.LINE_AA)
            # Línea p1-p2 si existen
            if "punto_1" in obj_data and "punto_2" in obj_data:
                p1 = obj_data["punto_1"]; p2 = obj_data["punto_2"]
                cv2.line(vis_img, (p1["x"], p1["y"]), (p2["x"], p2["y"]), (255, 0, 0), 2)
                cv2.circle(vis_img, (p1["x"], p1["y"]), 4, (255, 255, 0), -1)
                cv2.circle(vis_img, (p2["x"], p2["y"]), 4, (255, 255, 0), -1)
                # texto distancia si está en resultados
                obj_result = next((o for o in results[category] if o["id"] == obj_id), None)
                if obj_result and "distancia_punto1_punto2_mm" in obj_result:
                    midx = (p1["x"] + p2["x"]) // 2
                    midy = (p1["y"] + p2["y"]) // 2
                    dmm = obj_result["distancia_punto1_punto2_mm"]
                    cv2.putText(vis_img, f"{dmm} mm", (midx+10, midy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

    vis_output = os.path.join(IMAGENES_DIR, "resultados_visualizacion.png")
    cv2.imwrite(vis_output, vis_img)
    print(f"[Fase 4] Visualización guardada en: {vis_output}")
    return {"mm_json": output_path, "visual_img": vis_output}


def run_pipeline():
    """Ejecuta el flujo completo con modo 'camara' o 'foto'."""
    ensure_dirs()
    print("--- MODULO VISION: Flujo 4 Fases ---")
    print("1. [Enter] -> Modo FOTO")
    print("2. Escribe 'camara' -> Modo CÁMARA")
    modo = input("Opción: ").strip().lower()

    residue = _prompt_residue_type()
    desired_class_name = {"lata": "lata", "carton": "carton", "botella": "botella"}[residue]
    classifier_filter = _map_to_classifier_filter(residue)

    if modo == "camara":
        # Fase 1 (captura con calibración)
        captura_path = capture_image(camera_id=1, width=1280, height=720)
        # Fase 2: usar script por argumentos
        json_out = os.path.join(IMAGENES_DIR, "fase2_resultados.json")
        fase2 = process_with_classifier_via_script(captura_path, classifier_filter, json_out)
        if not fase2.get("results"):
            print("[INFO] No se encontraron objetos del tipo solicitado. Terminando flujo.")
            return
        # Fase 3 (detectar puntos)
        fase3 = detect_points_and_visualize(captura_path)
        # Filtrar puntos por clasificación
        puntos_filtrados = filter_points_by_classification(fase3["puntos_json"], fase2, desired_class_name)
        # Fase 4 (medir solo puntos filtrados)
        measure_with_aruco(captura_path, puntos_filtrados, aruco_mm=50.0, dict_name="AUTO")
        print("\nFlujo completado en modo cámara.")
    else:
        # Modo foto: usar 'foto_ejemplo.png' sin calibración
        image_path = "foto_ejemplo.png"
        if not os.path.exists(image_path):
            print(f"[ERROR] No se encontró la imagen: {image_path}")
            return
        # Fase 2: usar script por argumentos
        json_out = os.path.join(IMAGENES_DIR, "fase2_resultados.json")
        fase2 = process_with_classifier_via_script(image_path, classifier_filter, json_out)
        if not fase2.get("results"):
            print("[INFO] No se encontraron objetos del tipo solicitado. Terminando flujo.")
            return
        # Fase 3 (detectar puntos)
        fase3 = detect_points_and_visualize(image_path)
        # Filtrar puntos por clasificación
        puntos_filtrados = filter_points_by_classification(fase3["puntos_json"], fase2, desired_class_name)
        # Fase 4 (medir solo puntos filtrados)
        measure_with_aruco(image_path, puntos_filtrados, aruco_mm=50.0, dict_name="AUTO")
        print("\nFlujo completado en modo foto.")


if __name__ == "__main__":
    run_pipeline()
