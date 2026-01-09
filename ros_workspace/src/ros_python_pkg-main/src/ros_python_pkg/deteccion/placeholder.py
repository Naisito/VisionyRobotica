import os
import cv2
import json
import tempfile
from typing import Dict, Any
import subprocess
import sys
import numpy as np

from deteccion.funciones.aruco_batch_measure import detect_arucos, process_items

IMAGENES_DIR = "imagenes"
PUNTOS_DIR = "puntos"


def ensure_dirs():
    os.makedirs(IMAGENES_DIR, exist_ok=True)
    os.makedirs(PUNTOS_DIR, exist_ok=True)


def process_with_classifier_via_script(image_path: str, filter_class: str) -> Dict[str, Any]:
    """Fase 2: Invoca clasificador_imagenes.py y recoge JSON."""
    ensure_dirs()
    clasificacion_dir = os.path.join(os.path.dirname(__file__), "clasificacion")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        json_out = tmp.name
    
    cmd = [sys.executable, "clasificador_imagenes.py"]
    if filter_class:
        cmd += ["--filter-class", filter_class]
    if image_path:
        cmd += ["--image", os.path.abspath(image_path)]
    cmd += ["--json-results", json_out]

    result = subprocess.run(cmd, cwd=clasificacion_dir, capture_output=True, text=True)
    if result.returncode != 0:
        try:
            os.remove(json_out)
        except:
            pass
        raise RuntimeError(f"Fallo clasificador: {result.stderr}")

    with open(json_out, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    try:
        os.remove(json_out)
    except:
        pass
    
    return data


def get_detections_from_image(image_path: str) -> Dict[str, Any]:
    """Fase 3: Extrae puntos de la imagen y devuelve datos en memoria."""
    from deteccion.funciones.version_final_exporta_fotos import get_detections, adjust_brightness_to_target
    
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"No se pudo leer: {image_path}")
    
    img = adjust_brightness_to_target(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detections, _ = get_detections(gray)
    
    # Construir estructura de datos
    data = {"imagen": image_path, "objetos": []}
    for det in detections:
        gc = det.get("grip_center")
        p1 = det.get("grip_p1")
        p2 = det.get("grip_p2")
        if gc is None or p1 is None or p2 is None:
            continue
        obj = {
            "id": det.get("id"),
            "tipo": det.get("type"),
            "punto_central": {"x": int(gc[0]), "y": int(gc[1])},
            "punto_1": {"x": int(p1[0]), "y": int(p1[1])},
            "punto_2": {"x": int(p2[0]), "y": int(p2[1])},
            "es_tapon": det.get("has_cap", False)
        }
        data["objetos"].append(obj)
    
    return data


def _map_to_classifier_filter(residue: str) -> str:
    return "botella" if residue == "botella" else residue


def filter_points_by_classification(points_data: Dict[str, Any], class_results: Dict[str, Any], desired_class_name: str) -> Dict[str, Any]:
    """Filtra puntos por clasificación (en memoria)."""
    contours = []
    for r in class_results.get("results", []):
        if r.get("class") == desired_class_name:
            c = r.get("contour")
            if c is not None:
                if isinstance(c, list):
                    c = np.array(c, dtype=np.int32)
                contours.append(c)

    filtered_objs = []
    for obj in points_data.get("objetos", []):
        pc = obj.get("punto_central") or obj.get("center")
        if not pc:
            continue
        x, y = int(pc.get("x", 0)), int(pc.get("y", 0))
        inside = False
        for contour in contours:
            if isinstance(contour, np.ndarray):
                if len(contour.shape) == 3 and contour.shape[1] == 1:
                    cnt_pts = contour
                else:
                    cnt_pts = contour.reshape((-1, 1, 2)) if contour.size > 0 else contour
            else:
                contour = np.array(contour, dtype=np.int32)
                cnt_pts = contour.reshape((-1, 1, 2)) if contour.size > 0 else contour
            
            if cnt_pts.size > 0 and cv2.pointPolygonTest(cnt_pts, (x, y), False) >= 0:
                inside = True
                break
        if inside:
            filtered_objs.append(obj)

    return {
        "imagen": points_data.get("imagen"),
        "objetos": filtered_objs,
        "contenedores": []
    }


def measure_with_aruco(image_path: str, points_data: Dict[str, Any], aruco_mm: float = 50.0, dict_name: str = "AUTO") -> Dict[str, Any]:
    """Fase 4: Mide con ArUco y genera imagen final."""
    ensure_dirs()

    frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"No se pudo leer la imagen: {image_path}")

    markers = detect_arucos(frame, dict_name)
    if not markers:
        raise RuntimeError("No se detectaron marcadores ArUco.")

    # Forzar uso de ArUco ID=7 como referencia
    REF_MARKER_ID = 7
    main_marker = None
    for m in markers:
        if int(m.get("id")) == REF_MARKER_ID:
            main_marker = m
            break
    if main_marker is None:
        raise RuntimeError(f"No se encontró el marcador ArUco ID={REF_MARKER_ID}")

    # Calcular escala solo con el marcador de referencia
    mm_per_px = float(aruco_mm) / max(main_marker["W_px"], 1e-6)

    results = {
        "imagen": image_path,
        "aruco": {
            "ids": [REF_MARKER_ID],
            "reference_id": REF_MARKER_ID,
            "center_px": main_marker["center_px"],
            "avg_mm_per_px": mm_per_px
        },
        "objetos": process_items(points_data.get("objetos", []), main_marker, mm_per_px),
        "contenedores": process_items(points_data.get("contenedores", []), main_marker, mm_per_px)
    }

    # Guardar solo resultados_mm.json
    output_path = os.path.join(PUNTOS_DIR, "resultados_mm.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results


def run_pipeline(foto, residue_name):
    """Ejecuta el flujo completo.
    
    Args:
        foto: Imagen numpy (BGR)
        residue_name: 'lata', 'carton' o 'botella'
    
    Returns:
        dict con resultados o None si no hay objetos
    """
    ensure_dirs()
    
    desired_class_name = {"lata": "lata", "carton": "carton", "botella": "botella"}[residue_name]
    classifier_filter = _map_to_classifier_filter(residue_name)
    
    # Guardar imagen temporal
    image_path = os.path.join(IMAGENES_DIR, "temp_foto.png")
    cv2.imwrite(image_path, foto)
    
    # Fase 2: clasificación
    fase2 = process_with_classifier_via_script(image_path, classifier_filter)
    if not fase2.get("results"):
        print(f"No se encontraron objetos de tipo '{residue_name}'.")
        return None
    
    print(f"Detectados {len(fase2['results'])} objetos de tipo '{residue_name}'")
    
    # Fase 3: detectar puntos (en memoria)
    points_data = get_detections_from_image(image_path)
    
    # Filtrar puntos (en memoria)
    filtered_points = filter_points_by_classification(points_data, fase2, desired_class_name)
    
    # Fase 4: medir y generar imagen final
    results = measure_with_aruco(image_path, filtered_points, aruco_mm=48.0, dict_name="AUTO")
    
    return results


if __name__ == "__main__":
    image_path = "foto_ejemplo.png"
    foto = cv2.imread(image_path)
    if foto is None:
        raise RuntimeError(f"No se pudo leer la imagen: {image_path}")
    run_pipeline(foto, "lata")
