import os
import cv2
import json
import tempfile
from typing import Dict, Any
import subprocess
import sys
import numpy as np

from deteccion.funciones.aruco_batch_measure import detect_arucos, process_items

# Importar calibración
try:
    from clasificacion.camera_calibration import CameraCalibration
except ImportError:
    print("[WARN] No se pudo importar CameraCalibration")

# Importar función de cálculo de altura
try:
    from alturaObjetos.test_camara_horizontal import altura_objetos
    ALTURA_DISPONIBLE = True
except ImportError:
    print("[WARN] No se pudo importar altura_objetos, las alturas serán 0.0")
    ALTURA_DISPONIBLE = False

# Usar rutas absolutas consistentes con nodoObjetos
SCRIPT_DIR = os.path.join("/", "home", "laboratorio", "ros_workspace")
IMAGENES_DIR = os.path.join(SCRIPT_DIR, "imagenes")
PUNTOS_DIR = os.path.join(SCRIPT_DIR, "puntos")
CALIB_FILE = os.path.join(os.path.dirname(__file__), "clasificacion", "calibracion_camara_cenital.pkl")

# Cargar calibración global
_calibration = None
def get_calibration():
    global _calibration
    if _calibration is None and os.path.exists(CALIB_FILE):
        try:
            _calibration = CameraCalibration.from_pickle(CALIB_FILE)
        except Exception as e:
            print(f"[WARN] No se pudo cargar calibración: {e}")
    return _calibration

def undistort_image(img):
    """Aplica corrección de distorsión si hay calibración disponible."""
    calib = get_calibration()
    if calib is not None:
        return calib.undistort(img)
    return img


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
    # TEMPORAL: Sin filtro para debug
    # if filter_class:
    #     cmd += ["--filter-class", filter_class]
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
    
    # DEBUG: Ver qué devuelve el clasificador
    print(f"[DEBUG] Clasificador devuelve: {len(data.get('results', []))} resultados")
    for r in data.get('results', []):
        print(f"  -> clase: '{r.get('class')}', conf: {r.get('confidence')}")
    
    try:
        os.remove(json_out)
    except:
        pass
    
    return data


def get_detections_from_image(image_path: str) -> Dict[str, Any]:
    """Fase 3: Extrae puntos de la imagen y devuelve datos en memoria."""
    from deteccion.funciones.version_final_exporta_fotos import get_detections
    
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"No se pudo leer: {image_path}")
    
    # Aplicar calibración para corregir distorsión
    img = undistort_image(img)
    
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
        clase = r.get("class", "")
        # Comparación case-insensitive
        if clase.upper() == desired_class_name.upper():
            c = r.get("contour")
            if c is not None:
                if isinstance(c, list):
                    c = np.array(c, dtype=np.int32)
                contours.append(c)
                # Debug: mostrar bounding box del contorno
                if c.size > 0:
                    c_reshaped = c.reshape(-1, 2) if len(c.shape) != 2 else c
                    x_min, y_min = c_reshaped.min(axis=0)
                    x_max, y_max = c_reshaped.max(axis=0)
                    print(f"[DEBUG] Contorno '{clase}': bbox=({x_min},{y_min})-({x_max},{y_max})")

    print(f"[DEBUG] Contornos encontrados para '{desired_class_name}': {len(contours)}")

    filtered_objs = []
    for obj in points_data.get("objetos", []):
        pc = obj.get("punto_central") or obj.get("center")
        if not pc:
            continue
        x, y = int(pc.get("x", 0)), int(pc.get("y", 0))
        print(f"[DEBUG] Punto ID={obj.get('id')}: ({x}, {y})")
        inside = False
        for i, contour in enumerate(contours):
            if isinstance(contour, np.ndarray):
                if len(contour.shape) == 3 and contour.shape[1] == 1:
                    cnt_pts = contour
                else:
                    cnt_pts = contour.reshape((-1, 1, 2)) if contour.size > 0 else contour
            else:
                contour = np.array(contour, dtype=np.int32)
                cnt_pts = contour.reshape((-1, 1, 2)) if contour.size > 0 else contour
            
            if cnt_pts.size > 0:
                result = cv2.pointPolygonTest(cnt_pts, (x, y), False)
                print(f"[DEBUG]   -> Contorno {i}: pointPolygonTest = {result}")
                if result >= 0:
                    inside = True
                    break
        if inside:
            filtered_objs.append(obj)

    print(f"[DEBUG] Objetos filtrados: {len(filtered_objs)}")
    return {
        "imagen": points_data.get("imagen"),
        "objetos": filtered_objs,
        "contenedores": []
    }


def measure_with_aruco(image_path: str, points_data: Dict[str, Any], aruco_mm: float = 50.0, dict_name: str = "AUTO", clase_detectada: str = None, img_cenital=None, img_horizontal=None) -> Dict[str, Any]:
    """Fase 4: Mide con ArUco y genera imagen final.
    
    Args:
        image_path: Ruta a la imagen cenital
        points_data: Datos de los puntos detectados (cada objeto puede tener su propia 'clase')
        aruco_mm: Tamaño del marcador ArUco en mm
        dict_name: Diccionario de ArUco a usar
        clase_detectada: Clase fija para todos los objetos (deprecated, usar clase en cada objeto)
        img_cenital: Imagen cenital para cálculo de altura (opcional)
        img_horizontal: Imagen horizontal para cálculo de altura (opcional)
    """
    ensure_dirs()

    frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"No se pudo leer la imagen: {image_path}")

    # Aplicar calibración para corregir distorsión
    frame = undistort_image(frame)

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

    # DEBUG: ver cuántos objetos llegan a measure_with_aruco
    objetos_entrada = points_data.get("objetos", [])
    print(f"[DEBUG measure_with_aruco] Recibidos {len(objetos_entrada)} objetos para medir")
    for obj in objetos_entrada:
        print(f"  - ID={obj.get('id')}, clase={obj.get('clase', 'N/A')}, centro={obj.get('punto_central')}")

    # Procesar objetos y calcular alturas
    objetos_procesados = process_items(objetos_entrada, main_marker, mm_per_px)
    print(f"[DEBUG measure_with_aruco] Procesados {len(objetos_procesados)} objetos después de process_items")
    
    # Calcular altura para cada objeto si hay imágenes disponibles
    if ALTURA_DISPONIBLE and img_cenital is not None and img_horizontal is not None:
        print("[INFO] Calculando alturas de objetos...")
        for obj in objetos_procesados:
            pc = obj.get("punto_central")
            if pc:
                x_px = int(pc.get("x_px", 0))
                y_px = int(pc.get("y_px", 0))
                
                if x_px > 0 and y_px > 0:
                    try:
                        altura_mm = altura_objetos(
                            x=x_px,
                            y=y_px,
                            img_cenital=img_cenital,
                            img_horizontal=img_horizontal
                        )
                        obj["altura_mm"] = round(altura_mm, 2)
                        print(f"[INFO] Objeto ID={obj.get('id')}: altura={altura_mm:.2f} mm")
                    except Exception as e:
                        print(f"[WARN] Error calculando altura para objeto {obj.get('id')}: {e}")
                        obj["altura_mm"] = 0.0
                else:
                    obj["altura_mm"] = 0.0
            else:
                obj["altura_mm"] = 0.0
    else:
        # Si no hay imágenes disponibles, altura por defecto
        for obj in objetos_procesados:
            obj["altura_mm"] = 0.0
        if not ALTURA_DISPONIBLE:
            print("[WARN] Función de altura no disponible, usando altura=0.0")
        else:
            print("[WARN] Imágenes de cámara no disponibles, usando altura=0.0")

    results = {
        "imagen": image_path,
        "aruco": {
            "ids": [REF_MARKER_ID],
            "reference_id": REF_MARKER_ID,
            "center_px": main_marker["center_px"],
            "avg_mm_per_px": mm_per_px
        },
        "objetos": objetos_procesados,
        "contenedores": process_items(points_data.get("contenedores", []), main_marker, mm_per_px)
    }

    # DEBUG: Verificar cuántos objetos se van a guardar
    print(f"[DEBUG] Guardando JSON con {len(objetos_procesados)} objeto(s)")
    for obj in objetos_procesados:
        print(f"  - ID={obj.get('id')}, clase={obj.get('clase')}, x_mm={obj.get('punto_central', {}).get('x_mm')}, y_mm={obj.get('punto_central', {}).get('y_mm')}")

    # Guardar solo resultados_mm.json
    output_path = os.path.join(PUNTOS_DIR, "resultados_mm.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # === IMAGEN FINAL CON OBJETOS MARCADOS ===
    vis_img = frame.copy()
    
    # Colores por tipo de residuo
    colores = {
        "LATA": (0, 165, 255),      # Naranja
        "CARTON": (0, 100, 255),    # Marrón/Naranja oscuro
        "BOTELLA": (0, 255, 0),     # Verde
    }
    
    # Dibujar ArUcos
    for m in markers:
        cx, cy = int(m["center_px"]["x"]), int(m["center_px"]["y"])
        cv2.circle(vis_img, (cx, cy), 6, (0, 255, 255), -1)
        corners = np.array(m["corners"], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis_img, [corners], True, (0, 255, 255), 2)
    
    # Dibujar objetos detectados
    for obj_data in results.get("objetos", []):
        obj_id = obj_data.get("id")
        pc = obj_data.get("punto_central")
        clase_obj = obj_data.get("clase", "DESCONOCIDO")  # Usar clase del objeto
        
        # Color según la clase del objeto
        color = colores.get(clase_obj.upper(), (255, 255, 0))  # Cian por defecto
        
        if pc:
            # Obtener coordenadas en píxeles
            px = int(pc.get("x_px", pc.get("x", 0)))
            py = int(pc.get("y_px", pc.get("y", 0)))
            
            if px == 0 and py == 0:
                continue
            
            # Dibujar punto central
            cv2.circle(vis_img, (px, py), 10, color, -1)
            cv2.circle(vis_img, (px, py), 12, (255, 255, 255), 2)
            
            # Etiqueta con clase real del objeto y su ID
            label = f"{clase_obj} #{obj_id}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            # Fondo de la etiqueta
            cv2.rectangle(vis_img, (px - 5, py - th - 25), (px + tw + 10, py - 5), color, -1)
            # Texto de la etiqueta
            cv2.putText(vis_img, label, (px, py - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Guardar imagen
    vis_output = os.path.join(IMAGENES_DIR, "resultado_final.png")
    cv2.imwrite(vis_output, vis_img)
    print(f"Imagen guardada: {vis_output}")
    
    return results


def run_pipeline(foto, residue_name, img_horizontal=None):
    """Ejecuta el flujo completo usando solo clasificación (sin fase de detección).
    
    Args:
        foto: Imagen numpy (BGR) de la cámara cenital
        residue_name: 'lata', 'carton' o 'botella'
        img_horizontal: Imagen numpy (BGR) de la cámara horizontal (opcional, para cálculo de altura)
    
    Returns:
        dict con resultados o None si no hay objetos
    """
    ensure_dirs()
    
    # El clasificador devuelve clases en MAYÚSCULAS
    desired_class_name = {"lata": "LATA", "carton": "CARTON", "botella": "BOTELLA"}[residue_name]
    classifier_filter = _map_to_classifier_filter(residue_name)
    
    # Guardar imagen temporal
    image_path = os.path.join(IMAGENES_DIR, "temp_foto.png")
    cv2.imwrite(image_path, foto)
    
    # Fase 1: Clasificación (devuelve centro, contorno y clase)
    print("[INFO] Iniciando clasificación...")
    clasificacion_data = process_with_classifier_via_script(image_path, classifier_filter)
    if not clasificacion_data.get("results"):
        print(f"No se encontraron objetos.")
        return None
    
    # Filtrar solo los objetos del tipo solicitado por el gesto
    objetos_filtrados = [
        obj for obj in clasificacion_data["results"] 
        if obj.get("class", "").upper() == desired_class_name.upper()
    ]
    
    if not objetos_filtrados:
        print(f"No se encontraron objetos de tipo '{residue_name}' (se detectaron {len(clasificacion_data['results'])} objetos de otros tipos).")
        return None
    
    print(f"Detectados {len(objetos_filtrados)} objeto(s) de tipo '{residue_name}' (total objetos: {len(clasificacion_data['results'])})")
    
    # Convertir resultados de clasificación a formato de puntos
    # Usamos el centro del clasificador directamente Y MANTENEMOS LA CLASE REAL
    points_data = {"imagen": image_path, "objetos": []}
    for i, obj_class in enumerate(objetos_filtrados, start=1):
        center = obj_class.get("center", {})
        clase_real = obj_class.get("class", "DESCONOCIDO")  # Mantener clase real
        points_data["objetos"].append({
            "id": i,
            "punto_central": {"x": center.get("x"), "y": center.get("y")},
            "clase": clase_real  # NUEVO: guardar clase real del objeto
        })
    
    print(f"[INFO] Usando centros de {len(points_data['objetos'])} objeto(s) de tipo '{residue_name}'")
    
    # Fase 2: Medir con ArUco y generar imagen final (con cálculo de altura)
    print("[INFO] Iniciando medición ArUco...")
    results = measure_with_aruco(
        image_path, 
        points_data, 
        aruco_mm=48.0, 
        dict_name="AUTO", 
        clase_detectada=None,  # No pasar clase fija, usar la clase de cada objeto
        img_cenital=foto,
        img_horizontal=img_horizontal
    )
    print("[INFO] Pipeline completado")
    
    return results
