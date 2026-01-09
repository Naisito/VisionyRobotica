#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
camara_horizontal.py

Función de caja negra para obtener la altura de un objeto dado:
  - Imagen cenital
  - Imagen horizontal (lateral)
  - Coordenadas del objeto en la imagen cenital

Devuelve la altura en mm del objeto.
"""

import os
import pickle
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np


# ----------------------------
# ArUco helpers
# ----------------------------
@dataclass
class ArucoDetection:
    corners: np.ndarray
    ids: np.ndarray


def detect_aruco(img_bgr: np.ndarray, dict_name: str = "DICT_6X6_50") -> Optional[ArucoDetection]:
    aruco = cv2.aruco
    aruco_dict = getattr(aruco, dict_name, None)
    if aruco_dict is None:
        raise ValueError(f"Diccionario ArUco no válido: {dict_name}")
    dictionary = aruco.getPredefinedDictionary(aruco_dict)
    params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, params)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None
    return ArucoDetection(corners=np.array(corners, dtype=np.float32), ids=ids)


def select_marker_by_max_perimeter(det: ArucoDetection) -> Tuple[np.ndarray, int]:
    best_i, best_p, best_id = 0, -1.0, int(det.ids[0][0])
    for i in range(len(det.ids)):
        c = det.corners[i][0]
        per = cv2.arcLength(c.astype(np.float32), True)
        if per > best_p:
            best_p, best_i, best_id = per, i, int(det.ids[i][0])
    return det.corners[best_i], best_id


def marker_center(corners_1x4x2: np.ndarray) -> Tuple[int, int]:
    c = corners_1x4x2[0] if corners_1x4x2.ndim == 3 else corners_1x4x2
    cx = int(np.mean(c[:, 0]))
    cy = int(np.mean(c[:, 1]))
    return cx, cy


def px_per_cm_from_marker(corners_1x4x2: np.ndarray, marker_length_cm: float) -> float:
    c = corners_1x4x2[0] if corners_1x4x2.ndim == 3 else corners_1x4x2
    per = cv2.arcLength(c.astype(np.float32), True)
    return per / (4.0 * marker_length_cm)


# ----------------------------
# Shared helpers
# ----------------------------
def contour_solidity(cnt: np.ndarray) -> float:
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    return float(area / (hull_area + 1e-9))


def touches_border(x: int, y: int, w: int, h: int, W: int, H: int, margin: int) -> bool:
    return (x <= margin) or (y <= margin) or (x + w >= W - margin) or (y + h >= H - margin)


def height_px_from_contour(cnt: np.ndarray) -> int:
    _, _, _, h = cv2.boundingRect(cnt)
    return int(h)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


# ----------------------------
# CENITAL: multi-objeto
# ----------------------------
def get_mat_mask(gray: np.ndarray, mat_g_max: int = 150, close_ksize: int = 17, close_iter: int = 2) -> np.ndarray:
    _, m = cv2.threshold(gray, mat_g_max, 255, cv2.THRESH_BINARY_INV)
    m = cv2.medianBlur(m, 5)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, close_iter)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros_like(m)
    biggest = max(cnts, key=cv2.contourArea)
    mask = np.zeros_like(m)
    cv2.drawContours(mask, [biggest], -1, 255, thickness=cv2.FILLED)
    return mask


def objects_mask(gray: np.ndarray, roi_mask: np.ndarray, delta_g: int = 20) -> np.ndarray:
    roi_vals = gray[roi_mask > 0]
    if roi_vals.size == 0:
        return np.zeros_like(roi_mask)

    g_med = np.median(roi_vals)
    rel = cv2.threshold(gray, int(g_med + delta_g), 255, cv2.THRESH_BINARY)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    th = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    mask = cv2.bitwise_or(rel, th)
    mask = cv2.bitwise_and(mask, roi_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), 1)
    return mask


def detect_objects_cenital(img_bgr: np.ndarray, max_objs: int = 3) -> List[Tuple[np.ndarray, Tuple[int, int], np.ndarray]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    roi = get_mat_mask(gray, mat_g_max=150, close_ksize=17, close_iter=2)
    obj_mask = objects_mask(gray, roi, delta_g=20)

    cnts, hierarchy = cv2.findContours(obj_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape[:2]
    if not cnts:
        return []

    hier = hierarchy[0] if hierarchy is not None and len(hierarchy) > 0 else None
    top_level_idxs = [i for i in range(len(cnts)) if hier is None or int(hier[i][3]) == -1]

    candidates = []
    cx0, cy0 = W / 2.0, H / 2.0
    for i in top_level_idxs:
        c = cnts[i]
        area = cv2.contourArea(c)
        if area < 600 or area > 80000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if touches_border(x, y, w, h, W, H, margin=25):
            continue
        sol = contour_solidity(c)
        if sol < 0.35:
            continue
        M = cv2.moments(c)
        if abs(M["m00"]) < 1e-9:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        dist = np.hypot(cx - cx0, cy - cy0) / max(W, H)
        score = sol * 2.0 + (min(area, 20000) / 20000.0) * 0.5 - dist * 0.8
        candidates.append((score, c))

    if not candidates:
        return []

    candidates.sort(key=lambda t: t[0], reverse=True)
    results = []
    for score, c in candidates[:max_objs]:
        M = cv2.moments(c)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        results.append((c, (cx, cy), obj_mask))

    # Orden estable por X (izq -> der)
    results.sort(key=lambda r: cv2.boundingRect(r[0])[0])
    return results


# ----------------------------
# LATERAL: multi-objeto robusto
# ----------------------------
def detect_objects_lateral(
    img_bgr: np.ndarray,
    aruco_det: Optional[ArucoDetection],
    max_objs: int = 3,
    top_cut_ratio: float = 0.35,
) -> List[Tuple[np.ndarray, Tuple[int, int], np.ndarray]]:
    """
    Lateral multi-objeto (robusto para tu escena):
      - Prioriza segmentación por COLOR (verde caja + azul lata) en HSV con S alto.
      - Aplica ROI inferior (evita techo/ventana) y excluye ArUco.
      - findContours con jerarquía (RETR_CCOMP) y nos quedamos con top-level.
      - Filtrado geométrico: apoyado en mesa, no tocar borde derecho, tamaño razonable.

    Devuelve:
      results: [(contour, (cx,cy), mask_used_for_that_contour)]
    """
    H, W = img_bgr.shape[:2]

    # ROI inferior
    y0 = int(H * top_cut_ratio)
    roi_bottom = np.zeros((H, W), np.uint8)
    roi_bottom[y0:, :] = 255

    # Máscara ArUco (dilatada) para excluirlo
    ar_mask = np.zeros((H, W), np.uint8)
    if aruco_det is not None and aruco_det.corners is not None:
        for c in aruco_det.corners:
            poly = np.int32(c.reshape(-1, 2))
            cv2.fillConvexPoly(ar_mask, poly, 255)
        ar_mask = cv2.dilate(ar_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17)), 1)

    # HSV masks (S alto para evitar blancos)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    Hh, Ss, Vv = cv2.split(hsv)

    # Verde (caja): rango amplio, S alto
    mask_green = cv2.inRange(hsv, (30, 60, 40), (95, 255, 255))
    # Azul (lata): rango azul
    mask_blue = cv2.inRange(hsv, (90, 60, 40), (145, 255, 255))

    # Quitar brillos/blancos: donde S es baja, lo eliminamos
    low_sat = cv2.inRange(Ss, 0, 45)
    mask_green[low_sat > 0] = 0
    mask_blue[low_sat > 0] = 0

    # Aplicar ROI y excluir ArUco
    for m in (mask_green, mask_blue):
        m[roi_bottom == 0] = 0
        m[ar_mask > 0] = 0

    # Limpieza morfológica
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, k3, iterations=1)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, k7, iterations=2)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, k3, iterations=1)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, k7, iterations=2)

    def top_level_contours(mask: np.ndarray):
        cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None or len(cnts) == 0:
            return []
        hier = hier[0]
        return [cnts[i] for i in range(len(cnts)) if hier[i][3] == -1]

    def contour_ok(c: np.ndarray) -> bool:
        x, y, w, h = cv2.boundingRect(c)
        area = float(cv2.contourArea(c))
        if area < 800:
            return False
        if area > 0.45 * (W * (H - y0)):
            return False
        if y + h < int(H * 0.60):
            return False
        if x + w > int(W * 0.98):
            return False
        if w > 0.85 * W and h < 0.25 * (H - y0):
            return False
        ar = w / max(1.0, h)
        if ar > 3.0:
            return False
        if h < 30:
            return False
        # Filtrar objetos muy pequeños en relación al área total de la imagen
        min_area_ratio = 0.005  # 0.5% del área total mínimo
        total_area = W * (H - y0)
        if area < min_area_ratio * total_area:
            return False
        return True

    def scored_candidates(mask: np.ndarray, bonus: float = 0.0):
        cands = []
        for c in top_level_contours(mask):
            if not contour_ok(c):
                continue
            x, y, w, h = cv2.boundingRect(c)
            area = float(cv2.contourArea(c))
            score = area + bonus
            if y + h >= H - 2:
                score *= 0.6
            if x <= 2:
                score *= 0.8
            cands.append((score, c, mask))
        cands.sort(key=lambda t: t[0], reverse=True)
        return cands

    candidates = []
    candidates += scored_candidates(mask_green, bonus=5e5)
    candidates += scored_candidates(mask_blue, bonus=3e5)

    # Fallback genérico si no hay suficientes por color
    mask_mat = None
    if len(candidates) < 2:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        mat = get_mat_mask(gray)
        obj = objects_mask(gray, mat, delta_g=20)
        mask_mat = cv2.bitwise_and(obj, roi_bottom)
        mask_mat[ar_mask > 0] = 0
        mask_mat = cv2.morphologyEx(mask_mat, cv2.MORPH_OPEN, k3, iterations=1)
        mask_mat = cv2.morphologyEx(mask_mat, cv2.MORPH_CLOSE, k7, iterations=2)
        candidates += scored_candidates(mask_mat, bonus=0.0)

    # NMS por IoU en bounding boxes para evitar duplicados
    def iou(a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter <= 0:
            return 0.0
        ua = aw * ah + bw * bh - inter
        return inter / max(1.0, ua)

    picked = []
    picked_boxes = []
    for score, c, src_mask in candidates:
        box = cv2.boundingRect(c)
        if any(iou(box, pb) > 0.35 for pb in picked_boxes):
            continue
        picked.append((c, src_mask))
        picked_boxes.append(box)
        if len(picked) >= max_objs:
            break

    results = []
    for c, src_mask in picked:
        M = cv2.moments(c)
        if M["m00"] > 1e-6:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w // 2, y + h // 2
        results.append((c, (cx, cy), src_mask))

    return results


def load_coeffs_txt(path: str) -> dict:
    d: Dict[str, float] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, v = line.split("=", 1)
            d[k.strip()] = float(v.strip())
    for k in ["d_near_px", "d_far_px", "corr_near", "corr_far", "marker_length_cm"]:
        if k not in d:
            raise ValueError(f"Falta '{k}' en {path}")
    return d


def corr_from_depth(depth_px: float, d_near: float, d_far: float, c_near: float, c_far: float) -> Tuple[float, float, float]:
    """Devuelve (corr, t_raw, t_clamped)."""
    denom = (d_far - d_near)
    if abs(denom) < 1e-9:
        t = 0.0
    else:
        t = (depth_px - d_near) / denom
    tc = clamp01(t)
    corr = lerp(c_near, c_far, tc)
    return corr, t, tc


def normalize_depth_sign(depth_px: float, d_near: float, d_far: float) -> float:
    """Alinea el signo de depth_px con el dominio usado en calibración."""
    calib_mid = 0.5 * (d_near + d_far)
    if calib_mid < 0 and depth_px > 0:
        return -depth_px
    if calib_mid > 0 and depth_px < 0:
        return -depth_px
    return depth_px


def point_in_contour(contour: np.ndarray, point: Tuple[int, int]) -> bool:
    """Verifica si un punto está dentro de un contorno."""
    result = cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False)
    return result >= 0


def find_matching_object_cenital(
    objs_cenital: List[Tuple[np.ndarray, Tuple[int, int], np.ndarray]],
    coordenadas: Tuple[int, int]
) -> Optional[int]:
    """
    Encuentra el índice del objeto cenital cuyo contorno contiene las coordenadas dadas.
    Devuelve None si ninguno contiene el punto.
    """
    for idx, (cnt, center, mask) in enumerate(objs_cenital):
        if point_in_contour(cnt, coordenadas):
            return idx
    return None


def camara_horizontal(
    img_cenital: np.ndarray,
    img_horizontal: np.ndarray,
    coordenadas_objeto: Tuple[int, int],
    coeffs_path: str = "debug_out/depth_correction_coeffs.txt",
    aruco_dict: str = "DICT_6X6_50",
    max_objs: int = 5,
    export_dir: str = "imagenes_lateral",
    calib_cenital_path: str = "calibracion_camara_cenital.pkl",
    calib_horizontal_path: str = "calibracion_camara_horizontal.pkl"
) -> Optional[float]:
    """
    Función de caja negra que calcula la altura de un objeto.

    Parámetros:
        img_cenital: Imagen cenital (numpy array BGR de OpenCV)
        img_horizontal: Imagen horizontal/lateral (numpy array BGR de OpenCV)
        coordenadas_objeto: Tupla (x, y) con las coordenadas del objeto en la imagen cenital
        coeffs_path: Ruta al archivo de coeficientes de corrección
        aruco_dict: Diccionario ArUco a usar
        max_objs: Número máximo de objetos a detectar
        export_dir: Carpeta donde exportar las imágenes procesadas
        calib_cenital_path: Ruta al archivo .pkl de calibración cenital
        calib_horizontal_path: Ruta al archivo .pkl de calibración horizontal

    Devuelve:
        Altura del objeto en milímetros (float), o None si no se puede calcular.
    """
    # Aplicar calibración cenital
    roi_offset_cenital = (0, 0)  # Offset del ROI cenital (x, y)
    if os.path.exists(calib_cenital_path):
        try:
            with open(calib_cenital_path, 'rb') as f:
                calib_data = pickle.load(f)
            mtx = calib_data['camera_matrix']
            dist = calib_data['dist_coeff']
            h, w = img_cenital.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            img_cenital = cv2.undistort(img_cenital, mtx, dist, None, newcameramtx)
            # Recortar ROI para eliminar bordes deformados
            x, y, w_roi, h_roi = roi
            img_cenital = img_cenital[y:y+h_roi, x:x+w_roi]
            roi_offset_cenital = (x, y)
            # Ajustar coordenadas del objeto al nuevo ROI
            coordenadas_objeto = (coordenadas_objeto[0] - x, coordenadas_objeto[1] - y)
            print(f"[INFO] Calibración cenital aplicada desde {calib_cenital_path}")
            print(f"[INFO] ROI cenital offset: {roi_offset_cenital}, coordenadas ajustadas: {coordenadas_objeto}")
        except Exception as e:
            print(f"[WARNING] No se pudo aplicar calibración cenital: {e}")
    else:
        print(f"[WARNING] No se encontró archivo de calibración cenital: {calib_cenital_path}")
    
    # Aplicar calibración horizontal
    if os.path.exists(calib_horizontal_path):
        try:
            with open(calib_horizontal_path, 'rb') as f:
                calib_data = pickle.load(f)
            mtx = calib_data['camera_matrix']
            dist = calib_data['dist_coeff']
            h, w = img_horizontal.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            img_horizontal = cv2.undistort(img_horizontal, mtx, dist, None, newcameramtx)
            # Recortar ROI para eliminar bordes deformados
            x, y, w_roi, h_roi = roi
            img_horizontal = img_horizontal[y:y+h_roi, x:x+w_roi]
            print(f"[INFO] Calibración horizontal aplicada desde {calib_horizontal_path}")
        except Exception as e:
            print(f"[WARNING] No se pudo aplicar calibración horizontal: {e}")
    else:
        print(f"[WARNING] No se encontró archivo de calibración horizontal: {calib_horizontal_path}")
    
    # Cargar coeficientes
    coeffs = load_coeffs_txt(coeffs_path)
    d_near = coeffs["d_near_px"]
    d_far = coeffs["d_far_px"]
    c_near = coeffs["corr_near"]
    c_far = coeffs["corr_far"]
    marker_len_cm = coeffs["marker_length_cm"]

    # --- CENITAL ---
    det_c = detect_aruco(img_cenital, dict_name=aruco_dict)
    if det_c is None:
        raise RuntimeError("No se detectó ArUco en imagen cenital.")
    mk_c, mkid_c = select_marker_by_max_perimeter(det_c)
    mkcx_c, mkcy_c = marker_center(mk_c)

    objs_c = detect_objects_cenital(img_cenital, max_objs=max_objs)
    
    # Exportar imagen cenital con detecciones SIEMPRE (antes de cualquier error)
    os.makedirs(export_dir, exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    img_cenital_export = img_cenital.copy()
    if objs_c:
        for idx, (cnt_obj, (cx_obj, cy_obj), _) in enumerate(objs_c):
            cv2.drawContours(img_cenital_export, [cnt_obj], -1, (255, 128, 0), 2)
            cv2.putText(img_cenital_export, f"Obj#{idx+1}", (cx_obj - 30, cy_obj - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
    cv2.circle(img_cenital_export, coordenadas_objeto, 8, (0, 0, 255), -1)
    cv2.circle(img_cenital_export, (mkcx_c, mkcy_c), 10, (255, 0, 0), 2)
    cenital_out_path = os.path.join(export_dir, f"cenital_{timestamp}.jpg")
    cv2.imwrite(cenital_out_path, img_cenital_export)
    
    if not objs_c:
        raise RuntimeError("No se detectaron objetos en imagen cenital.")

    # Encontrar qué objeto cenital corresponde a las coordenadas dadas
    obj_idx = find_matching_object_cenital(objs_c, coordenadas_objeto)
    if obj_idx is None:
        raise RuntimeError(f"Las coordenadas {coordenadas_objeto} no están dentro de ningún objeto detectado.")

    # Calcular depth y corrección para el objeto encontrado
    cnt_c, (ocx_c, ocy_c), mask_c = objs_c[obj_idx]
    depth_px = float(ocy_c - mkcy_c)
    depth_px = normalize_depth_sign(depth_px, d_near, d_far)
    corr, t_raw, t_clamped = corr_from_depth(depth_px, d_near, d_far, c_near, c_far)

    # --- LATERAL ---
    det_l = detect_aruco(img_horizontal, dict_name=aruco_dict)
    if det_l is None:
        raise RuntimeError("No se detectó ArUco en imagen horizontal.")
    mk_l, mkid_l = select_marker_by_max_perimeter(det_l)
    mkcx_l, mkcy_l = marker_center(mk_l)
    pxcm = px_per_cm_from_marker(mk_l, marker_len_cm)

    objs_l = detect_objects_lateral(img_horizontal, det_l, max_objs=max_objs + 2)
    if not objs_l:
        raise RuntimeError("No se detectaron objetos en imagen horizontal.")

    # Calcular distancia X normalizada del objeto seleccionado en cenital respecto al ArUco
    # En cenital: distancia horizontal (X) del centro del objeto al centro del ArUco
    dist_x_cenital = ocx_c - mkcx_c  # Distancia con signo (negativo = izq del aruco, positivo = der)
    
    # Calcular px/cm en cenital para normalizar
    pxcm_cenital = px_per_cm_from_marker(mk_c, marker_len_cm)
    dist_cm_cenital = dist_x_cenital / max(pxcm_cenital, 1e-9)
    
    # Debug: Imprimir información de emparejamiento
    print(f"\n=== EMPAREJAMIENTO CENITAL-LATERAL ===")
    print(f"Cenital - Obj#{obj_idx+1}: pos_x={ocx_c}, ArUco_x={mkcx_c}, dist_x={dist_x_cenital:.1f}px, dist={dist_cm_cenital:.2f}cm")
    print(f"Lateral - ArUco_x={mkcx_l}")
    
    # Para cada objeto en lateral, calcular su distancia X al ArUco en cm
    # y encontrar el que tenga distancia más similar
    # IMPORTANTE: Las vistas pueden estar invertidas, comparamos por distancia absoluta
    best_lateral_idx = None
    best_dist_diff = float('inf')
    
    for idx_l, (cnt_l_cand, (cx_l, cy_l), mask_l_cand) in enumerate(objs_l):
        dist_x_lateral = cx_l - mkcx_l
        dist_cm_lateral = dist_x_lateral / max(pxcm, 1e-9)
        
        # Comparar con signo invertido (las vistas pueden estar opuestas)
        dist_diff_normal = abs(dist_cm_cenital - dist_cm_lateral)
        dist_diff_inverted = abs(dist_cm_cenital - (-dist_cm_lateral))
        dist_diff = min(dist_diff_normal, dist_diff_inverted)
        
        print(f"  Lateral Obj#{idx_l+1}: pos_x={cx_l}, dist_x={dist_x_lateral:.1f}px, dist={dist_cm_lateral:.2f}cm, diff={dist_diff:.2f}cm")
        
        if dist_diff < best_dist_diff:
            best_dist_diff = dist_diff
            best_lateral_idx = idx_l
    
    print(
    f"Mejor match: Lateral Obj#{best_lateral_idx+1} | "
    f"depth_px={depth_px:.2f} | "
    f"t={t_raw:.3f} (clamped={t_clamped:.3f}) | "
    f"corr={corr:.6f} | "
    f"diff={best_dist_diff:.2f}cm\n"
)

    
    if best_lateral_idx is None:
        raise RuntimeError("No se pudo emparejar el objeto cenital con ningún objeto lateral.")

    cnt_l, center_l, mask_l = objs_l[best_lateral_idx]
    hpx = height_px_from_contour(cnt_l)
    raw_h_cm = float(hpx / max(pxcm, 1e-9))
    final_h_cm = raw_h_cm * corr

    # Convertir a mm
    altura_mm = final_h_cm * 10.0

    # Actualizar imagen cenital exportada con el objeto seleccionado resaltado
    img_cenital_export = img_cenital.copy()
    for idx, (cnt_obj, (cx_obj, cy_obj), _) in enumerate(objs_c):
        if idx == obj_idx:
            cv2.drawContours(img_cenital_export, [cnt_obj], -1, (0, 255, 0), 3)
            cv2.putText(img_cenital_export, f"Obj#{idx+1}*", (cx_obj - 30, cy_obj - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.drawContours(img_cenital_export, [cnt_obj], -1, (255, 128, 0), 2)
            cv2.putText(img_cenital_export, f"Obj#{idx+1}", (cx_obj - 30, cy_obj - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
    cv2.circle(img_cenital_export, coordenadas_objeto, 8, (0, 0, 255), -1)
    cv2.circle(img_cenital_export, (mkcx_c, mkcy_c), 10, (255, 0, 0), 2)
    cenital_out_path = os.path.join(export_dir, f"cenital_{timestamp}.jpg")
    cv2.imwrite(cenital_out_path, img_cenital_export)
    
    # Imagen horizontal con TODOS los objetos detectados
    img_horizontal_export = img_horizontal.copy()
    for idx_l, (cnt_obj_l, (cx_l_obj, cy_l_obj), _) in enumerate(objs_l):
        if idx_l == best_lateral_idx:
            cv2.drawContours(img_horizontal_export, [cnt_obj_l], -1, (0, 255, 0), 3)
            cv2.circle(img_horizontal_export, (cx_l_obj, cy_l_obj), 8, (0, 255, 255), -1)
            cv2.putText(img_horizontal_export, f"H={altura_mm:.1f}mm", (cx_l_obj - 50, cy_l_obj - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.drawContours(img_horizontal_export, [cnt_obj_l], -1, (255, 128, 0), 2)
            cv2.putText(img_horizontal_export, f"Obj#{idx_l+1}", (cx_l_obj - 30, cy_l_obj - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
    cv2.circle(img_horizontal_export, (mkcx_l, mkcy_l), 10, (255, 0, 0), 2)
    horizontal_out_path = os.path.join(export_dir, f"horizontal_{timestamp}.jpg")
    cv2.imwrite(horizontal_out_path, img_horizontal_export)

    return altura_mm
