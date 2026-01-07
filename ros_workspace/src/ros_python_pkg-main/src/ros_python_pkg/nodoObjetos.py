#!/usr/bin/env python3
#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
import sys

#!/usr/bin/env python3
import os
import sys
import json
import cv2
from copy import deepcopy
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# --- CONFIGURACIÓN DE RUTAS Y PARÁMETROS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGENES_DIR = os.path.join(SCRIPT_DIR, "imagenes")
PUNTOS_DIR = os.path.join(SCRIPT_DIR, "puntos")
CALIB_FILE = os.path.join(SCRIPT_DIR, "clasificacion", "calibracion_camara_cenital.pkl")
CAPTURA_PATH = os.path.join(IMAGENES_DIR, "captura_cenital.png")
RESULT_JSON = os.path.join(PUNTOS_DIR, "resultados_mm.json")

class NodoObjetos:
    def __init__(self):
        rospy.init_node('nodo_objetos', anonymous=False)
        self.bridge = CvBridge()
        self.img_cam1 = None
        self.procesando = False
        rospy.Subscriber('/cam1/usb_cam1/image_raw', Image, self._image_cb, queue_size=1)
        rospy.Subscriber('nc_gestos', String, self._gesto_cb, queue_size=10)
        self.pub_coordenadas = rospy.Publisher('coordenadas_objetos', String, queue_size=10)
        rospy.loginfo("NodoObjetos: Nodo iniciado y esperando gestos e imagen de cámara...")

    def _image_cb(self, msg):
        try:
            self.img_cam1 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"NodoObjetos: Error al convertir imagen: {e}")

    def _gesto_cb(self, msg):
        gesto = msg.data.strip().lower()
        if self.procesando:
            rospy.logwarn("NodoObjetos: Pipeline ya en ejecución, ignorando gesto...")
            return
        if gesto == "nada":
            rospy.loginfo("NodoObjetos: Gesto 'Nada' recibido, ignorando...")
            return
        if self.img_cam1 is None:
            rospy.logwarn("NodoObjetos: No hay imagen de cámara disponible aún.")
            return
        self.procesando = True
        rospy.loginfo(f"NodoObjetos: Recibido gesto '{gesto}', ejecutando pipeline...")
        os.makedirs(IMAGENES_DIR, exist_ok=True)
        # --- Calibrar y recortar imagen como en modulo_vision.py ---
        try:
            import pickle
            # Cargar calibración
            if not os.path.exists(CALIB_FILE):
                rospy.logerr(f"NodoObjetos: No se encuentra el archivo de calibración: {CALIB_FILE}")
                self.procesando = False
                return
            with open(CALIB_FILE, "rb") as f:
                data = pickle.load(f)
            mtx = data["camera_matrix"]
            dist = data["dist_coeff"]
            frame = deepcopy(self.img_cam1)
            h, w = frame.shape[:2]
            # Calcular mapas de distorsión y ROI
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
            undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w_roi, h_roi = roi
            undistorted_cropped = undistorted[y:y+h_roi, x:x+w_roi]
            cv2.imwrite(CAPTURA_PATH, undistorted_cropped)
        except Exception as e:
            rospy.logerr(f"NodoObjetos: Error en calibración/recorte de imagen: {e}")
            self.procesando = False
            return
        # Ejecutar pipeline completo usando el gesto como filtro
        try:
            self.run_pipeline(gesto)
        finally:
            self.procesando = False

    def run_pipeline(self, gesto):
        # Mapear gesto a filtro de clasificador (como en modulo_vision)
        def map_to_classifier_filter(residue):
            return "plastico" if residue == "botella" else residue

        filtro = map_to_classifier_filter(gesto)
        clasificacion_dir = os.path.join(SCRIPT_DIR, "clasificacion")
        clasificador_script = os.path.join(clasificacion_dir, "clasificador_imagenes.py")
        clasif_json = os.path.join(PUNTOS_DIR, "clasificacion.json")
        cmd_clasif = [sys.executable, clasificador_script, "--image", CAPTURA_PATH, "--filter-class", filtro, "--json-results", clasif_json]
        try:
            rospy.loginfo(f"NodoObjetos: Ejecutando clasificador_imagenes.py con filtro '{filtro}'...")
            import subprocess
            subprocess.run(cmd_clasif, check=True)
        except Exception as e:
            rospy.logerr(f"NodoObjetos: Error en clasificación: {e}")
            return

        # 2. DETECCIÓN Y MEDICIÓN (llama aruco_batch_measure.py como en modulo_vision)
        deteccion_dir = os.path.join(SCRIPT_DIR, "deteccion", "funciones")
        aruco_script = os.path.join(deteccion_dir, "aruco_batch_measure.py")
        cmd_aruco = [sys.executable, aruco_script, "--image", CAPTURA_PATH, "--points-json", clasif_json, "--aruco-mm", "50", "--aruco-tamano", "AUTO"]
        try:
            rospy.loginfo("NodoObjetos: Ejecutando aruco_batch_measure.py...")
            import subprocess
            subprocess.run(cmd_aruco, check=True)
        except Exception as e:
            rospy.logerr(f"NodoObjetos: Error en medición/detección: {e}")
            return

        # 3. Publicar el JSON final si existe
        if os.path.exists(RESULT_JSON):
            with open(RESULT_JSON, "r", encoding="utf-8") as f:
                resultados = json.load(f)
            self.pub_coordenadas.publish(String(data=json.dumps(resultados)))
            rospy.loginfo("NodoObjetos: Coordenadas publicadas correctamente")
        else:
            rospy.logwarn("NodoObjetos: No se encontró el JSON de resultados para publicar.")

    def start(self):
        rospy.spin()

if __name__ == "__main__":
    node = NodoObjetos()
    node.start()

