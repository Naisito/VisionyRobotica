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
from placeholder import run_pipeline

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
        rospy.wait_for_message('/cam1/usb_cam1/image_raw', Image)
        rospy.Subscriber('nc_gestos', String, self._gesto_cb, queue_size=10)
        self.pub_coordenadas = rospy.Publisher('coordenadas_objetos', String, queue_size=10)
        rospy.loginfo("NodoObjetos: Nodo iniciado y esperando gestos e imagen de cámara...")

    def _image_cb(self, msg):
        try:
            self.img_cam1 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"NodoObjetos: Error al convertir imagen: {e}")

    def _gesto_cb(self, msg: String) -> None:
        imagen = deepcopy(self.img_cam1)
        # coord_dict = run_pipeline(imagen, msg.data)
        coord_dict = run_pipeline(imagen, "carton")
        self.pub_coordenadas.publish(json.dumps(coord_dict))

    def start(self):
        rospy.spin()

if __name__ == "__main__":
    node = NodoObjetos()    
    while True:
        coord_dict = run_pipeline(node.img_cam1, "carton")
    node.start()

