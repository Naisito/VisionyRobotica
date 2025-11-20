#!/usr/bin/env python3
"""nodoGestos.py

Nodo ROS que se suscribe a un tópico de cámara (sensor_msgs/Image), realiza
un procesamiento simple (movimiento / oscuridad / color dominante) y publica
una etiqueta como std_msgs/String para que el `central_listener` la reciba.

Parámetros ROS (namespace privado):
  ~image_topic (string): tópico de imagen a suscribir (por defecto '/usb_cam/image_raw')
  ~output_topic (string): tópico donde publicar la etiqueta (por defecto '/nodo_gestos/label')
  ~min_brightness (float): umbral para considerar la imagen 'oscura'
  ~motion_threshold (int): umbral por pixel para detectar movimiento
  ~motion_area_fraction (float): fracción mínima de píxeles en movimiento para reportar 'movimiento'
"""

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from numpy import ndarray
from copy import deepcopy
from time import time, sleep

class NodoGestos:
    def __init__(self) -> None:
        rospy.init_node('nodo_gestos', anonymous=False)

        # # Parámetros
        # self.image_topic = rospy.get_param('~image_topic', '/usb_cam/image_raw')
        # self.output_topic = rospy.get_param('~output_topic', '/nodo_gestos/label')
        # self.min_brightness = rospy.get_param('~min_brightness', 30.0)
        # self.motion_threshold = rospy.get_param('~motion_threshold', 25)
        # self.motion_area_fraction = rospy.get_param('~motion_area_fraction', 0.01)

        # Publicador y bridge
        self.pub = rospy.Publisher("gesto", String, queue_size=10)
        self.bridge = CvBridge()

        # Suscriptor a la cámara
        rospy.Subscriber('/usb_cam/image_raw', Image, self._image_cb, queue_size=1)
        self.img = None
        rospy.wait_for_message('/usb_cam/image_raw', Image)
        
        # Estado para detección de movimiento
        self._prev_gray = None

    def _image_cb(self, msg: Image) -> None:
        self.img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
    def _process_image(self, img: ndarray) -> str:
        # Logica opencv
        pass
    
    def procesar_imagenes(self, img: ndarray, tiempo: float=2) -> str:
        inicio = time()
        resultados = []
        while time() - inicio < tiempo:
            resultados.append(self._process_image(img))
            sleep(0.03)
            
        return resultados
    
    def start(self) -> None:
        while not rospy.is_shutdown():            
            imagen = deepcopy(self.img)
            self.img = None
            resultado = self._process_image(imagen)
            self.pub.publish(String(data=resultado))


if __name__ == '__main__':
    node = NodoGestos()
    node.start()
