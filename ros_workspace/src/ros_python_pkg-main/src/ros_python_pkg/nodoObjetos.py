#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from numpy import ndarray
from copy import deepcopy
from time import time, sleep

class NodoObjetos:
    def __init__(self) -> None:
        rospy.init_node('nodo_objetos', anonymous=False)

        # Publicador y bridge
        self.pub = rospy.Publisher("objetos", String, queue_size=10)
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
        resultado = "HOLA"
        #resultados = []
        
        #while time() - inicio < tiempo:
        #    resultados.append(self._process_image(img))
        #    sleep(0.03)
            
        return resultado
    
    def start(self) -> None:
        while not rospy.is_shutdown():
            while self.img is None:
                sleep(0.01)
            imagen = deepcopy(self.img)
            self.img = None
            # resultado = self._process_image(imagen)
            resultado = self.procesar_imagenes(imagen)
            self.pub.publish(String(data=resultado))


if __name__ == '__main__':
    node = NodoGestos()
    node.start()
