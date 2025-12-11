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
        rospy.Subscriber('/usb_cam1/image_raw', Image, self._image_cb1, queue_size=1)
        self.img1 = None
        rospy.wait_for_message('/usb_cam1/image_raw', Image)
        
        rospy.Subscriber('/usb_cam2/image_raw', Image, self._image_cb2, queue_size=1)
        self.img2 = None
        rospy.wait_for_message('/usb_cam2/image_raw', Image)
        
        rospy.Subscriber('/gesto', String, self._gesto_cb, queue_size=10)
        self.gesto = None
        rospy.wait_for_message('/gesto', String)
        
    def _image_cb1(self, msg: Image) -> None:
        if self.img1 is None: # para guardar solo el primer frame del video
            self.img1 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
    def _image_cb2(self, msg: Image) -> None:
        if self.img2 is None:
            self.img2 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    
    def _gesto_cb(self, msg):
        self.gesto = msg.data     
    
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
            while self.img1 is None or self.img2 is None: # and
                sleep(0.01)
            imagen1 = deepcopy(self.img1)
            imagen2 = deepcopy(self.img2)
            self.img1 = None
            self.img2 = None
            # resultado = self._process_image(imagen)
            resultado = self.procesar_imagenes(imagen1)
            self.pub.publish(String(data=resultado))


if __name__ == '__main__':
    node = NodoObjetos()
    node.start()

