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

class NodoCentral:
    def __init__(self) -> None:
        rospy.init_node('nodo_central', anonymous=False)
        # Publicador y bridge
        
        self.pub1 = rospy.Publisher("nc_gestos", String, queue_size=10)
                
        self.pub2 = rospy.Publisher("nc_obj", String, queue_size=10)

        #self.pub3 = rospy.Publisher("nc-rob", String, queue_size=10)
        
        # Suscriptor 
        rospy.Subscriber('gesto', String, self.callbackGesto, queue_size=10)
        self.gesto = None
        rospy.wait_for_message('gesto', String)
        
        
    '''
    def procesar_imagenes(self, img: ndarray, tiempo: float=2) -> str:
        inicio = time()
        resultado = "HOLA"
        resultados = []
        
        #while time() - inicio < tiempo:
        #    resultados.append(self._process_image(img))
            sleep(0.03)
            
        return resultado
    '''
    
    def callbackGesto(self,msg: String):
        self.pub1.publish(msg)
        
    def start(self) -> None:
        while not rospy.is_shutdown():
            while self.gesto is None:
                sleep(0.01)
            imagen = deepcopy(self.img)
            self.gesto = None
            # resultado = self._process_image(imagen)
            resultado = 'HOLA'
            self.pub.publish(String(data=resultado))


if __name__ == '__main__':
    node = NodoCentral()
    node.start()