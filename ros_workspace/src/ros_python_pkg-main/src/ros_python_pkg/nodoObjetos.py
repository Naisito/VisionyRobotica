#!/usr/bin/env python3
import os
import sys
import json
import cv2
from copy import deepcopy
from typing import List, Dict, Any
import time

import rospy
from std_msgs.msg import String, Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from placeholder import run_pipeline

# --- CONFIGURACIÓN DE RUTAS Y PARÁMETROS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.join("/","home","laboratorio","ros_workspace")
IMAGENES_DIR = os.path.join(SCRIPT_DIR, "imagenes")
PUNTOS_DIR = os.path.join(SCRIPT_DIR, "puntos")
CALIB_FILE = os.path.join(SCRIPT_DIR, "clasificacion", "calibracion_camara_cenital.pkl")
CAPTURA_PATH = os.path.join(IMAGENES_DIR, "captura_cenital.png")
RESULT_JSON = os.path.join(PUNTOS_DIR, "resultados_mm.json")


def leer_coordenadas_objetos() -> List[Dict[str, float]]:
    """
    Lee resultados_mm.json y devuelve una lista con las coordenadas
    x_mm, y_mm del punto central y la altura de cada objeto.
    
    Returns:
        Lista de diccionarios con 'id', 'x', 'y', 'altura' para cada objeto.
        Ejemplo: [{'id': 1, 'x': -285.16, 'y': -289.7, 'altura': 120.5}, ...]
    """
    if not os.path.exists(RESULT_JSON):
        rospy.logwarn(f"Archivo no encontrado: {RESULT_JSON}")
        return []
    
    try:
        with open(RESULT_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        rospy.logerr(f"Error leyendo {RESULT_JSON}: {e}")
        return []
    
    coordenadas = []
    for obj in data.get("objetos", []):
        punto_central = obj.get("punto_central", {})
        x_mm = punto_central.get("x_mm")
        y_mm = punto_central.get("y_mm")
        altura_mm = obj.get("altura_mm", 0.0)
        
        if x_mm is not None and y_mm is not None:
            coordenadas.append({
                'id': obj.get('id'),
                'x': x_mm,
                'y': y_mm,
                'altura': altura_mm
            })
    
    return coordenadas


def coordenadas_to_array(coordenadas: List[Dict[str, float]]) -> Float32MultiArray:
    """
    Convierte la lista de coordenadas a Float32MultiArray para publicar en ROS.
    
    Formato del array: [id1, x1, y1, altura1, id2, x2, y2, altura2, ...]
    Cada objeto ocupa 4 posiciones consecutivas.
    """
    msg = Float32MultiArray()
    
    # Configurar dimensiones del array
    msg.layout.dim.append(MultiArrayDimension())
    msg.layout.dim[0].label = "objetos"
    msg.layout.dim[0].size = len(coordenadas)
    msg.layout.dim[0].stride = len(coordenadas) * 4
    
    msg.layout.dim.append(MultiArrayDimension())
    msg.layout.dim[1].label = "campos"  # id, x, y, altura
    msg.layout.dim[1].size = 4
    msg.layout.dim[1].stride = 4
    
    # Llenar datos: [id, x, y, altura] por cada objeto
    data = []
    for coord in coordenadas:
        data.extend([
            float(coord.get('id', 0)),
            float(coord.get('x', 0)),
            float(coord.get('y', 0)),
            float(coord.get('altura', 0))
        ])
    
    msg.data = data
    return msg


class NodoObjetos:
    def __init__(self):
        rospy.init_node('nodo_objetos', anonymous=False)
        self.bridge = CvBridge()
        self.img_cam1 = None  # Cámara cenital
        self.img_cam2 = None  # Cámara horizontal
        self.procesando = False
        self.tiempo_inicio = rospy.Time.now()  # Tiempo de inicio del nodo
        
        # Suscriptores
        rospy.Subscriber('/cam1/usb_cam1/image_raw', Image, self._image_cb_cam1, queue_size=1)
        rospy.wait_for_message('/cam1/usb_cam1/image_raw', Image)
        
        # Suscriptor para cámara horizontal (cam2) - opcional
        try:
            rospy.Subscriber('/cam2/usb_cam1/image_raw', Image, self._image_cb_cam2, queue_size=1)
            rospy.wait_for_message('/cam2/usb_cam1/image_raw', Image)
            rospy.loginfo("NodoObjetos: Suscrito a cámara horizontal (cam2)")
        except Exception as e:
            rospy.logwarn(f"NodoObjetos: No se pudo suscribir a cam2: {e}")
        
        rospy.Subscriber('nc_gestos', String, self.gesto_cb, queue_size=10)
        
        # Publicador de coordenadas como Float32MultiArray
        self.pub_coordenadas = rospy.Publisher('/coordenadas_objetos', Float32MultiArray, queue_size=10)
        
        # Esperar 5 segundos para estabilizar las cámaras
        rospy.loginfo("NodoObjetos: Esperando 5 segundos para estabilizar cámaras...")
        rospy.sleep(5.0)
        rospy.loginfo("NodoObjetos: Nodo iniciado y esperando gestos e imagen de cámara...")

    def _image_cb_cam1(self, msg):

        try:
            self.img_cam1 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"NodoObjetos: Error al convertir imagen cam1: {e}")
    
    def _image_cb_cam2(self, msg):

        try:
            self.img_cam2 = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"NodoObjetos: Error al convertir imagen cam2: {e}")

    def gesto_cb(self, msg: String) -> None:
        # Verificar que hayan pasado 5 segundos antes de procesar
        tiempo_transcurrido = (rospy.Time.now() - self.tiempo_inicio).to_sec()
        if tiempo_transcurrido < 5.0:
            rospy.logwarn(f"NodoObjetos: Esperando estabilización de cámaras ({tiempo_transcurrido:.1f}/5.0 s)")
            return
        
        if self.procesando:
            rospy.logwarn("NodoObjetos: Ya hay un procesamiento en curso, ignorando gesto.")
            return        
        
        self.procesando = True
        try:
            residue_type = msg.data.lower().strip()
            
            # Esperar 2 segundos antes de capturar para que el objeto esté quieto
            rospy.loginfo("NodoObjetos: Esperando 2 segundos antes de capturar...")
            rospy.sleep(2.0)
            
            # Capturar imagen FRESCA de la cámara cenital
            rospy.loginfo("NodoObjetos: Capturando imagen cenital...")
            try:
                img_msg_cenital = rospy.wait_for_message('/cam1/usb_cam1/image_raw', Image, timeout=2.0)
                imagen_cenital = self.bridge.imgmsg_to_cv2(img_msg_cenital, desired_encoding='bgr8')
            except Exception as e:
                rospy.logerr(f"NodoObjetos: Error capturando imagen cenital: {e}")
                return
            
            # Capturar imagen FRESCA de la cámara horizontal (opcional)
            imagen_horizontal = None
            try:
                rospy.loginfo("NodoObjetos: Capturando imagen horizontal...")
                img_msg_horizontal = rospy.wait_for_message('/cam2/usb_cam1/image_raw', Image, timeout=2.0)
                imagen_horizontal = self.bridge.imgmsg_to_cv2(img_msg_horizontal, desired_encoding='bgr8')
            except Exception as e:
                rospy.logwarn(f"NodoObjetos: No se pudo capturar imagen horizontal: {e}")
            
            # Mapear gestos a tipos de residuo
            if residue_type not in ["lata", "carton", "botella"]:
                rospy.logwarn(f"NodoObjetos: Tipo de residuo desconocido: '{residue_type}'")
                return
            
            rospy.loginfo(f"NodoObjetos: Procesando gesto '{residue_type}'...")
            
            # Llamar al pipeline con ambas imágenes
            if imagen_horizontal is not None:
                rospy.loginfo("NodoObjetos: Usando cámara horizontal para cálculo de altura")
            else:
                rospy.logwarn("NodoObjetos: Cámara horizontal no disponible, altura será 0.0")
            
            run_pipeline(imagen_cenital, residue_type, img_horizontal=imagen_horizontal)
            
            # Leer coordenadas del JSON y publicar
            coordenadas = leer_coordenadas_objetos()
            
            # DEBUG: Verificar cuántas coordenadas se leyeron
            rospy.loginfo(f"[DEBUG nodoObjetos] Se leyeron {len(coordenadas)} coordenadas del JSON")
            
            if coordenadas:
                msg_array = coordenadas_to_array(coordenadas)
                self.pub_coordenadas.publish(msg_array)
                rospy.loginfo(f"NodoObjetos: Publicadas {len(coordenadas)} coordenadas de objetos.")
                
                # Log de coordenadas
                for coord in coordenadas:
                    rospy.loginfo(f"  Objeto ID={coord.get('id')}: x={coord.get('x'):.2f} mm, y={coord.get('y'):.2f} mm, altura={coord.get('altura'):.2f} mm")
            else:
                rospy.logwarn("NodoObjetos: No se encontraron objetos para publicar.")
        finally:
            self.procesando = False

    def start(self):
        rospy.spin()


if __name__ == "__main__":
    node = NodoObjetos()
    # coord_dict = run_pipeline(node.img_cam1, "carton")
    # while True:
    #     node.gesto_cb(String(data="carton"))
    node.start()

