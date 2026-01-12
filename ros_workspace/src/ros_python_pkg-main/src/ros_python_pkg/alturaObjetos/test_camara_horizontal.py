#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_camara_horizontal.py

Script de test para la función camara_horizontal.
Carga dos imágenes y unas coordenadas, calcula la altura y la imprime en mm.
"""

import os
import cv2
from camara_horizontal import camara_horizontal


def altura_objetos(x, y, img_cenital=None, img_horizontal=None, cenital_path=None, horizontal_path=None):
    """
    Calcula la altura de un objeto en mm usando dos cámaras (cenital y horizontal).
    
    Args:
        x (int): Coordenada x del objeto en la imagen cenital (píxeles)
        y (int): Coordenada y del objeto en la imagen cenital (píxeles)
        img_cenital (numpy.ndarray, optional): Imagen cenital BGR. Si None, se carga desde cenital_path
        img_horizontal (numpy.ndarray, optional): Imagen horizontal BGR. Si None, se carga desde horizontal_path
        cenital_path (str, optional): Ruta a la imagen cenital. Default: "test_1_cen.jpg"
        horizontal_path (str, optional): Ruta a la imagen horizontal. Default: "test_1_lat.jpg"
    
    Returns:
        float: Altura del objeto en milímetros, o 0.0 si hay error
    """
    # Rutas por defecto
    if cenital_path is None:
        cenital_path = os.path.join(os.path.dirname(__file__), "test_1_cen.jpg")
    if horizontal_path is None:
        horizontal_path = os.path.join(os.path.dirname(__file__), "test_1_lat.jpg")
    
    # Coordenadas del objeto en la imagen cenital (x, y)
    coordenadas_objeto = (x, y)
    
    # Ruta al archivo de coeficientes
    coeffs_path = os.path.join(os.path.dirname(__file__), "debug_out", "depth_correction_coeffs.txt")
    
    # Cargar imágenes si no se proporcionaron
    if img_cenital is None:
        img_cenital = cv2.imread(cenital_path)
        if img_cenital is None:
            print(f"Error: No se pudo cargar la imagen cenital: {cenital_path}")
            return 0.0
    
    if img_horizontal is None:
        img_horizontal = cv2.imread(horizontal_path)
        if img_horizontal is None:
            print(f"Error: No se pudo cargar la imagen horizontal: {horizontal_path}")
            return 0.0
    
    # Llamar a la función de caja negra
    try:
        altura_mm = camara_horizontal(
            img_cenital=img_cenital,
            img_horizontal=img_horizontal,
            coordenadas_objeto=coordenadas_objeto,
            coeffs_path=coeffs_path
        )
        
        # Imprimir solo la altura en mm
        print(f"[INFO] Altura calculada: {altura_mm:.2f} mm")
        return altura_mm
        
    except RuntimeError as e:
        print(f"Error calculando altura: {e}")
        return 0.0


if __name__ == "__main__":
    altura_objetos(600, 270)
