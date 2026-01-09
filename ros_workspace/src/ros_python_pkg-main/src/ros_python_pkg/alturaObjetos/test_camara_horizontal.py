#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_camara_horizontal.py

Script de test para la función camara_horizontal.
Carga dos imágenes y unas coordenadas, calcula la altura y la imprime en mm.
"""

import cv2
from camara_horizontal import camara_horizontal


def altura_objetos(x, y):
    # Rutas a las imágenes (modifica según tus imágenes)
    cenital_path = "test_1_cen.jpg"  # Cambia por tu ruta
    horizontal_path = "test_1_lat.jpg"  # Cambia por tu ruta
    
    # Coordenadas del objeto en la imagen cenital (x, y)
    # Por ejemplo, el punto rojo encima de la lata en la imagen 3
    coordenadas_objeto = (x, y)  # Ajusta según tu caso
    
    # Ruta al archivo de coeficientes
    coeffs_path = "debug_out/depth_correction_coeffs.txt"
    
    # Cargar imágenes con OpenCV
    img_cenital = cv2.imread(cenital_path)
    img_horizontal = cv2.imread(horizontal_path)
    
    if img_cenital is None:
        print(f"Error: No se pudo cargar la imagen cenital: {cenital_path}")
        return
    
    if img_horizontal is None:
        print(f"Error: No se pudo cargar la imagen horizontal: {horizontal_path}")
        return
    
    # Llamar a la función de caja negra
    try:
        altura_mm = camara_horizontal(
            img_cenital=img_cenital,
            img_horizontal=img_horizontal,
            coordenadas_objeto=coordenadas_objeto,
            coeffs_path=coeffs_path
        )
        
        # Imprimir solo la altura en mm
        print(f"{altura_mm:.2f}")
        return altura_mm
        
    except RuntimeError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    altura_objetos(600, 270)
