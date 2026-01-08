#!/usr/bin/env python3
"""
Script de prueba para el pipeline de detección.
Carga una imagen y ejecuta el pipeline sin necesidad de ROS.
"""
import os
import sys
import cv2

# Asegurar que el directorio actual esté en el path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from placeholder import run_pipeline

# Configuración
IMAGENES_DIR = os.path.join(SCRIPT_DIR, "imagenes")
IMAGEN_TEST = os.path.join(IMAGENES_DIR, "test.png")

def main():
    # Verificar que existe la imagen
    if not os.path.exists(IMAGEN_TEST):
        print(f"Error: No se encontró la imagen {IMAGEN_TEST}")
        print(f"Coloca una imagen de prueba en: {IMAGEN_TEST}")
        return
    
    # Cargar imagen
    img = cv2.imread(IMAGEN_TEST)
    if img is None:
        print(f"Error: No se pudo leer la imagen {IMAGEN_TEST}")
        return
    
    print(f"Imagen cargada: {IMAGEN_TEST}")
    print(f"Tamaño: {img.shape[1]}x{img.shape[0]}")
    
    # Ejecutar pipeline
    print("\n--- Ejecutando pipeline con 'carton' ---")
    resultado = run_pipeline(img, "carton")
    
    if resultado:
        objetos = resultado.get('objetos', [])
        print(f"\nResultado: {len(objetos)} objetos detectados")
        for obj in objetos:
            print(f"  - ID {obj.get('id')}: {obj.get('punto_central')}")
    else:
        print("\nNo se detectaron objetos de tipo 'carton'")

if __name__ == "__main__":
    main()
