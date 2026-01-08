#!/usr/bin/env python3
"""
Ejecuta el pipeline con una imagen sin necesidad de ROS ni cámara.
Uso: python3 test_imagen.py <ruta_imagen> <tipo_residuo>
Ejemplo: python3 test_imagen.py imagenes/foto.png carton
"""
import os
import sys
import cv2

# Configurar path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from placeholder import run_pipeline

def main():
    # Valores por defecto
    imagen_path = "imagenes/test.png"
    tipo_residuo = "carton"
    
    # Argumentos de línea de comandos
    if len(sys.argv) >= 2:
        imagen_path = sys.argv[1]
    if len(sys.argv) >= 3:
        tipo_residuo = sys.argv[2]
    
    # Verificar tipo válido
    if tipo_residuo not in ["lata", "carton", "botella"]:
        print(f"Error: tipo debe ser 'lata', 'carton' o 'botella'")
        return
    
    # Cargar imagen
    if not os.path.exists(imagen_path):
        print(f"Error: No existe {imagen_path}")
        return
    
    img = cv2.imread(imagen_path)
    if img is None:
        print(f"Error: No se pudo leer {imagen_path}")
        return
    
    print(f"Imagen: {imagen_path} ({img.shape[1]}x{img.shape[0]})")
    print(f"Buscando: {tipo_residuo}")
    print("-" * 40)
    
    # Ejecutar pipeline
    resultado = run_pipeline(img, tipo_residuo)
    
    if resultado:
        objetos = resultado.get('objetos', [])
        print(f"\n✓ Detectados {len(objetos)} objetos")
        for obj in objetos:
            pc = obj.get('punto_central', {})
            print(f"  ID {obj.get('id')}: ({pc.get('x')}, {pc.get('y')})")
        print(f"\nArchivos generados:")
        print(f"  - imagenes/resultado_final.png")
        print(f"  - puntos/resultados_mm.json")
    else:
        print(f"\n✗ No se encontraron objetos de tipo '{tipo_residuo}'")

if __name__ == "__main__":
    main()
