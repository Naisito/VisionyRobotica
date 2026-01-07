# camera_calibration.py
import numpy as np
import yaml
import cv2
import pickle

class CameraCalibration:
    def __init__(self, K, D, image_size=None):
        """
        K: matriz intrínseca 3x3 (np.ndarray)
        D: coeficientes de distorsión (np.ndarray, shape (5,) o (1,5))
        image_size: (width, height) opcional
        """
        self.K = K
        self.D = D
        self.image_size = image_size
        self.map1 = None
        self.map2 = None

    @classmethod
    def from_yaml(cls, path):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        K = np.array(data['camera_matrix']['data'], dtype=np.float32).reshape(3, 3)
        D = np.array(data['distortion_coefficients']['data'], dtype=np.float32).reshape(-1, 1)

        w = data.get('image_width', None)
        h = data.get('image_height', None)
        image_size = (w, h) if (w is not None and h is not None) else None

        return cls(K, D, image_size=image_size)

    @classmethod
    def from_ost_txt(cls, path):
        """
        Para ost.txt típico de ROS (con líneas K: y D:).
        Ajusta este parser si tu formato es distinto.
        """
        K = None
        D = None
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('K:'):
                    # K: fx 0 cx 0 fy cy 0 0 1
                    vals = list(map(float, line.split()[1:]))
                    K = np.array(vals, dtype=np.float32).reshape(3, 3)
                elif line.startswith('D:'):
                    # D: k1 k2 p1 p2 k3
                    vals = list(map(float, line.split()[1:]))
                    D = np.array(vals, dtype=np.float32).reshape(-1, 1)
        if K is None or D is None:
            raise ValueError("No se encontraron K o D en el ost.txt")
        return cls(K, D)

    @classmethod
    def from_pickle(cls, path):
        """
        Carga calibración desde un archivo .pkl (diccionario con 'camera_matrix' y 'dist_coeff').
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        K = data.get("camera_matrix")
        D = data.get("dist_coeff")
        
        if K is None or D is None:
            raise ValueError("El archivo pickle debe contener 'camera_matrix' y 'dist_coeff'")
            
        # Intentar inferir tamaño si está guardado, si no, dejar None
        w = data.get('image_width', None)
        h = data.get('image_height', None)
        image_size = (w, h) if (w is not None and h is not None) else None
        
        return cls(K, D, image_size=image_size)

    def init_undistort_maps(self, image_shape):
        """
        Precalcula los mapas de remapeo para tiempo real.
        image_shape: (alto, ancho, canales) -> frame.shape
        """
        h, w = image_shape[:2]
        self.image_size = (w, h)

        # Puedes usar newCameraMatrix = K para mantener FOV original
        new_K, _ = cv2.getOptimalNewCameraMatrix(self.K, self.D, (w, h), 0)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.K, self.D, None, new_K, (w, h), cv2.CV_16SC2
        )

    def undistort(self, img):
        """
        Devuelve la imagen corregida.
        """
        if self.map1 is None or self.map2 is None or self.image_size != (img.shape[1], img.shape[0]):
            self.init_undistort_maps(img.shape)
        return cv2.remap(img, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)