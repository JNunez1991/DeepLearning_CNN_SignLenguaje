#!/usr/bin/env python3
# pylint: disable=no-name-in-module
"""Utilizo el modelo con mejores resulados, para predecir en tiempo real (con webcam)"""

import os
from dataclasses import dataclass

# import numpy as np
# import tensorflow as tf
from cv2 import VideoCapture#, imshow, waitKey
# from keras.models import load_model #, Model

from config import Rutas, ModelNames #, ImageParameters

@dataclass
class RealTime:
    """Carga el mejor modelo, y lo utiliza para predecir en real time"""

    modelname: str

    def __post_init__(self):
        """Se ejecuta luego de instanciar la clase"""

        print("")
        print(f"-. Cargando modelo {self.modelname.upper()} y activando camara...")
        self.full_model_name = os.path.join(Rutas.MODEL_PATH, f"model_{self.modelname}.keras")

    def run_all(self) -> None:
        """Ejecuta el modelo paso a paso"""

        # model = load_model(self.full_model_name)
        self.initialize_webcam()

    def initialize_webcam(self) -> None:
        """Inicializa la webcam para la prediccion en tiempo real"""

        cap = VideoCapture(0) #type:ignore
        if not cap.isOpened():
            print("Error: No se pudo acceder a la cámara.")
            exit()

if __name__ == "__main__":

    realtime = RealTime(ModelNames.DATA_AUGMENTATION)
