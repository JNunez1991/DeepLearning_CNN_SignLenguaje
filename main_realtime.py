#!/usr/bin/env python3
# pylint: disable=no-name-in-module, no-member
"""Utilizo el modelo con mejores resulados, para predecir en tiempo real (con webcam)"""

import os
from dataclasses import dataclass

import numpy as np
# import tensorflow as tf
import cv2
from keras.models import load_model, Model

from config import Rutas, ModelNames, ImageParameters

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

        model = load_model(self.full_model_name)
        self.start_camera(model) # type:ignore

    def start_camera(self, model:Model) -> None:
        """Inicializa la webcam para la prediccion en tiempo real"""

        cap = cv2.VideoCapture(0) #type:ignore

        # Abro la camara
        if not cap.isOpened():
            raise RuntimeError("No se pudo acceder a la cámara")

        print("Presiona 'q' para salir.")

        # Capturar frame por frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            input_tensor = self.preprocess_frame(frame)
            prediction = self.predict(input_tensor)
            self.draw_prediction(frame, prediction)

            cv2.imshow("Sign Language - Real Time", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def preprocess_frame(self, frame):
        """
        Opencv (cv2) captura las imagenes en formato BGR.
        Entonces transformo cada frame a tensor (1, H, W, 1)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, ImageParameters.DIMS[:2])
        normalized = resized / 255.0
        return normalized.reshape(1, ImageParameters.DIMS)

    def predict(self, model:Model, input_tensor:tuple[float, ...]):
        """Predice la clase de la imagen"""
        preds = model.predict(input_tensor, verbose=0)
        return int(np.argmax(preds))

    def draw_prediction(self, frame, prediction):
        cv2.putText(
            frame,
            f"Prediccion: {prediction}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

if __name__ == "__main__":

    realtime = RealTime(ModelNames.DATA_AUGMENTATION)
