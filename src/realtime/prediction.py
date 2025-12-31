#!/usr/bin/env python3
# pylint: disable=no-name-in-module, no-member
"""Inicializa la camara"""

from dataclasses import dataclass

import numpy as np
import cv2
from keras.models import Model

@dataclass
class Prediction:
    """Procesa el frame y realiza la prediccion"""

    def preprocess_frame(self, frame:np.ndarray, params:tuple[int,...]) -> np.ndarray:
        """
        Opencv (cv2) captura las imagenes en formato BGR.
        Entonces transformo cada frame a tensor (1, H, W, 1)
        """

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, params[:2])
        resized = resized.astype("float32")
        return resized.reshape(1, *params)

    def predict(
        self,
        model:Model,
        input_tensor: np.ndarray,
    ) -> tuple[int, float]:
        """Predice la clase de la imagen"""

        preds = model.predict(input_tensor, verbose=0) # type:ignore
        clase = int(np.argmax(preds))
        confianza = float(np.max(preds))
        return clase, confianza
