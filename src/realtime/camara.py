#!/usr/bin/env python3
# pylint: disable=no-name-in-module, no-member
"""Inicializa la camara"""

from dataclasses import dataclass, field

import numpy as np
import cv2
from keras.models import Model

from .text import TextInFrame


@dataclass
class Camara:
    """Carga el mejor modelo, y lo utiliza para predecir en real time"""

    texto: TextInFrame = field(default_factory=TextInFrame)

    def initialize(
        self,
        model:Model,
        params:tuple[int, ...]
    ) -> None:
        """Inicializa la webcam para la prediccion en tiempo real"""

        cap = cv2.VideoCapture(0) #type:ignore

        # Abro la camara
        if not cap.isOpened():
            raise RuntimeError("No se pudo acceder a la cámara")

        # Capturar frame por frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ---- ROI (parametros) ----
            roi_size = params[0] * 2
            roi, (x1, y1, x2, y2) = self.get_roi(frame, roi_size)

            # ---- DETECCIÓN DE MANO (parche, mejor usar mediapipe) ----
            if not self.hand_present(roi):
                prediction = "NA"
                porcentaje = 0.0
            else:
                input_tensor = self.preprocess_frame(roi, params)
                prediction, porcentaje = self.predict(model, input_tensor)

            # ---- BLUR ----
            blurred = cv2.GaussianBlur(frame, (75, 75), 0)
            blurred[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

            # ---- ROI ----
            cv2.rectangle(blurred, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # ---- TEXTO EN PANTALLA ----
            self.texto.header(blurred)
            self.texto.info(blurred, prediction, porcentaje) #type:ignore

            cv2.imshow("Sign Language - Real Time", blurred)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def get_roi(
        self,
        frame: np.ndarray,
        size: int,
        padding: int = 20,
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        """
        Devuelve:
        - roi: region of interest -- imagen recortada NxN
        - (x1, y1, x2, y2): coordenadas del cuadro
        """
        h, w, _ = frame.shape

        size = min(size, h, w)
        x1 = padding # alineacion horizontal derecha
        y1 = (h - size) // 2
        x2 = x1 + size
        y2 = y1 + size

        roi = frame[y1:y2, x1:x2]
        return roi, (x1, y1, x2, y2)

    def hand_present(
        self,
        roi: np.ndarray,
        threshold: float = 15.0,
    ) -> bool:
        """
        Heurística simple:
        Devuelve False si el ROI es demasiado uniforme (sin mano)
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray)) > threshold # type:ignore

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
