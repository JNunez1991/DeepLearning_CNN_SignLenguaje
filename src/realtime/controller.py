#!/usr/bin/env python3
# pylint: disable=no-name-in-module, no-member
"""Inicializa la camara"""

from dataclasses import dataclass, field

import cv2
from keras.models import Model

from .text import TextInFrame
from .hand_detection import HandDetection
from .roi import Region
from .prediction import Prediction

@dataclass
class Camara:
    """Carga el mejor modelo, y lo utiliza para predecir en real time"""

    roi: Region = field(default_factory=Region)
    texto: TextInFrame = field(default_factory=TextInFrame)
    hand: HandDetection = field(default_factory=HandDetection)
    prediction: Prediction = field(default_factory=Prediction)

    def initialize(
        self,
        model:Model,
        modelname:str,
        params:tuple[int, ...],
        exit_keys = (ord('q'), ord('Q'))
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

            # Espejo
            frame = cv2.flip(frame, 1)

            # ---- ROI (parametros) ----
            roi_size = params[0] * 2
            roi, (x1, y1, x2, y2) = self.roi.create(frame, roi_size)

            # ---- DETECCIÓN DE MANO (parche, mejor usar mediapipe) ----
            if self.hand.find(roi):
                input_tensor = self.prediction.preprocess_frame(roi, params)
                prediction, porcentaje = self.prediction.predict(model, input_tensor)
            else:
                prediction = "NA"
                porcentaje = 0.0

            # ---- BLUR ----
            blurred = cv2.GaussianBlur(frame, (75, 75), 0)
            blurred[y1:y2, x1:x2] = frame[y1:y2, x1:x2]

            # ---- ROI ----
            cv2.rectangle(blurred, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # ---- TEXTO EN PANTALLA ----
            self.texto.header(blurred, modelname)
            self.texto.info(blurred, prediction, porcentaje) #type:ignore
            self.texto.footer(blurred)

            cv2.imshow("Sign Language - Real Time", blurred)

            if cv2.waitKey(1) & 0xFF in exit_keys:
                break

        cap.release()
        cv2.destroyAllWindows()
