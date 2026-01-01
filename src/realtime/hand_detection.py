#!/usr/bin/env python3
# pylint: disable=no-name-in-module, no-member
"""Inicializa la camara"""

from dataclasses import dataclass

import numpy as np
# import mediapipe as mp

@dataclass
class HandDetection:
    """
    Detecta si en ROI existe una mano.
    Se crea una clase por posibles futuros desarrollos en este sentido.
    """

    def find(
        self,
        roi: np.ndarray,
        threshold: float = 15.0,
    ) -> bool:
        """
        Heurística simple:
        Devuelve False si el ROI es demasiado uniforme (sin mano)
        """
        return float(np.std(roi)) > threshold


# @dataclass
# class HandDetectionMediaPipe:
#     """
#     NO CARGA SOLUTIONS
#     Detecta la presencia de una mano en un ROI.
#     """

#     min_detection_conf: float = 0.5

#     def run_all(self, roi: np.ndarray) -> bool:
#         """
#         Devuelve True si detecta una mano en el ROI, de lo contrario False.
#         cv2 trabaja con BGR, por lo que hay que pasarlo a RGB.
#         """

#         params = self.hand_parameters()
#         hand_exists = self.scanner(roi, params)
#         return hand_exists

#     def hand_parameters(self) -> mp_hands.Hands: #type:ignore
#         """Parametros a utilizar para la deteccion"""

#         return mp_hands.Hands(
#             static_image_mode=False,
#             max_num_hands=1,
#             min_detection_confidence=self.min_detection_conf
#         )

#     def scanner(self, roi:np.ndarray, params:mp_hands.Hands) -> bool: #type:ignore
#         """Detecta """

#         roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
#         results = params.process(roi_rgb)
#         return bool(results.multi_hand_landmarks)
