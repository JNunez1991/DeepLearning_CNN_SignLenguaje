#!/usr/bin/env python3
# pylint: disable=no-name-in-module, no-member
"""Inicializa la camara"""

from dataclasses import dataclass

import numpy as np
import cv2

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
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray)) > threshold # type:ignore
