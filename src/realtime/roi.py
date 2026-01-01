#!/usr/bin/env python3
# pylint: disable=no-name-in-module, no-member
"""Inicializa la camara"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Region:
    """
    Crea la ROI (Region Of Interest).
    Cuadro dentro de la imagen que se utilizara para la deteccion de la mano.
    """

    def create(
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
