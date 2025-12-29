#!/usr/bin/env python3
"""Clases comunes al resto de modulos"""

from enum import StrEnum
from typing import Protocol


class RutasProtocol(Protocol):
    """Rutas del proyecto"""

    IMGS_PATH: str
    TRAIN_PATH: str
    VAL_PATH: str
    TEST_PATH: str
    MODEL_PATH: str


class Colores(StrEnum):
    """Colores para entrenamiento"""

    GRAYSCALE = "grayscale"
