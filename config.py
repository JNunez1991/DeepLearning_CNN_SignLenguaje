#!/usr/bin/env python3
#pylint: disable=invalid-name
"""Rutas del proyecto"""

import os
from dataclasses import dataclass
from pathlib import Path

from enum import StrEnum

class Rutas(StrEnum):
    """Rutas del proyecto"""

    IMGS_PATH = os.path.join(Path(__file__).resolve().parent, "data", "raw")
    TRAIN_PATH = os.path.join(Path(__file__).resolve().parent, "data", "processed", "train")
    VAL_PATH = os.path.join(Path(__file__).resolve().parent, "data", "processed", "val")
    MODEL_PATH = os.path.join(Path(__file__).resolve().parent, "models")


@dataclass
class ImageParameters:
    """Parametros de las imagenes"""

    DIMS: tuple[int, int] = (100, 100)
