#!/usr/bin/env python3
#pylint: disable=invalid-name
"""Rutas del proyecto"""

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

class Rutas(StrEnum):
    """Rutas del proyecto"""

    IMGS_PATH = os.path.join(Path(__file__).resolve().parent, "data", "raw")
    TRAIN_PATH = os.path.join(Path(__file__).resolve().parent, "data", "processed", "train")
    VAL_PATH = os.path.join(Path(__file__).resolve().parent, "data", "processed", "val")
    TEST_PATH = os.path.join(Path(__file__).resolve().parent, "data", "processed", "test")
    MODEL_PATH = os.path.join(Path(__file__).resolve().parent, "models")

@dataclass
class ImageParameters:
    """Parametros de las imagenes"""

    DIMS: tuple[int, ...] = (100, 100, 1) # imgs a color que se las pasa a blanco y negro
    EPOCS: int = 40
    BATCH: int = 32


class ModelNames(StrEnum):
    """Nombres de los modelos a utilizar"""

    RAW = "raw"
    TRANSFER_LEARNING = "tl"
    DATA_AUGMENTATION = "augmentation"
    TRANSFER_AND_AUGMENTATION = "tlaugm"
