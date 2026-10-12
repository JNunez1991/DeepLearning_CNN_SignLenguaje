#!/usr/bin/env python3
"""Clases comunes al resto de modulos"""

from enum import StrEnum
from dataclasses import dataclass

from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping


class Colores(StrEnum):
    """Colores para entrenamiento"""

    GRAYSCALE = "grayscale"


@dataclass
class Callbacks:
    """Callbacks del modelo"""

    tensorboard: TensorBoard
    checkpoint: ModelCheckpoint
    early_stop: EarlyStopping
