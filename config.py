#!/usr/bin/env python3
"""Rutas del proyecto"""

import os
from pathlib import Path

from enum import StrEnum

class Rutas(StrEnum):
    """Rutas del proyecto"""

    IMGS_PATH:str = os.path.join(Path(__file__).resolve().parent, "data", "raw")
    TRAIN_PATH: str = os.path.join(Path(__file__).resolve().parent, "data", "processed", "train")
    TEST_PATH: str = os.path.join(Path(__file__).resolve().parent, "data", "processed", "test")
