#!/usr/bin/env python3
# pylint: disable=no-name-in-module, no-member
"""Utilizo el modelo con mejores resulados, para predecir en tiempo real (con webcam)"""

import os
from dataclasses import dataclass, field

from keras.models import load_model

from config import Rutas, ModelNames, ImageParameters
from src.realtime import Camara

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

@dataclass
class RealTime:
    """Carga el mejor modelo, y lo utiliza para predecir en real time"""

    modelname: str
    camara: Camara = field(default_factory=Camara)

    def __post_init__(self):
        """Se ejecuta luego de instanciar la clase"""

        print("")
        print(f"-. Cargando modelo {self.modelname.upper()} y activando camara...")
        self.full_model_name = os.path.join(Rutas.MODEL_PATH, f"model_{self.modelname}.keras")

    def run_all(self) -> None:
        """Ejecuta el modelo paso a paso"""

        model = load_model(self.full_model_name)
        self.camara.initialize(model, ImageParameters.DIMS) # type:ignore

if __name__ == "__main__":

    realtime = RealTime(ModelNames.DATA_AUGMENTATION)
    realtime.run_all()
