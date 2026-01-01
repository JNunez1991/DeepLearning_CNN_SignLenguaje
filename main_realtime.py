#!/usr/bin/env python3
# pylint: disable=wrong-import-position
"""Utilizo el modelo con mejores resulados, para predecir en tiempo real (con webcam)"""

import os
from dataclasses import dataclass, field
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # evita que aparezcan msjs de tensorflow

import pyfiglet
from keras.models import load_model

from config import Rutas, ModelNames, ImageParameters
from src.realtime.controller import Camara


@dataclass
class RealTime:
    """Carga el mejor modelo, y lo utiliza para predecir en real time"""

    camara: Camara = field(default_factory=Camara)

    def run_all(self) -> None:
        """Ejecuta el modelo paso a paso"""

        self.print_header("Sign Lenguaje Interface")
        seleccion = self.seleccionar_opcion()
        path = os.path.join(Rutas.MODEL_PATH, f"model_{seleccion}.keras")
        model = load_model(path)
        self.camara.initialize(model, seleccion, ImageParameters.DIMS) # type:ignore

    def print_header(self, texto:str, font="slant") -> None:
        """Titulo en consola"""

        titulo = pyfiglet.figlet_format(texto, font=font)
        print(titulo)

    def seleccionar_opcion(self) -> str:
        """Le pregunta al usuario que modelo quiere usar"""

        while True:
            self.print_options()
            try:
                opcion = int(input("Seleccione una opción (1-4): "))
                if 1 <= opcion <= 4:
                    return self.set_model(opcion)
                else:
                    print("-. ❌ ¡Error!: Por favor, elija un número entre 1 y 4. \n")
            except ValueError:
                print("-. ❌ ¡Error!: Entrada no válida. Debe ingresar un número entero. \n")

    def print_options(self) -> None:
        """Opciones disponibles"""

        msg = \
        """ ¿Que modelo quiere utilizar?
        1. RAW
        2. TRANSFER LEARNING
        3. DATA AUGMENTATION
        4. DATA AUGMENTATION + TRANSFER LEARNING
        """

        print(msg)

    def set_model(self, option:int) -> str:
        """Transformo la opcion del usuario (int) al nombre del modelo"""

        if option == 1:
            return ModelNames.RAW

        if option == 2:
            return ModelNames.TRANSFER_LEARNING

        if option == 3:
            return ModelNames.DATA_AUGMENTATION

        return ModelNames.TRANSFER_AND_AUGMENTATION



if __name__ == "__main__":

    realtime = RealTime()
    realtime.run_all()
