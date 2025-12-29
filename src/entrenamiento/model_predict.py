#!/usr/bin/env python3
#pylint: disable=line-too-long
"""Modelo con el que se va a entrenar"""

import os
import random
from dataclasses import dataclass, field
from glob import glob
from typing import Protocol

import numpy as np
from keras.preprocessing import image
from keras.models import load_model, Model
from matplotlib import pyplot as plt, image as mpimg


class RutasProtocol(Protocol):
    """Protocolo de rutas"""

    TEST_PATH: str
    MODEL_PATH: str


@dataclass
class ModelPredict:
    """Definicion del modelo a utilizar"""

    rutas: RutasProtocol
    modelname:str
    img_size:tuple[int, int, int]
    plot:bool = False

    full_model_name:str = field(init=False)

    def __post_init__(self):
        """Se ejecuta luego de instanciar la clase"""

        print("")
        print(f"-. Cargando modelo {self.modelname.upper()} y evaluando sobre Test...")
        self.full_model_name = os.path.join(self.rutas.MODEL_PATH, f"model_{self.modelname}.keras")

    def run_all(self) -> dict[str, SyntaxWarning]:
        """Ejecuta el proceso paso a paso"""

        file_names = self.get_files_names()
        model = load_model(self.full_model_name)
        resultados = self.create_predictions(file_names, model) # type:ignore
        if self.plot:
            self.show_predictions(file_names, resultados)
        return resultados # type:ignore

    def get_files_names(self) -> list[str]:
        """Carga las imagnes en la carpeta Test"""

        return glob( os.path.join(self.rutas.TEST_PATH, '*') )

    def create_predictions(
        self,
        filenames:list[str],
        model:Model,
    ) -> dict[str, str]:
        """Diccionario que tiene, para cada imagen la prediccion de la clase a la que pertenece"""

        resultados = {}
        for img in filenames:

            # Cargo la imagen con keras
            imagen = image.load_img(img, target_size=self.img_size)

            # Imagen a array + dimension extra para batch
            img_array = image.img_to_array(imagen)
            img_array = np.expand_dims(img_array, axis=0)

            # Realizo la prediccion y la guardo
            prediction = model.predict(img_array, verbose=0) # type:ignore
            resultados[os.path.basename(img)] = np.argmax(prediction, axis=1)[0]

        return resultados

    def show_predictions(
        self,
        filenames:list[str],
        preds:dict[str, str],
        nimages:int = 5,
    ) -> None:
        """Selecciona aleatoriamente una carpeta y muestra algunas imagenes"""

        plt.figure(figsize=(15,15))
        plt.suptitle("Imagenes de Test", y=0.6)

        imgs_to_show = random.sample(filenames, nimages)

        # for idx, nombreimg in enumerate(imgs_to_show):
        for idx, img in enumerate(imgs_to_show):
            plt.subplot(1, nimages, idx+1)
            image_name = os.path.basename(img)
            pred = preds[image_name]
            plt.title(f"Prediccion: {pred}")
            imagen = mpimg.imread(os.path.abspath(img)) # mpimg.imread lee los pixeles
            plt.imshow(imagen)
