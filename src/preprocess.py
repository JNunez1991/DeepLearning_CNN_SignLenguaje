#!/usr/bin/env python3
"""Pre-Procesa las imagenes"""

import os
import random
import shutil
from dataclasses import dataclass
from glob import glob
from typing import Protocol

import keras
import tensorflow as tf
from tqdm import tqdm



class RutasProtocol(Protocol):
    """Protocolo de rutas"""

    IMGS_PATH: str
    TRAIN_PATH: str
    TEST_PATH: str


@dataclass
class PreProcess:
    """Clase que se encarga de analizar y pre-procesar las imagenes"""

    rutas: RutasProtocol
    train_size: int = 180

    def run_all(self, nfolds:list[str]) -> None:
        """Se encarga de ejecutar el paso a paso del proyecto"""

        self.create_subfolders(nfolds)
        self.copy_imgs_to_train_test(nfolds)

    def create_subfolders(self, nfolds:list[str]) -> None:
        """Creo las subcarpetas dentro de Train/Test"""

        for fold in nfolds:
            os.mkdir(os.path.join(self.rutas.TRAIN_PATH, fold))
            os.mkdir(os.path.join(self.rutas.TEST_PATH, fold))
        return

    def copy_imgs_to_train_test(self, nfolds:list[str]) -> None:
        """Obtengo el listado con el nombre completo de todas las imagenes"""

        for fold in tqdm(nfolds, colour="green", desc="Copiando imagenes hacia Train/Test"):
            files = glob(os.path.join(self.rutas.IMGS_PATH, fold, '*'))

            train_imgs = random.sample(files, self.train_size)
            for idx, img in enumerate(train_imgs):
                fullname = os.path.join(self.rutas.TRAIN_PATH, str(fold), f"{str(idx)}.jpg")
                shutil.copy2(img, fullname)

            test_imgs = list(set(files) - set(train_imgs))
            for idx, img in enumerate(test_imgs):
                fullname = os.path.join(self.rutas.TEST_PATH, str(fold), f"{str(idx)}.jpg")
                shutil.copy2(img, fullname)

        return

    def data_augmentation(self, nfolds:list[str]) -> None:
        """A las imagenes de cada subcarpeta, les aplico data augmentation"""

        parameters = self.set_augmentation_parameters()

        for fold in nfolds:
            dataset = self.set_dataset(str(fold))

    ###  QUEDA POR VER COMO GENERAR/GUARDAR LAS IMAGENES AUMENTADAS


    def set_augmentation_parameters(self) -> keras.Sequential:
        """Parametros del aumento de datos"""

        params = keras.Sequential([
            keras.layers.Rescaling(1./255),
            keras.layers.RandomRotation(30 / 360),
            keras.layers.RandomTranslation(0.25, 0.25),
            keras.layers.RandomZoom(
                height_factor=(-0.2, 0.2),
                width_factor=(-0.2, 0.2)
            ),
        ])
        return params


    def set_dataset(self, fold:str) -> tf.data.Dataset:
        """Seteo las imagenes a aplicarles data augmentation"""

        dataset = keras.utils.image_dataset_from_directory(
            os.path.join(self.rutas.TRAIN_PATH, fold),
            image_size=(224, 224),
            batch_size=1,
            shuffle=False
        )
        return dataset
