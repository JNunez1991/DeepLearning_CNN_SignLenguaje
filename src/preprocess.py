#!/usr/bin/env python3
"""Pre-Procesa las imagenes"""

import os
import random
import shutil
from dataclasses import dataclass
from glob import glob
from typing import Protocol, Any

import keras
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
    ncopies: int = 5

    def run_all(self, nfolds:list[str]) -> None:
        """Se encarga de ejecutar el paso a paso del proyecto"""

        self.create_subfolders(nfolds)
        self.copy_imgs_to_train_test(nfolds)
        self.data_augmentation()

    def create_subfolders(self, nfolds:list[str]) -> None:
        """Creo las subcarpetas dentro de Train/Test"""

        for fold in nfolds:
            os.mkdir(os.path.join(self.rutas.TRAIN_PATH, str(fold)))
            os.mkdir(os.path.join(self.rutas.TEST_PATH, str(fold)))
        return

    def copy_imgs_to_train_test(self, nfolds:list[str]) -> None:
        """Obtengo el listado con el nombre completo de todas las imagenes"""

        for fold in tqdm(nfolds, colour="green", desc="Copiando imagenes hacia Train/Test"):
            files = glob(os.path.join(self.rutas.IMGS_PATH, fold, '*'))

            train_imgs = random.sample(files, self.train_size)
            test_imgs = list( set(files) - set(train_imgs) )
            self.move_copies(train_imgs, self.rutas.TRAIN_PATH, str(fold))
            self.move_copies(test_imgs, self.rutas.TEST_PATH, str(fold))
        return

    def move_copies(self, files:list[str], dest_path:str, fold:str) -> None:
        """Genera copia de los archivos desde el origen a la subcarpeta destino"""

        for idx, img in enumerate(files):
            fullname = os.path.join(dest_path, fold, f"{str(idx)}.jpg")
            shutil.copy2(img, fullname)
        return

    def data_augmentation(self) -> None:
        """A las imagenes de cada subcarpeta, les aplico data augmentation"""

        dataset = self.set_dataset()
        file_paths = dataset.file_paths # type:ignore
        augm_model = self.set_augmentation_parameters()

        for idx, (image, _) in enumerate(tqdm(dataset, desc="Aumentando Dataset")):
            original_name = os.path.basename(file_paths[idx])[:-4] # quito la extension '.jpg'
            for j in range(self.ncopies):
                augmented_img = augm_model(image, training=True)
                final_img = keras.utils.array_to_img(augmented_img[0])
                save_name = f"{original_name}_augm{j+1}.jpg"
                final_img.save(os.path.join(os.path.dirname(file_paths[idx]), save_name))

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

    def set_dataset(self) -> list[Any]:
        """
        Utiliza image_dataset_from_directory, la cual toma como uno de sus parametros la
        carpeta padre donde estan almacenadas las imagenes. Es necesario que cada categoria
        tenga una subcarpeta dentro de ella. Dicha funcion detecta automaticamente la cantidad
        de imagenes por clase
        """

        data = keras.utils.image_dataset_from_directory(
            self.rutas.TRAIN_PATH,
            image_size=(224, 224),
            batch_size=1,
            shuffle=False
        )
        return data
