#!/usr/bin/env python3
"""Orquestador principal del proyecto"""

import os
import shutil
from dataclasses import dataclass, field

from config import Rutas, ImageParameters
from src import PreProcess, Visualization, Model, DataAugmentation


@dataclass
class Main:
    """Clase central del proyecto"""

    folders: list[str] = field(init=False)
    nfolders: int = field(init=False)

    def __post_init__(self):
        """Se ejecuta luego de instanciar la clase"""

        self.folders = os.listdir(Rutas.IMGS_PATH)
        self.nfolders = len(self.folders)
        self.delete_folders_train_validation()

    def delete_folders_train_validation(self) -> None:
        """Elimina las carpetas de Train & Validation"""

        if os.path.exists(Rutas.TRAIN_PATH):
            shutil.rmtree( os.path.abspath(Rutas.TRAIN_PATH) )

        if os.path.exists(Rutas.VAL_PATH):
            shutil.rmtree( os.path.abspath(Rutas.VAL_PATH) )

    def visualize_base_images(self) -> None:
        """Se encarga de ejecutar el paso a paso del proyecto"""

        viz = Visualization(Rutas.IMGS_PATH, self.folders)
        viz.run_all()

    def preprocess_images(self, train_size:int) -> None:
        """Genera copias de las imagenes y las manda a train/test"""

        preproc = PreProcess(Rutas, train_size) # type:ignore
        preproc.run_all(self.folders)

    def apply_data_augmentation(self) -> None:
        """Aplica Data Augmentation a las imagenes de Train"""

        augm = DataAugmentation(Rutas.TRAIN_PATH, ImageParameters.DIMS)
        augm.run_all()

    def execute_model(self, version:str):
        """Ejecuta el entrenamiento sobre las imagenes crudas, sin ninguna modificación"""

        modelname = f"model_{version}.keras"
        rawmodel = Model(
            Rutas, # type:ignore
            self.nfolders,
            modelname,
            version,ImageParameters.DIMS,
        )
        rawmodel.run_all()

if __name__ == "__main__":

    main = Main()
    # main.visualize_base_images()

    # Divido las imagenes originales en carpetas de Train/Test
    main.preprocess_images(train_size=180)

    # # Primer modelo: Imagenes puras
    # main.execute_model(version="raw")
    # main.evaluate_model()

    # # Segundo modelo: Data augmentation
    # main.apply_data_augmentation()
    # main.execute_model(version="data_augm")
    # main.evaluate_model()
