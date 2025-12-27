#!/usr/bin/env python3
"""Orquestador principal del proyecto"""

import os
import shutil
from dataclasses import dataclass, field

from config import Rutas
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

    def analize_base_images(self) -> None:
        """Se encarga de ejecutar el paso a paso del proyecto"""

        viz = Visualization(Rutas.IMGS_PATH, self.folders)
        viz.run_all()

    def split_images(self, train_size:int) -> None:
        """Genera copias de las imagenes y las manda a train/test"""

        preproc = PreProcess(Rutas, train_size) # type:ignore
        preproc.run_all(self.folders)

    def apply_data_augmentation(self) -> None:
        """Aplica Data Augmentation a las imagenes de Train"""

        augm = DataAugmentation(Rutas.TRAIN_PATH)
        augm.run_all()

    def execute_model(self, version:str):
        """Ejecuta el entrenamiento sobre las imagenes crudas, sin ninguna modificación"""

        modelname = f"model_{version}.keras"
        rawmodel = Model(Rutas, self.nfolders, modelname, version) # type:ignore
        rawmodel.run_all()


    def delete_train_test_items(self, nfolds:list[str]) -> None:
        """Elimina todas las imagenes en las carpetas de Train/Test"""

        for fold in nfolds:
            shutil.rmtree( os.path.join(Rutas.TRAIN_PATH, str(fold)) )
            shutil.rmtree( os.path.join(Rutas.VAL_PATH, str(fold)) )


if __name__ == "__main__":

    main = Main()
    # main.analize_base_images()

    # # Divido las imagenes originales en carpetas de Train/Test
    main.split_images(train_size=180)

    # # Primer modelo: Imagenes puras
    # main.execute_model(version="raw")
    # main.evaluate_model()

    # # Segundo modelo: Data augmentation
    # main.apply_data_augmentation()
    # main.execute_model(version="data_augm")
    # main.evaluate_model()
