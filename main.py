#!/usr/bin/env python3
"""Orquestador principal del proyecto"""

import os
import shutil
from dataclasses import dataclass, field

from config import Rutas, ImageParameters, ModelNames
from src import (
    PreProcess,
    Visualization,
    Model,
    DataAugmentation,
    EvalModel,
)


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

    def visualize_train_images(self) -> None:
        """Se encarga de ejecutar el paso a paso del proyecto"""

        viz = Visualization(Rutas.TRAIN_PATH, self.folders)
        viz.run_all()

    def preprocess_images(self, train_size:int) -> None:
        """Genera copias de las imagenes y las manda a train/test"""

        preproc = PreProcess(Rutas, train_size) # type:ignore
        preproc.run_all(self.folders)

    def execute_model(
        self,
        version:str,
        use_tl:bool=False, # Transfer Learning
    ):
        """Ejecuta el entrenamiento sobre las imagenes crudas, sin ninguna modificación"""

        modelname = f"model_{version}.keras"
        model = Model(
            Rutas, # type:ignore
            self.nfolders,
            modelname,
            version,
            ImageParameters.DIMS,
            ImageParameters.EPOCS,
            ImageParameters.BATCH,
            use_tl,

        )
        model.run_all()
        return model

    def apply_data_augmentation(self) -> None:
        """Aplica Data Augmentation a las imagenes de Train"""

        augm = DataAugmentation(Rutas.TRAIN_PATH, ImageParameters.DIMS)
        augm.run_all()

    def evaluate_model(self, modelname:str) -> None:
        """Evalua el rendimiento de un modelo determinado"""

        score = EvalModel(
            Rutas, # type:ignore
            modelname,
            ImageParameters.DIMS,
            ImageParameters.BATCH,
        )
        score.run_all(modelname)


if __name__ == "__main__":

    main = Main()
    names = ModelNames()

    # Divido las imagenes originales en carpetas de Train/Test
    main.preprocess_images(train_size=180)

    # Visualizo las imagenes en Train
    main.visualize_train_images()

    # Primer modelo: Imagenes puras
    raw_model = main.execute_model(version=names.RAW)

    # Segundo modelo: Transfer Learning
    tl_model = main.execute_model(version=names.TRANSFER_LEARNING, use_tl=True)

    # Aplico data augmentation sobre las imagenes
    main.apply_data_augmentation()

    # Tercer modelo: Data augmentation
    augm_model = main.execute_model(version=names.DATA_AUGMENTATION)

    # Cuarto modelo: Data augmentation + Transfer Learning
    tlaugm_model = main.execute_model(version=names.TRANSFER_AND_AUGMENTATION, use_tl=True)

    # # Evaluo los modelos
    # main.evaluate_model(names.RAW)
    # main.evaluate_model(names.TRANSFER_LEARNING)
    # main.evaluate_model(names.DATA_AUGMENTATION)
    # main.evaluate_model(names.TRANSFER_AND_AUGMENTATION)
