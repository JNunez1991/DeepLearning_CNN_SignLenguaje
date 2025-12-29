#!/usr/bin/env python3
"""Orquestador principal de la pata de entrenamiento de modelos"""

import os
import shutil
from dataclasses import dataclass, field

from keras.models import Sequential

from config import Rutas, ImageParameters
from src.entrenamiento import (
    DataAugmentation,
    EvalModel,
    Model,
    ModelPredict,
    PreProcess,
    Visualization,
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

    def preprocess_images(self, train_size:float, val_size:float, ncopies:int=8) -> None:
        """Genera copias de las imagenes y las manda a train/test"""

        self.preprocess_images_delete_folders()
        preproc = PreProcess(Rutas, train_size, val_size, ncopies) # type:ignore
        preproc.run_all(self.folders)

    def preprocess_images_delete_folders(self) -> None:
        """Elimina las carpetas de Train & Validation"""

        if os.path.exists(Rutas.TRAIN_PATH):
            shutil.rmtree( os.path.abspath(Rutas.TRAIN_PATH) )

        if os.path.exists(Rutas.VAL_PATH):
            shutil.rmtree( os.path.abspath(Rutas.VAL_PATH) )

        if os.path.exists(Rutas.TEST_PATH):
            shutil.rmtree( os.path.abspath(Rutas.TEST_PATH) )

    def visualize_train_images(self) -> None:
        """Se encarga de ejecutar el paso a paso del proyecto"""

        viz = Visualization(Rutas.TRAIN_PATH, self.folders)
        viz.run_all()

    def execute_model(
        self,
        version:str,
        use_tl:bool=False, # Transfer Learning
    ) -> Sequential:
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
        return model.run_all()

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

    def predict(self, modelname:str, plot:bool=False) -> dict[str, str]:
        """A cada imagen de Test, le predice una clase"""

        pred = ModelPredict(
            Rutas, # type:ignore
            modelname,
            ImageParameters.DIMS,
            plot,
        )
        return pred.run_all() # type:ignore
