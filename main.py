#!/usr/bin/env python3
"""Orquestador principal del proyecto"""

import os
import shutil
from dataclasses import dataclass

from config import Rutas
from src import PreProcess, Visualization


@dataclass
class Main:
    """Clase central del proyecto"""

    def run_all(self) -> None:
        """Se encarga de ejecutar el paso a paso del proyecto"""

        # Cantidad de imagenes por subcarpeta, e imagenes de prueba
        viz = Visualization() # type:ignore
        folders = viz.run_all(Rutas.IMGS_PATH, "-. Imagenes Reales...")

        # Generacion de carpetas Train/Test e imagenes con Data Augmentation
        # preproc = PreProcess(Rutas) # type:ignore
        # preproc.run_all(folders)
        viz.run_all(Rutas.TRAIN_PATH, "- Imagenes con data augmentation...", aug=True)


        # # Elimino las imagenes de Train/Test para limpieza
        # self.delete_train_test_items(folders)



    def delete_train_test_items(self, nfolds:list[str]) -> None:
        """Elimina todas las imagenes en las carpetas de Train/Test"""

        for fold in nfolds:
            shutil.rmtree( os.path.join(Rutas.TRAIN_PATH, str(fold)) )
            shutil.rmtree( os.path.join(Rutas.TEST_PATH, str(fold)) )


if __name__ == "__main__":

    main = Main()
    main.run_all()
