#!/usr/bin/env python3
"""Pre-Procesa las imagenes"""

import os
import random
import shutil
from dataclasses import dataclass
from glob import glob
from typing import Protocol

from tqdm import tqdm

class RutasProtocol(Protocol):
    """Protocolo de rutas"""

    IMGS_PATH: str
    TRAIN_PATH: str
    VAL_PATH: str


@dataclass
class PreProcess:
    """Clase que se encarga de analizar y pre-procesar las imagenes"""

    rutas: RutasProtocol
    train_size:int
    ncopies: int = 5

    def run_all(self, nfolds:list[str]) -> None:
        """Se encarga de ejecutar el paso a paso del proyecto"""

        self.create_subfolders(nfolds)
        self.copy_imgs_to_train_test(nfolds)

    def create_subfolders(self, nfolds:list[str]) -> None:
        """Creo las subcarpetas dentro de Train/Test"""

        for fold in nfolds:
            os.mkdir(os.path.join(self.rutas.TRAIN_PATH, str(fold)))
            os.mkdir(os.path.join(self.rutas.VAL_PATH, str(fold)))
        return

    def copy_imgs_to_train_test(self, nfolds:list[str]) -> None:
        """Obtengo el listado con el nombre completo de todas las imagenes"""

        for fold in tqdm(nfolds, colour="green", desc="Copiando imagenes hacia Train/Test"):
            files = glob(os.path.join(self.rutas.IMGS_PATH, fold, '*'))
            train_imgs = random.sample(files, self.train_size)
            test_imgs = list( set(files) - set(train_imgs) )
            self.move_copies(train_imgs, self.rutas.TRAIN_PATH, str(fold))
            self.move_copies(test_imgs, self.rutas.VAL_PATH, str(fold))
        return

    def move_copies(self, files:list[str], dest_path:str, fold:str) -> None:
        """Genera copia de los archivos desde el origen a la subcarpeta destino"""

        for idx, img in enumerate(files):
            fullname = os.path.join(dest_path, fold, f"{str(idx)}.jpg")
            shutil.copy2(img, fullname)
        return
