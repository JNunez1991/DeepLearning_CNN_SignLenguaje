#!/usr/bin/env python3
"""Pre-Procesa las imagenes"""

import os
import random
import shutil
from dataclasses import dataclass, field
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
    train_path:str = field(init=False)
    validation_path:str = field(init=False)
    raw_images_path:str = field(init=False)

    def __post_init__(self):
        """Se ejecuta luego de instanciar la clase"""

        self.train_path = os.path.abspath(self.rutas.TRAIN_PATH)
        self.validation_path = os.path.abspath(self.rutas.VAL_PATH)
        self.raw_images_path = os.path.abspath(self.rutas.IMGS_PATH)

    def run_all(self, nfolds:list[str]) -> None:
        """Se encarga de ejecutar el paso a paso del proyecto"""

        self.create_folders_train_validation()
        self.create_folders_for_categories(nfolds)
        self.copy_imgs_to_train_validation(nfolds)

    def create_folders_train_validation(self) -> None:
        """Si no existen, creo las carpetas de Train & Validation"""

        if not os.path.exists(self.train_path):
            os.mkdir(self.train_path)

        if not os.path.exists(self.validation_path):
            os.mkdir(self.validation_path)

    def create_folders_for_categories(self, nfolds:list[str]) -> None:
        """Si las carpetas de Train & Validation esetan vacias, creo las subcarpetas"""

        for fold in nfolds:
            os.mkdir( os.path.join(self.train_path, str(fold)) )
            os.mkdir( os.path.join(self.validation_path, str(fold)) )

    def copy_imgs_to_train_validation(self, nfolds:list[str]) -> None:
        """Obtengo el listado con el nombre completo de todas las imagenes"""

        for fold in tqdm(nfolds,
                         colour="green",
                         desc="Copiando imagenes hacia Train/Test"):
            files = glob(os.path.join(self.raw_images_path, fold, '*'))
            train_imgs = random.sample(files, self.train_size)
            test_imgs = list( set(files) - set(train_imgs) )
            self.move_copies(train_imgs, self.train_path, str(fold))
            self.move_copies(test_imgs, self.validation_path, str(fold))

    def move_copies(
        self,
        files:list[str],
        dest_path:str,
        fold:str,
    ) -> None:
        """Genera copia de los archivos desde el origen a la subcarpeta destino"""

        for img in files:
            img_name = os.path.basename(img)
            fullname = os.path.join(dest_path, fold, f"{str(img_name)}")
            shutil.copy2(img, fullname)
