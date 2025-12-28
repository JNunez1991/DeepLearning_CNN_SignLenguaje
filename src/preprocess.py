#!/usr/bin/env python3
"""Pre-Procesa las imagenes"""

import os
import shutil
from dataclasses import dataclass, field
from glob import glob
from typing import Protocol

from tqdm import tqdm
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # ignora los msjs de tensorflow

class RutasProtocol(Protocol):
    """Protocolo de rutas"""

    IMGS_PATH: str
    TRAIN_PATH: str
    VAL_PATH: str
    TEST_PATH: str

class InvalidProportions(ValueError):
    """Las proporciones deben ser menor a uno"""

@dataclass
class PreProcess:
    """Clase que se encarga de analizar y pre-procesar las imagenes"""

    rutas: RutasProtocol
    train_size:float
    val_size:float
    ncopies: int = 5
    train_path:str = field(init=False)
    validation_path:str = field(init=False)
    raw_images_path:str = field(init=False)

    def __post_init__(self):
        """Se ejecuta luego de instanciar la clase"""

        print("")
        print("-. Pre procesando datos...")
        self.raw_images_path = os.path.abspath(self.rutas.IMGS_PATH)
        self.train_path = os.path.abspath(self.rutas.TRAIN_PATH)
        self.validation_path = os.path.abspath(self.rutas.VAL_PATH)
        self.test_path = os.path.abspath(self.rutas.TEST_PATH)
        self.check_sizes()

    def check_sizes(self) -> None:
        """Corta la ejecucion si (train_size + val_size) > 1 """

        if self.train_size + self.val_size > 1:
            raise InvalidProportions("Las proporciones deben ser <= 1")

    def run_all(self, nfolds:list[str]) -> None:
        """Se encarga de ejecutar el paso a paso del proyecto"""

        self.create_folders()
        self.create_folders_for_categories(nfolds)
        self.copy_imgs_to_folder(nfolds)

    def create_folders(self) -> None:
        """Si no existen, creo las carpetas de Train & Validation"""

        if not os.path.exists(self.train_path):
            os.mkdir(self.train_path)

        if not os.path.exists(self.validation_path):
            os.mkdir(self.validation_path)

        if not os.path.exists(self.test_path):
            os.mkdir(self.test_path)

    def create_folders_for_categories(self, nfolds:list[str]) -> None:
        """Si las carpetas de Train & Validation esetan vacias, creo las subcarpetas"""

        for fold in nfolds:
            os.mkdir( os.path.join(self.train_path, str(fold)) )
            os.mkdir( os.path.join(self.validation_path, str(fold)) )

    def copy_imgs_to_folder(self, nfolds:list[str]) -> None:
        """Obtengo el listado con el nombre completo de todas las imagenes"""

        for fold in tqdm(nfolds,
                         colour="green",
                         desc="Copiando imagenes hacia Train/Test"):

            files = glob(os.path.join(self.raw_images_path, fold, '*'))
            train_imgs, temp = train_test_split( files, train_size=self.train_size )
            val_imgs, test_imgs = train_test_split( temp, train_size= 1-self.val_size )

            # train_imgs = random.sample(files, self.train_size)
            # aux = list( set(files) - set(train_imgs) )
            # val_imgs = random.sample(aux, self.val_size)
            # test_imgs = list( set(aux) - set(val_imgs) )

            self.move_copies(train_imgs, self.train_path, str(fold))
            self.move_copies(val_imgs, self.validation_path, str(fold))
            self.move_copies(test_imgs, self.test_path)

    def move_copies(
        self,
        files:list[str],
        dest_path:str,
        fold:str | None = None,
    ) -> None:
        """Genera copia de los archivos desde el origen a la subcarpeta destino"""

        if fold is not None:
            for img in files:
                img_name = os.path.basename(img)
                fullname = os.path.join(dest_path, fold, f"{str(img_name)}")
                shutil.copy2(img, fullname)
        else:
            for img in files:
                img_name = os.path.basename(img)
                fullname = os.path.join(dest_path, f"{str(img_name)}")
                shutil.copy2(img, fullname)
