#!/usr/bin/env python3
"""Pre-Procesa las imagenes"""

import os
import random
from dataclasses import dataclass
from typing import Protocol


import numpy as np
from matplotlib import pyplot as plt, image as mpimg

class RutasProtocol(Protocol):
    """Protocolo de rutas"""

    IMGS_PATH: str
    TRAIN_PATH: str
    TEST_PATH: str


@dataclass
class Visualization:
    """Clase que se encarga de analizar y pre-procesar las imagenes"""

    rutas: RutasProtocol
    train_size: int = 180

    def run_all(self) -> list[str]:
        """Se encarga de ejecutar el paso a paso del proyecto"""

        nfolders = os.listdir(self.rutas.IMGS_PATH)
        self.folders_components(nfolders)
        self.show_images_same_folder(nfolders)
        self.show_images_different_folder(nfolders)
        return nfolders

    def folders_components(
        self,
        nfolds:list[str],
        ) -> None:
        """Imprime por consola la cantidad de imagenes en cada carpeta"""

        for fold in nfolds:
            path_folder = os.path.join(self.rutas.IMGS_PATH, str(fold))
            items = len(os.listdir(path_folder))
            msg = f"Carpeta '{fold}': {items} elementos"
            print(msg)
        return

    def show_images_same_folder(
        self,
        nfolds:list[str],
        nimages:int = 5,
        ) -> plt.figure:
        """Selecciona aleatoriamente una carpeta y muestra algunas imagenes"""

        plt.figure(figsize=(15,15))

        folder = np.random.randint(len(nfolds))
        path = os.path.join(self.rutas.IMGS_PATH, str(folder))
        imgs_to_show = random.sample(os.listdir(path), nimages)

        for idx, nombreimg in enumerate(imgs_to_show):
            plt.subplot(1, nimages, idx+1)
            plt.title(folder)
            imagen = mpimg.imread(os.path.join(path, nombreimg)) # mpimg.imread lee los pixeles
            plt.imshow(imagen)

        return plt.figure

    def show_images_different_folder(
        self,
        nfolds:list[str],
        nimages:int = 5,
        ) -> plt.figure:
        """Selecciona aleatoriamente una carpeta y muestra algunas imagenes"""

        plt.figure(figsize=(15,15))

        imgs_to_show = []
        for _ in range(nimages):
            folder = np.random.randint(len(nfolds))
            path = os.path.join(self.rutas.IMGS_PATH, str(folder))
            nombreimg = random.sample(os.listdir(path), 1)
            fullpath = os.path.join(path, nombreimg[0])
            imgs_to_show.append(fullpath)

        for idx, nombreimg in enumerate(imgs_to_show):
            plt.subplot(1, nimages, idx+1)
            plt.title(nombreimg.split("IMG")[0][-2])
            imagen = mpimg.imread(os.path.join(path, nombreimg)) # mpimg.imread lee los pixeles
            plt.imshow(imagen)

        return plt.figure
