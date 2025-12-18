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

    train_size: int = 180

    def run_all(
        self,
        path:str,
        header:str,
        aug:bool=False,
    ) -> list[str]:
        """Se encarga de ejecutar el paso a paso del proyecto"""

        print("")
        print(header)
        nfolders = os.listdir(path)
        self.folders_components(nfolders, path)
        self.show_images_same_folder(nfolders, path, aug)
        self.show_images_different_folder(nfolders, path, aug)
        return nfolders

    def folders_components(
        self,
        nfolds:list[str],
        path:str,
    ) -> None:
        """Imprime por consola la cantidad de imagenes en cada carpeta"""

        for fold in nfolds:
            path_folder = os.path.join(path, str(fold))
            items = len(os.listdir(path_folder))
            msg = f"Carpeta '{fold}': {items} elementos"
            print(msg)
        return

    def show_images_same_folder(
        self,
        nfolds:list[str],
        path: str,
        aug:bool = False,
        nimages:int = 5,
    ) -> None:
        """Selecciona aleatoriamente una carpeta y muestra algunas imagenes"""

        plt.figure(figsize=(15,15))

        folder = np.random.randint(len(nfolds))
        path = os.path.join(path, str(folder))
        imgs_to_show = random.sample(os.listdir(path), nimages)

        for idx, nombreimg in enumerate(imgs_to_show):
            plt.subplot(1, nimages, idx+1)
            title = f"augm_{str(folder)}" if aug else str(folder)
            plt.title(title)
            imagen = mpimg.imread(os.path.join(path, nombreimg)) # mpimg.imread lee los pixeles
            plt.imshow(imagen)

        return

    def show_images_different_folder(
        self,
        nfolds:list[str],
        path:str,
        aug:bool = False,
        nimages:int = 5,
    ) -> None:
        """Selecciona aleatoriamente una carpeta y muestra algunas imagenes"""

        plt.figure(figsize=(15,15))

        imgs_to_show = []

        for _ in range(nimages):
            folder = np.random.randint(len(nfolds))
            folder_path = os.path.join(path, str(folder))
            nombreimg = random.sample(os.listdir(folder_path), 1)
            fullpath = os.path.join(path, str(folder), nombreimg[0])
            imgs_to_show.append(fullpath)

        for idx, nombreimg in enumerate(imgs_to_show):
            plt.subplot(1, nimages, idx+1)
            if aug:
                msg = nombreimg.split('train')[1][1]
                print(msg)
                title = f"augm_{msg}"
            else:
                title = nombreimg.split("IMG")[0][-2]
            plt.title(title)
            path = os.path.join(path, nombreimg)
            imagen = mpimg.imread(path) # mpimg.imread lee los pixeles
            plt.imshow(imagen)
