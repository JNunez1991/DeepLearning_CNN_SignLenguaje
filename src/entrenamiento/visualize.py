#!/usr/bin/env python3
"""Pre-Procesa las imagenes"""

import os
import random
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt, image as mpimg


@dataclass
class Visualization:
    """Clase que se encarga de analizar y pre-procesar las imagenes"""

    path:str
    nfolders:list[str]
    augmented:bool = False

    def __post_init__(self):
        """Se ejecuta luego de instanciar la clase"""

        print("-. Generando imagenes...")

    def run_all(self)-> None:
        """Se encarga de ejecutar el paso a paso del proyecto"""

        print(self.set_console_text())
        self.items_in_folder()
        self.show_images_same_folder()
        self.show_images_different_folder()

    def set_console_text(self) -> str:
        """Ajusta el texto en consola"""

        if self.augmented:
            hdr = "- Imagenes de entrenamiento con data augmentation..."
        else:
            hdr = "-. Imagenes Reales para entrenamiento..."

        return "\n" + hdr

    def items_in_folder(self) -> None:
        """Imprime por consola la cantidad de imagenes en cada carpeta"""

        for fold in self.nfolders:
            path_folder = os.path.join(self.path, str(fold))
            items = len(os.listdir(path_folder))
            msg = f"Carpeta '{fold}': {items} elementos"
            print(msg)

    def show_images_same_folder(self, nimages:int = 5) -> None:
        """Selecciona aleatoriamente una carpeta y muestra algunas imagenes"""

        plt.figure(figsize=(15,15))
        plt.suptitle("Imagenes aleatorias dentro de la misma carpeta", y=0.6)

        folder = np.random.randint(len(self.nfolders))
        path = os.path.join(self.path, str(folder))
        imgs_to_show = random.sample(os.listdir(path), nimages)

        for idx, nombreimg in enumerate(imgs_to_show):
            plt.subplot(1, nimages, idx+1)
            title = f"augm_{str(folder)}" if self.augmented else str(folder)
            plt.title(title)
            imagen = mpimg.imread(os.path.join(path, nombreimg)) # mpimg.imread lee los pixeles
            plt.imshow(imagen)
            plt.axis("off")

    def show_images_different_folder(self, nimages:int = 5) -> None:
        """Selecciona aleatoriamente una carpeta y muestra algunas imagenes"""

        plt.figure(figsize=(15,15))
        plt.suptitle("Imagenes aleatorias dentro de distintas carpeta", y=0.6)

        imgs_to_show = []

        for _ in range(nimages):
            folder = np.random.randint(len(self.nfolders))
            folder_path = os.path.join(self.path, str(folder))
            nombreimg = random.sample(os.listdir(folder_path), 1)
            fullpath = os.path.join(self.path, str(folder), nombreimg[0])
            imgs_to_show.append(fullpath)

        for idx, nombreimg in enumerate(imgs_to_show):
            plt.subplot(1, nimages, idx+1)
            if self.augmented:
                msg = nombreimg.split('train')[1][1]
                title = f"augm_{msg}"
            else:
                title = nombreimg.split("IMG")[0][-2]
            plt.title(title)
            path = os.path.join(self.path, nombreimg)
            imagen = mpimg.imread(path) # mpimg.imread lee los pixeles
            plt.imshow(imagen)
            plt.axis("off")
