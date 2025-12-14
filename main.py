#!/usr/bin/env python3
"""Orquestador principal del proyecto"""

import os
import shutil
from dataclasses import dataclass

# import imageio
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# import tensorflow.keras as keras
# from keras.models import load_model
# from keras.callbacks import ModelCheckpoint
# # from keras.preprocessing.image import ImageDataGenerator
# from matplotlib import pyplot as plt, image as mpimg
# from skimage import io, transform
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Activation
# from tensorflow.keras.optimizers import Adam, Nadam
# from tensorflow.keras.callbacks import TensorBoard

from config import Rutas
from src import PreProcess, Visualization


@dataclass
class Main:
    """Clase central del proyecto"""

    def run_all(self) -> None:
        """Se encarga de ejecutar el paso a paso del proyecto"""

        # Cantidad de imagenes por subcarpeta, e imagenes de prueba
        viz = Visualization(Rutas)
        folders = viz.run_all()

        # Generacion de Train/Test y Data Augmentation
        preproc = PreProcess(Rutas)
        preproc.run_all(folders)


        # # Elimino las imagenes de Train/Test para limpieza
        # self.delete_train_test_items(folders)



    def delete_train_test_items(self, nfolds:list[str]) -> None:
        """Elimina todas las imagenes en las carpetas de Train/Test"""

        for fold in nfolds:
            shutil.rmtree(os.path.join(Rutas.TRAIN_PATH, fold))
            shutil.rmtree(os.path.join(Rutas.TEST_PATH, fold))


if __name__ == "__main__":

    main = Main()
    main.run_all()
