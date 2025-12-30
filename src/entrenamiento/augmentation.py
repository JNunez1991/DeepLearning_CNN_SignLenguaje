#!/usr/bin/env python3
"""Pre-Procesa las imagenes"""

import os
from dataclasses import dataclass

import keras
import tensorflow as tf
from tqdm import tqdm

from .blueprint import Colores


@dataclass
class DataAugmentation:
    """Clase que se encarga de analizar y pre-procesar las imagenes"""

    train_path:str
    img_size:tuple[int, ...]
    ncopies: int = 8

    def __post_init__(self):
        """Se ejecuta luego de instanciar la clase"""

        print("-. Aplicando Data Augmentation...")

    def run_all(self) -> None:
        """A las imagenes de cada subcarpeta, les aplico data augmentation"""

        dataset = self.set_dataset()
        file_paths = dataset.file_paths # type:ignore
        augm_model = self.set_augmentation_parameters()

        for idx, (image, _) in enumerate(tqdm(dataset, desc="Aumentando Dataset")):
            original_name = os.path.basename(file_paths[idx])[:-4] # quito la extension '.jpg'
            for j in range(self.ncopies):
                augmented_img = augm_model(image, training=True)
                final_img = keras.utils.array_to_img(augmented_img[0])
                save_name = f"{original_name}_augm{j+1}.jpg"
                final_img.save(os.path.join(os.path.dirname(file_paths[idx]), save_name))

    def set_dataset(self) -> tf.data.Dataset:
        """
        Utiliza image_dataset_from_directory, la cual toma como uno de sus parametros la
        carpeta padre donde estan almacenadas las imagenes. Es necesario que cada categoria
        tenga una subcarpeta dentro de ella. Dicha funcion detecta automaticamente la cantidad
        de imagenes por clase
        """

        data = keras.utils.image_dataset_from_directory(
            self.train_path,
            image_size=self.img_size[:2],
            color_mode=Colores.GRAYSCALE,
            batch_size=1,
            shuffle=False
        )
        return data # type:ignore

    def set_augmentation_parameters(self) -> keras.Sequential:
        """Aplico data augmentation sobre las imagenes"""
        return keras.Sequential([

            # GEOMÉTRICAS (imagen en [0,255])
            keras.layers.RandomRotation(0.08),
            keras.layers.RandomTranslation(0.1, 0.1),
            keras.layers.RandomZoom(0.1),
            keras.layers.RandomFlip("horizontal"),

            # COLOR / ILUMINACIÓN
            keras.layers.RandomContrast(0.1),
            keras.layers.RandomBrightness(0.1),

            # NORMALIZACIÓN
            keras.layers.Rescaling(1./255),

            # RUIDO (muy leve)
            keras.layers.GaussianNoise(0.02),
        ])
