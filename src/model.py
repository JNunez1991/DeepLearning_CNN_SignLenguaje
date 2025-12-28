#!/usr/bin/env python3
#pylint: disable=line-too-long
"""Modelo con el que se va a entrenar"""

import os
from dataclasses import dataclass, field
from typing import Protocol
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from keras.applications.vgg19 import VGG19
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, Dropout#, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam #, Nadam


class RutasProtocol(Protocol):
    """Protocolo de rutas"""

    IMGS_PATH: str
    TRAIN_PATH: str
    VAL_PATH: str
    MODEL_PATH: str


@dataclass
class Model:
    """Definicion del modelo a utilizar"""

    rutas: RutasProtocol
    nfolders:int
    modelname:str
    version:str
    img_size:tuple[int, int]
    epochs:int
    batch_size:int
    use_tl:bool = False     # Transfer Learning
    mode:str = "max"
    metric:str = 'accuracy'
    label:str = "categorical"
    monitor:str = "val_accuracy"
    loss:str = 'categorical_crossentropy'

    full_model_name:str = field(init=False)

    def __post_init__(self):
        """Se ejecuta luego de instanciar la clase"""

        print("")
        print(f"-. Realizando entrenamiento con modelo {self.version.upper()}...")
        self.full_model_name = os.path.join(self.rutas.MODEL_PATH, self.modelname)

    def run_all(self) -> Sequential:
        """Ejecuta todas las instancias del modelado"""

        data_train = self.get_data(self.rutas.TRAIN_PATH)
        data_val = self.get_data(self.rutas.VAL_PATH)
        tensorboard, model_checkpoint = self.set_structures()

        nitems_train = len( list(Path(self.rutas.TRAIN_PATH).glob('*/*')) )
        nitems_val = len( list(Path(self.rutas.VAL_PATH).glob('*/*')) )

        model = self.set_model()
        model = self.fit_model(
            model=model,
            data_train=data_train,
            data_val=data_val,
            cant_train=nitems_train,
            cant_val=nitems_val,
            tensorboard=tensorboard,
            checkpoint=model_checkpoint,
        )
        return model

    def get_data(self, path:str) -> tf.data.Dataset:
        """Cargo las imagenes de train y test"""

        data = keras.utils.image_dataset_from_directory(
            path,
            image_size=self.img_size,
            batch_size=self.batch_size,
            label_mode=self.label,
        )
        return data # type:ignore

    def set_structures(self) -> tuple[TensorBoard, ModelCheckpoint]:
        """Setea las estructuras necesarias"""

        tensorbrd = TensorBoard(log_dir=f'logs/model_{self.version}')
        checkpoint = ModelCheckpoint(
            self.full_model_name,
            mode=self.mode,
            monitor=self.monitor,
            save_best_only=True,
        )
        return tensorbrd, checkpoint

    def set_model(self, dropout:float = 0.25) -> Sequential:
        """Capas del modelo"""

        # al tamaño de las imagenes, le agrego los canales de color
        input_size_with_channels = (*self.img_size, 3)

        if self.use_tl:
            tl_base = VGG19(
                 weights='imagenet',
                 input_shape = input_size_with_channels,
                 include_top = False,
            )
            tl_base.trainable = False
            model = Sequential(
                [
                    tl_base,
                    Dropout(dropout),
                    Flatten(),
                    Dense(self.nfolders, activation="softmax")
            ])
        else:
            model = Sequential(
                [
                    Input(shape=input_size_with_channels),

                    Conv2D(32, kernel_size = (3, 3), padding = 'same', activation='relu'),
                    MaxPooling2D(pool_size = (2, 2)),

                    Conv2D(64, kernel_size = (3, 3), padding = 'same', activation="relu"),
                    MaxPooling2D(pool_size = (2, 2)),

                    Conv2D(128, kernel_size = (3, 3), padding = 'same', activation="relu"),
                    MaxPooling2D(pool_size = (2, 2)),

                    Dropout(dropout),
                    Flatten(),
                    Dense(self.nfolders, activation = "softmax")
                ],
            )

        opt = Adam(learning_rate=0.001)
        model.compile(
            loss = self.loss,
            optimizer = opt, # type:ignore
            metrics = [self.metric],
        )
        return model

    def fit_model(
        self,
        model:Sequential,
        data_train:tf.data.Dataset,
        data_val:tf.data.Dataset,
        cant_train:int,
        cant_val:int,
        tensorboard:TensorBoard,
        checkpoint:ModelCheckpoint,
    ) -> Sequential:
        """Ajusta el modelo"""

        model.fit(
            data_train,
            validation_data=data_val,
            epochs=self.epochs,
            batch_size=self.batch_size,
            steps_per_epoch=int( np.ceil(cant_train / float(self.batch_size)) ),
            validation_steps=int( np.ceil(cant_val / float(self.batch_size)) ),
            callbacks=[tensorboard, checkpoint]
        )
        return model

    # def evaluate_model(self):
    #     """Evalua el modelo con la mejor performance"""

    #     print("")
    #     print(f"-. Cargando mejor modelo {self.version.upper}...")
    #     data_val = self.get_data(self.rutas.VAL_PATH)
    #     best_model = load_model(self.full_model_name)
    #     scores = best_model.evaluate(data_val, verbose=1) # type:ignore
    #     print('   -. Val loss:', scores[0])
    #     print('   -. Val accuracy:', scores[1])


@dataclass
class EvalModel:
    """Evalua un modelo luego de entrenado"""

    rutas: RutasProtocol
    modelname:str
    img_size:tuple[int, int]
    batch_size:int
    label:str = "categorical"

    def run_all(self, modelname:str):
        """Evalua el modelo con la mejor performance"""

        print(f"-. Evaluando resultados del mejor modelo {modelname.upper()}...")
        model_name = f"model_{modelname}.keras"
        full_model_name = os.path.join(self.rutas.MODEL_PATH, model_name)

        data_val = self.get_data(self.rutas.VAL_PATH)
        best_model = load_model(full_model_name)

        scores = best_model.evaluate(data_val, verbose=1) # type:ignore
        print('   -. Val loss:', scores[0])
        print('   -. Val accuracy:', scores[1])

    def get_data(self, path:str) -> tf.data.Dataset:
        """Cargo las imagenes de train y test"""

        data = keras.utils.image_dataset_from_directory(
            path,
            image_size=self.img_size,
            batch_size=self.batch_size,
            label_mode=self.label,
        )
        return data # type:ignore
