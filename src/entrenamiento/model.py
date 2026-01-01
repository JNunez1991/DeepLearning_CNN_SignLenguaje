#!/usr/bin/env python3
#pylint: disable=line-too-long
"""Modelo con el que se va a entrenar"""

import os
from dataclasses import dataclass, field
from pathlib import Path
import keras
import numpy as np
import tensorflow as tf
# from keras import backend
from keras.applications.vgg19 import VGG19
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, Dropout#, Activation
from keras.models import Sequential, load_model, Model
from keras.optimizers import Adam, Nadam

from .blueprint import RutasProtocol, Callbacks#, Colores


@dataclass
class RunModel:
    """Definicion del modelo a utilizar"""

    rutas: RutasProtocol
    nfolders:int
    modelname:str
    version:str
    img_size:tuple[int, ...]
    epochs:int
    batch_size:int
    use_tl:bool = False     # Transfer Learning
    patience: int = 8       # para early stopping
    lr:float = 0.001
    dropout:float = 0.25
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
        nitems_train = len( list(Path(self.rutas.TRAIN_PATH).glob('*/*')) )
        nitems_val = len( list(Path(self.rutas.VAL_PATH).glob('*/*')) )

        callbacks = self.set_callbacks()
        optimizer = Adam(learning_rate=self.lr)
        model = self.model_set()
        self.model_compile(model, optimizer)
        model = self.model_fit(
            model=model, #type:ignore
            data_train=data_train,
            data_val=data_val,
            cant_train=nitems_train,
            cant_val=nitems_val,
            callbacks=callbacks,
        )
        return model

    def get_data(self, path:str) -> tf.data.Dataset:
        """Cargo las imagenes de train y test"""

        data = keras.utils.image_dataset_from_directory(
            path,
            image_size=self.img_size[:2],
            batch_size=self.batch_size,
            label_mode=self.label,
            # color_mode=Colores.GRAYSCALE, # si quisiera pasarlas a escala de grises
        )
        return data # type:ignore

    def set_callbacks(self) -> Callbacks:
        """Setea las estructuras necesarias"""

        tensorboard = TensorBoard(log_dir=f'logs/model_{self.version}')
        checkpoint = ModelCheckpoint(
            self.full_model_name,
            mode=self.mode,
            monitor=self.monitor,
            save_best_only=True,
        )

        early_stop = EarlyStopping(
            monitor=self.monitor,
            patience=self.patience,
            mode=self.mode,
            restore_best_weights=True
        )

        return Callbacks(
            tensorboard=tensorboard,
            checkpoint=checkpoint,
            early_stop=early_stop,
        )

    def model_set(self) -> Sequential:
        """Capas del modelo"""

        # Limpio los weights que hayan quedado en memoria
        # backend.clear_session()

        if self.use_tl:
            tl_base = VGG19(
                 weights='imagenet',
                 include_top = False,
                 input_shape = self.img_size, # no requiere pasarle las imagenes, solo su tamaño
            )
            tl_base.trainable = False
            model = Sequential(
                [
                    tl_base,
                    Dropout(self.dropout),
                    Flatten(),
                    Dense(self.nfolders, activation="softmax")
                ]
            )
            return model

        model = Sequential(
            [
                Input(shape=self.img_size),

                Conv2D(32, kernel_size = (3, 3), padding = 'same', activation='relu'),
                MaxPooling2D(pool_size = (2, 2)),

                Conv2D(64, kernel_size = (3, 3), padding = 'same', activation="relu"),
                MaxPooling2D(pool_size = (2, 2)),

                Conv2D(128, kernel_size = (3, 3), padding = 'same', activation="relu"),
                MaxPooling2D(pool_size = (2, 2)),

                Dropout(self.dropout),
                Flatten(),
                Dense(self.nfolders, activation = "softmax")
            ],
        )
        return model

    def model_compile(self, model:Model, optimizer: Adam | Nadam) -> None:
        """Defino las variables de compilacion del modelo"""

        model.compile(
            loss = self.loss,
            optimizer = optimizer, #type:ignore
            metrics = [self.metric],
        )

    def model_fit(
        self,
        model:Sequential,
        data_train:tf.data.Dataset,
        data_val:tf.data.Dataset,
        cant_train:int,
        cant_val:int,
        callbacks:Callbacks,
    ) -> Sequential:
        """Ajusta el modelo"""

        model.fit(
            data_train,
            validation_data=data_val,
            epochs=self.epochs,
            batch_size=self.batch_size,
            steps_per_epoch=int( np.ceil(cant_train / float(self.batch_size)) ),
            validation_steps=int( np.ceil(cant_val / float(self.batch_size)) ),
            callbacks=[callbacks.tensorboard, callbacks.checkpoint, callbacks.early_stop]
        )
        return model


@dataclass
class EvalModel:
    """Evalua un modelo luego de entrenado"""

    rutas: RutasProtocol
    modelname:str
    img_size:tuple[int, ...]
    batch_size:int
    label:str = "categorical"

    def run_all(self, modelname:str) -> None:
        """Evalua el modelo con la mejor performance"""

        print(f"-. Evaluando resultados del mejor modelo {modelname.upper()}...")
        model_name = f"model_{modelname}.keras"
        full_model_name = os.path.join(self.rutas.MODEL_PATH, model_name)

        data_val = self.get_data(self.rutas.VAL_PATH)
        best_model = load_model(full_model_name)

        best_model.evaluate(data_val, verbose=1) # type:ignore
        print("")
        # print('   -. Val loss:', scores[0])
        # print('   -. Val accuracy:', scores[1])

    def get_data(self, path:str) -> tf.data.Dataset:
        """Cargo las imagenes de train y test"""

        data = keras.utils.image_dataset_from_directory(
            path,
            image_size=self.img_size[:2],
            batch_size=self.batch_size,
            label_mode=self.label,
            # color_mode=Colores.GRAYSCALE,
        )
        return data # type:ignore
