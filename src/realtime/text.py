#!/usr/bin/env python3
# pylint: disable=no-name-in-module, no-member
"""Textos en pantalla"""

from dataclasses import dataclass

import cv2
import numpy as np

@dataclass
class TextInFrame:
    """Carga el mejor modelo, y lo utiliza para predecir en real time"""

    def header(
        self,
        frame:np.ndarray,
        text:str = "Presiona 'q' para salir.",
        pos:tuple[int, int]=(20, 40),
        font:int=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale:float=0.5, # tamaño de letra
        font_color:tuple[int,...]=(255, 255, 255), #color blanco
        font_thickness:int=1, # 1: sin negrita, 2: negrita
        bg_padding:int=6,
        bg_color:tuple[int,...]=(0,0,0), #color negro
    ) -> None:
        """Rectangulo negro con letras blancas para header"""
        x, y = pos

        # Tamaño del texto
        (text_w, text_h), baseline = cv2.getTextSize(
            text, font, font_scale, font_thickness
        )

        # Rectángulo de fondo negro
        cv2.rectangle(
            frame,
            (x - bg_padding, y - text_h - bg_padding),
            (x + text_w + bg_padding, y + baseline + bg_padding),
            bg_color,
            -1
        )

        cv2.putText(
            frame,
            text,
            (x, y),
            font,
            font_scale,
            font_color,   # letras blancas
            font_thickness,
            cv2.LINE_AA, # suaviza los bordes
        )


    def info(
        self,
        frame:np.ndarray,
        prediction:int,
        porc:float,
        pos:tuple[int, int]=(20, 70),
        font:int=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale:float=0.5, # tamaño de letra
        font_color:tuple[int,...]=(0,0,0), # letras negras
        font_thickness:int=1, # 1: sin negrita, 2: negrita
        bg_padding:int=6,
        bg_color:tuple[int,...]=(255,255,255), #background blanco
    ) -> None:
        """Rectangulo negro con letras blancas para header"""
        x, y = pos

        msg = f"Prediccion: {prediction}, Probabilidad: {porc*100:.2f}%"

        # Tamaño del texto
        (text_w, text_h), baseline = cv2.getTextSize(
            msg, font, font_scale, font_thickness
        )

        # Rectángulo de fondo negro
        cv2.rectangle(
            frame,
            (x - bg_padding, y - text_h - bg_padding),
            (x + text_w + bg_padding, y + baseline + bg_padding),
            bg_color,
            -1
        )

        cv2.putText(
            frame,
            msg,
            (x, y),
            font,
            font_scale,
            font_color,   # letras blancas
            font_thickness,
            cv2.LINE_AA, # suaviza los bordes
        )
