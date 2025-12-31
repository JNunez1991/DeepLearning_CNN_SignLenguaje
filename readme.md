<div>
  <img src="/imgs/header.jpg" width="120" align="right">

  <h1 align="center">Sign Lenguaje</h1>
  <p align="center">Entrenamiento y predicción en tiempo real de lenguaje de señas</p>
</div>


## 📌 Descripción
Proyecto que busca entrenar una red neuronal convolucional (CNN) que permita predecir, en tiempo real, los numeros del 0 al 9 en lenguaje de señas, mediante la utilización de la camara incorporada al pc.

## 🚀 Funcionalidades Clave
* **Análisis de Datos:** Analisis de las imagenes que se utilizan para el entrenamiento, y generacion de carpetas (una por categoria).
* **Modelado:** Implementación de algoritmos de Deep Learning + Data Augmentation (opcional) + Transfer Learning (Opcional)
* **Visualización:** Prediccion en tiempo real mediante keras + OpenCV.

## 🛠️ Stack Tecnológico
* **Lenguaje:** Python 3.11.7
* **Librerías principales:** Tensorflow 2.20.0, Keras 3.12.0, Scikit-learn 1.8-0, Numpy 1.26.4
* **Versionado:** GitHub.

## 📋 Estructura del Repositorio
```text
├── data/
  ├── raw                 # Contiene las subcarpetas con las imagenes
  ├── processed           # Contiene las carpetas de train, validation y test
├── notebooks/
  ├── SignLenguaje.ipynb  # jupyter notebooks que ejecuta main_train.py (entrenamiento de modelos)
├── models/               # carpeta donde se almacenan los modelos luego de entrenados
├── logs/                 # logs generados durante el entrenamiento de los modelos
├── src/
  ├── entrenamiento
    ├── augmentation.py   # para aumento de datos
    ├── blueprint.py      # clases plantilla
    ├── model.py          # modelado
    ├── model_predict.py  # prediccion sobre imagenes de test
    ├── preprocess.py     # distribuye las imagenes en subcarpetas de train/test
    ├── visualize.py      # visualizacion de imagenes
  ├── realtime
    ├── controller.py     # orquestador del modulo
    ├── hand_detection.py # deteccion de manos en el frame
    ├── prediction.py     # prediccion sobre la imagen del frame
    ├── roi.py            # genera la region de interes para la prediccion
    ├── text.py           # textos en pantalla
├── imgs/                 # imagenes auxiliares
├── main_train.py         # logica para el entrenamiento de los modelos
├── main_realtime.py      # logica para cargar modelo, iniciar la camara y predecir realtime
├── config.py             # contiene variables estátticas como rutas o nombres
├── requirements.txt      # Dependencias del proyecto
└── README.md             # Documentación principal
```

## 📋 Funcionamiento
* **Entrenamiento**
Acceder al archivo `SignLenguaje.ipynb` dentro de la carpeta `Notebook`.
Alli se ejecuta, paso a paso, la logica necesaria para entrenar hasta 4 modelos diferentes.
Luego de entrenado, cada modelo se guarda en la carpeta `models`, y sus correspondientes logs en la carpeta `logs`.

* **Prediccion tiempo real**
Ejecutar `main_realtime.py`.
Se abrirá consola que le preguntará al usuario que modelo quiere utilizar.
Luego de seleccionado el modelo, se inicializa la camara y comienza la prediccion en tiempo real.

<div align="center">
  <a href="https://www.youtube.com/watch?v=Srxq49WOVNk">
    <img src="https://img.youtube.com/vi/Srxq49WOVNk/0.jpg" width="200">
  </a>
</div>
