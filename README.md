<div style="display: flex; justify-content: space-between; align-items: flex-start; width: 100%; border-bottom: 1px solid #333; padding-bottom: 10px;">

  <div style="display: flex; flex-direction: column;">
    <h1 style="margin: 0; font-size: 2.5em;">Sign Lenguaje</h1>
    <h4 style="margin: 5px 0 0 0; font-weight: normal; color: #888;">
      Entrenamiento y predicción en tiempo real de lenguaje de señas <br>
      Deep Learning - Convolutional Neural Network
    </h4>
  </div>

  <img src="/imgs/header.jpg" width="100" style="border-radius: 8px;">

</div>


## 📌 Descripción
Proyecto que busca entrenar una red neuronal convolucional (CNN) que permita predecir, en tiempo real, los numeros del 0 al 9 en lenguaje de señas, mediante la utilización de la camara incorporada al pc.

## ⚠️ Disclaimer
El proyecto no busca la excelencia.
Las imagenes utilizadas para el entrenamiento son de dimensiones (100, 100, 3), y en calidad baja, por lo que no es de esperar que los resultados de las predicciones sean infalibles.

## 🚀 Funcionalidades Clave
* **Procesamiento de imagenes:** Analisis de las imagenes que se utilizan para el entrenamiento, y generacion de carpetas (una por categoria).
* **Modelado:** Implementación de algoritmos de Deep Learning + Data Augmentation (opcional) + Transfer Learning (Opcional)
* **Visualización:** Prediccion en tiempo real mediante keras & OpenCV.

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

## 👣 Funcionamiento
* **Entrenamiento**
Acceder al archivo `SignLenguaje.ipynb` dentro de la carpeta `Notebook`.
Alli se ejecuta, paso a paso, la logica necesaria para entrenar hasta 4 modelos diferentes.
Luego de entrenado, cada modelo se guarda en la carpeta `models`, y sus correspondientes logs en la carpeta `logs`.

* **Prediccion tiempo real**
Ejecutar `main_realtime.py`.
Se abrirá consola que le preguntará al usuario que modelo quiere utilizar.
Luego de seleccionado el modelo, se inicializa la camara y comienza la prediccion en tiempo real.
Se adjunta video a Youtube con el funcionamiento de la interfaz.

## ▶️ Videos
* **Totalidad del proyecto**
<div align="center">
  <a href="https://www.youtube.com/watch?v=gF9f2Tq1E1A">
    <img src="https://img.youtube.com/vi/gF9f2Tq1E1A/0.jpg" width="200">
  </a>
</div>
