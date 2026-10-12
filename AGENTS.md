# AGENTS.md - Proyecto CNN Sign Lenguaje

## Visión General
Reconocimiento de lenguaje de señas (dígitos 0-9) usando CNN con inferencia en tiempo real mediante webcam.

## Inicio Rápido

### Configuración del Entorno
```bash
# Usando uv (recomendado - existe uv.lock)
uv sync

# O usando pip
pip install -r requirements.txt
```

### Estructura del Proyecto
```
├── config.py              # Rutas, parámetros de imagen, nombres de modelos
├── main_train.py          # Orquestador de entrenamiento
├── main_realtime.py       # Punto de entrada inferencia tiempo real
├── src/
│   ├── train/             # Módulos de entrenamiento
│   │   ├── augmentation.py    # Aumento de datos
│   │   ├── blueprint.py       # Clases base
│   │   ├── model.py           # Arquitectura del modelo (4 variantes)
│   │   ├── model_predict.py   # Predicciones sobre test
│   │   ├── preprocess.py      # División train/val/test
│   │   └── visualize.py       # Visualización de datos
│   └── realtime/        # Módulos tiempo real
│       ├── controller.py      # Orquestación de cámara
│       ├── hand_detection.py  # Detección de manos MediaPipe
│       ├── prediction.py      # Predicción por frame
│       ├── roi.py             # Región de interés
│       └── text.py            # Texto en pantalla
├── data/
│   ├── raw/             # Imágenes fuente (organizadas por carpetas de clase)
│   └── processed/       # Divisiones train/val/test (generadas)
├── models/              # Modelos .keras guardados
├── logs/                # Logs de entrenamiento
└── notebook/            # Jupyter notebooks
```

## Comandos Clave

### Pipeline de Entrenamiento (vía notebook)
```bash
# Abrir notebooks/SignLenguaje.ipynb y ejecutar celdas secuencialmente
# O ejecutar main_train.py programáticamente:
from main_train import Main
main = Main()
main.preprocess_images(train_size=0.7, val_size=0.15, ncopies=8)
main.visualize_train_images()
model = main.execute_model(version="raw", use_tl=False)  # o "tl", "augmentation", "tlaugm"
main.evaluate_model("model_raw.keras")
main.predict("model_raw.keras")
```

### Inferencia en Tiempo Real
```bash
python main_realtime.py
# Seleccionar modelo (1-4) cuando se solicite
# Presionar 'q' para salir de la ventana de cámara
```

## Variantes de Modelo (config.ModelNames)
| Nombre | Descripción |
|--------|-------------|
| `raw` | CNN base |
| `tl` | Transfer Learning (MobileNetV2) |
| `augmentation` | CNN + Aumento de Datos |
| `tlaugm` | Transfer Learning + Aumento de Datos |

## Configuración (config.py)
- **Dimensiones imagen**: 100×100×3
- **Épocas**: 40
- **Batch size**: 32
- **Clases**: 10 (dígitos 0-9)

## Dependencias
- Python ≥3.11 (según .python-version)
- tensorflow≥2.20.0, keras≥3.12.0
- opencv-python, mediapipe (detección manos)
- numpy, matplotlib, scikit-learn, tqdm, pyfiglet

## Notas de Desarrollo
- Usa `uv` para gestión de dependencias (uv.lock presente)
- MediaPipe para detección de manos en módulo realtime
- Modelos guardados en formato `.keras` en `models/`
- Logs de entrenamiento en `logs/`
- Datos crudos esperados en `data/raw/<clase>/`