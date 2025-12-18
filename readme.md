# Electric Arc Furnace (EAF) – Análisis y Predicción Industrial

Este proyecto implementa un sistema de analítica avanzada para predecir la temperatura y la composición química (azufre) en un Horno de Arco Eléctrico.

## Dataset Utilizado
El proyecto utiliza el dataset público **“Industrial data from the arc furnace”** de Kaggle.

- **Enlace:** [Kaggle - Industrial data from the arc furnace](https://www.kaggle.com/datasets/yuriykatser/industrial-data-from-the-arc-furnace)
- **Reproducibilidad:** El notebook `EAF_notebook.ipynb` incluye una celda de descarga automática que obtiene los archivos directamente mediante la API de Kaggle si no se encuentran en la carpeta `data/raw/`.

## 🛠️ Requisitos y Dependencias
El proyecto requiere Python 3.9+ y las dependencias especificadas en `requirements.txt`.

### Instalación
1. Clona el repositorio.
2. Crea un entorno virtual:
   - Windows: `python -m venv venv` y luego `venv\Scripts\activate`
3. Instala las dependencias:
   - `pip install -r requirements.txt`

### Ejecución
Para iniciar la interfaz visual (dashboard):
- `streamlit run dashboard/app.py`

No es necesario ejecutar BentoML por separado, ya que hemos implementado un sistema que despliega automáticamente la instancia de BentoML desde el propio dashboard.

## 📂 Estructura del Repositorio
- **EAF_notebook.ipynb**: Análisis exploratorio, ingeniería de variables y entrenamiento de modelos.
- **dashboard/**: Código de la interfaz visual.
- **deploy/**: Definición del servicio de modelos con BentoML.
- **models/**: Almacenamiento de modelos entrenados.
- **data/**: Directorios para datos crudos (`raw`) y procesados (`processed`).
