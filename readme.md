# Electric Arc Furnace (EAF) - Análisis y Predicción Industrial

Este proyecto implementa un sistema de analítica avanzada para predecir la temperatura y la composición química (Azufre) en un Horno de Arco Eléctrico.

## Dataset Utilizado
El proyecto utiliza el dataset público **"Industrial data from the arc furnace"** de Kaggle.
* **Enlace:** [Kaggle - Industrial data from the arc furnace](https://www.kaggle.com/datasets/yuriykatser/industrial-data-from-the-arc-furnace)
* **Nota:** El notebook `EAF_notebook.ipynb` incluye una celda de descarga automática que baja los archivos directamente vía API si no se encuentran en la carpeta `data/raw/`.

##  Requisitos y Dependencias
El proyecto requiere Python 3.9+ y las dependencias listadas en `requirements.txt`.

### Instalación
1. Clona el repositorio o descomprime el archivo.
2. Crea un entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate