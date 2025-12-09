import bentoml
import numpy as np
import pandas as pd

# 1. Definimos el servicio
@bentoml.service(
    name="eaf_temperature_service",
    traffic={"timeout": 10},
    resources={"cpu": "1"} # Usamos CPU básica
)
class EAFModel:
    # 2. Recuperamos el modelo guardado anteriormente
    bento_model = bentoml.models.BentoModel("eaf_temperature_model:latest")

    def __init__(self):
        self.model = self.bento_model.load_model()

    # 3. Definimos el punto de entrada (API)
    @bentoml.api
    def predict(self, inputs: pd.DataFrame) -> float:
        """
        Recibe un DataFrame con las mismas columnas que el entrenamiento.
        Devuelve la temperatura predicha.
        """
        # XGBoost espera un DMatrix o DataFrame/Numpy directo
        prediction = self.model.predict(inputs)
        
        # Devolvemos el primer valor (como float nativo de Python)
        return float(prediction[0])