"""API REST para inferencia del modelo P3 de MiBici."""

from contextlib import asynccontextmanager
from pathlib import Path

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI

from serving.schemas import P3Input, P3Prediction


MODEL_NAME = "mibici-equipo32-p3-grupos"
MODEL_VERSION = 1

MODEL_PATH = Path(__file__).resolve().parent / "models" / "p3"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cargar el modelo una sola vez al iniciar la API."""
    app.state.model = mlflow.pyfunc.load_model(str(MODEL_PATH))
    yield
    del app.state.model


app = FastAPI(
    title="MiBici - API de Clustering P3",
    description=(
        "Servicio de inferencia con el modelo KMeans registrado "
        "en MLflow. Requiere 12 caracteristicas ya transformadas "
        "y estandarizadas."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    """Verificar que el servicio y el modelo estan disponibles."""
    return {
        "status": "ok",
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
    }


@app.post("/predict", response_model=P3Prediction)
def predict(datos: P3Input):
    """Asignar un cluster a las caracteristicas de una estacion."""
    columnas = list(P3Input.model_fields)

    entrada = pd.DataFrame(
        [datos.model_dump()],
        columns=columnas,
    )

    resultado = app.state.model.predict(entrada)

    return P3Prediction(
        cluster=int(resultado[0]),
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
    )
