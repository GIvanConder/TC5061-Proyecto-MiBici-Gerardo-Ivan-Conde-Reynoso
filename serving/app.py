"""API REST P3 con Pipeline completo y metadata del Registry."""
import json
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request

from serving.schemas import P3Input, P3Prediction

MODEL_PATH = Path(__file__).resolve().parent / "models" / "p3"


def leer_metadata(directorio):
    directorio = Path(directorio).resolve()
    ruta = directorio / "deployment_metadata.json"

    if not ruta.is_file():
        raise RuntimeError(
            "No existe metadata de despliegue. "
            "Ejecutar serving.descargar_modelo_p3 primero."
        )

    datos = json.loads(ruta.read_text(encoding="utf-8"))

    campos = {
        "model_name", "model_version", "model_alias",
        "run_id", "input_semantics", "feature_names",
        "file_sha256",
    }
    if not campos.issubset(datos):
        raise RuntimeError("Metadata de modelo incompleta")

    if (
        not isinstance(datos["model_name"], str)
        or not datos["model_name"].strip()
    ):
        raise RuntimeError("Nombre de modelo invalido")

    if type(datos["model_version"]) is not int:
        raise RuntimeError("Version de modelo invalida")

    if datos["model_version"] < 1:
        raise RuntimeError("Version de modelo invalida")

    if datos["input_semantics"] != "raw_p3_features":
        raise RuntimeError(
            "El artefacto no admite variables originales"
        )

    if datos["feature_names"] != list(P3Input.model_fields):
        raise RuntimeError("Esquema API / modelo incompatible")

    from serving.descargar_modelo_p3 import verificar_archivos
    verificar_archivos(directorio, datos["file_sha256"])

    return datos


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cargar y validar el artefacto una sola vez al arrancar."""
    app.state.metadata = leer_metadata(MODEL_PATH)
    app.state.model = mlflow.pyfunc.load_model(str(MODEL_PATH))

    yield

    del app.state.model
    del app.state.metadata


app = FastAPI(
    title="MiBici - API de Clustering P3",
    description=(
        "Servicio de inferencia P3. Recibe 12 caracteristicas "
        "originales; el artefacto aplica log1p y StandardScaler "
        "antes de KMeans."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request):
    metadata = request.app.state.metadata
    return {
        "status": "ok",
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
    }


@app.post("/predict", response_model=P3Prediction)
def predict(datos: P3Input, request: Request):
    """Agrupar una estación a partir de características originales."""
    columnas = list(P3Input.model_fields)

    entrada = pd.DataFrame(
        [datos.model_dump()],
        columns=columnas,
    ).astype("float64")

    resultado = request.app.state.model.predict(entrada)
    cluster = int(np.asarray(resultado).reshape(-1)[0])

    metadata = request.app.state.metadata

    return P3Prediction(
        cluster=cluster,
        model_name=metadata["model_name"],
        model_version=metadata["model_version"],
    )
