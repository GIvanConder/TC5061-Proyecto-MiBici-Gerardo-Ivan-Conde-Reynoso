"""Pruebas de la API REST de clustering MiBici P3."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from serving.app import app


EJEMPLO_PATH = (
    Path(__file__).resolve().parents[1]
    / "serving/models/p3/input_example.json"
)

PREDICCIONES_ESPERADAS = [3, 1, 1, 3, 3]


@pytest.fixture(scope="module")
def client():
    # El contexto ejecuta el lifespan y carga el modelo real.
    with TestClient(app) as cliente:
        yield cliente


@pytest.fixture(scope="module")
def ejemplos():
    contenido = json.loads(EJEMPLO_PATH.read_text(encoding="utf-8"))
    return [
        dict(zip(contenido["columns"], fila))
        for fila in contenido["data"]
    ]


def test_health(client):
    respuesta = client.get("/health")

    assert respuesta.status_code == 200
    assert respuesta.json()["status"] == "ok"
    assert respuesta.json()["model_version"] == 1


@pytest.mark.parametrize(
    "indice,esperado",
    list(enumerate(PREDICCIONES_ESPERADAS)),
)
def test_predicciones_mlflow(client, ejemplos, indice, esperado):
    respuesta = client.post("/predict", json=ejemplos[indice])

    assert respuesta.status_code == 200
    assert respuesta.json() == {
        "cluster": esperado,
        "model_name": "mibici-equipo32-p3-grupos",
        "model_version": 1,
    }


def test_reproducibilidad(client, ejemplos):
    primera = client.post("/predict", json=ejemplos[0])
    segunda = client.post("/predict", json=ejemplos[0])

    assert primera.status_code == 200
    assert segunda.status_code == 200
    assert primera.json() == segunda.json()


def test_variable_faltante(client, ejemplos):
    datos = ejemplos[0].copy()
    del datos["salidas_dia"]

    respuesta = client.post("/predict", json=datos)

    assert respuesta.status_code == 422


def test_variable_adicional(client, ejemplos):
    datos = ejemplos[0].copy()
    datos["variable_desconocida"] = 123

    respuesta = client.post("/predict", json=datos)

    assert respuesta.status_code == 422


def test_valor_no_numerico(client, ejemplos):
    datos = ejemplos[0].copy()
    datos["salidas_dia"] = "texto"

    respuesta = client.post("/predict", json=datos)

    assert respuesta.status_code == 422


@pytest.mark.parametrize(
    "valor",
    ["NaN", "Infinity", "-Infinity"],
)
def test_valor_no_finito(client, ejemplos, valor):
    datos = ejemplos[0].copy()
    datos["salidas_dia"] = valor

    respuesta = client.post("/predict", json=datos)

    assert respuesta.status_code == 422
