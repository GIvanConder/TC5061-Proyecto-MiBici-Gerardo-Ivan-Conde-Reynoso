"""Pruebas del contrato de metadatos y la API dinámica."""
import hashlib
import importlib
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from serving.descargar_modelo_p3 import verificar_archivos

modulo = importlib.import_module("serving.app")


def test_detecta_artefacto_modificado(tmp_path):
    archivo = tmp_path / "modelo.bin"
    archivo.write_bytes(b"original")

    huella = hashlib.sha256(b"original").hexdigest()
    verificar_archivos(
        tmp_path, {"modelo.bin": huella}
    )

    archivo.write_bytes(b"alterado")

    with pytest.raises(ValueError, match="SHA256"):
        verificar_archivos(
            tmp_path, {"modelo.bin": huella}
        )


def test_rechaza_metadata_inexistente(tmp_path):
    with pytest.raises(RuntimeError, match="metadata"):
        modulo.leer_metadata(tmp_path)


def test_api_utiliza_metadata_del_artefacto(
    monkeypatch, tmp_path
):
    archivo = tmp_path / "modelo.bin"
    archivo.write_bytes(b"modelo-de-prueba")

    digest = hashlib.sha256(
        archivo.read_bytes()
    ).hexdigest()

    metadata = {
        "model_name": "modelo-prueba-dinamico",
        "model_version": 42,
        "model_alias": "serving",
        "run_id": "run-prueba",
        "input_semantics": "raw_p3_features",
        "feature_names": list(
            modulo.P3Input.model_fields
        ),
        "file_sha256": {
            "modelo.bin": digest,
        },
    }

    (tmp_path / "deployment_metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    class ModeloSimulado:
        def predict(self, entrada):
            assert isinstance(entrada, pd.DataFrame)
            assert entrada.shape == (1, 12)
            assert all(
                tipo == "float64"
                for tipo in entrada.dtypes.astype(str)
            )
            return [2]

    monkeypatch.setattr(modulo, "MODEL_PATH", tmp_path)
    monkeypatch.setattr(
        modulo.mlflow.pyfunc,
        "load_model",
        lambda ruta: ModeloSimulado(),
    )

    ejemplo = {
        campo: 0.5
        for campo in modulo.P3Input.model_fields
    }

    with TestClient(modulo.app) as cliente:
        salud = cliente.get("/health")
        assert salud.status_code == 200
        assert salud.json()["model_name"] == (
            "modelo-prueba-dinamico"
        )
        assert salud.json()["model_version"] == 42

        respuesta = cliente.post(
            "/predict", json=ejemplo
        )
        assert respuesta.status_code == 200
        assert respuesta.json() == {
            "cluster": 2,
            "model_name": "modelo-prueba-dinamico",
            "model_version": 42,
        }

        ejemplo["salidas_dia"] = -2
        invalida = cliente.post(
            "/predict", json=ejemplo
        )
        assert invalida.status_code == 422


def test_rechaza_ruta_fuera_del_artefacto(tmp_path):
    with pytest.raises(ValueError, match="no segura"):
        verificar_archivos(tmp_path, {"../fuera.bin": "abc"})


@pytest.mark.parametrize(
    "nombre,version",
    [
        ("", 2),
        ("modelo", True),
        ("modelo", 0),
    ],
)
def test_rechaza_identidad_de_despliegue_invalida(
    tmp_path, nombre, version
):
    from serving.schemas import P3Input

    archivo = tmp_path / "modelo.bin"
    archivo.write_bytes(b"modelo")

    meta = {
        "model_name": nombre,
        "model_version": version,
        "model_alias": "serving",
        "run_id": "run-test",
        "input_semantics": "raw_p3_features",
        "feature_names": list(P3Input.model_fields),
        "file_sha256": {
            "modelo.bin": hashlib.sha256(b"modelo").hexdigest()
        },
    }

    (tmp_path / "deployment_metadata.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )

    with pytest.raises(RuntimeError):
        modulo.leer_metadata(tmp_path)


def test_rechaza_proporcion_fuera_de_rango():
    from pydantic import ValidationError
    from serving.schemas import P3Input

    datos = {campo: 0.5 for campo in P3Input.model_fields}
    datos["share_fin_semana"] = 1.5

    with pytest.raises(ValidationError):
        P3Input.model_validate(datos)


def test_rechaza_balance_fuera_de_rango():
    from pydantic import ValidationError
    from serving.schemas import P3Input

    datos = {campo: 0.5 for campo in P3Input.model_fields}
    datos["balance_relativo"] = -1.5

    with pytest.raises(ValidationError):
        P3Input.model_validate(datos)
