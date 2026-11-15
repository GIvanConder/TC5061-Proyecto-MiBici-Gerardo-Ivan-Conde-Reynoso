"""Descarga y verifica el modelo de servicio desde MLflow Registry.

La version real procede del alias indicado y queda fijada
en el artefacto empaquetado, no en constantes de FastAPI.
"""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient

from serving.schemas import P3Input

DESTINO = Path("serving/models/p3")
EJEMPLO = Path("serving/example_request.json")


def sha256(ruta):
    digest = hashlib.sha256()
    with open(ruta, "rb") as archivo:
        for bloque in iter(
            lambda: archivo.read(1024 * 1024), b""
        ):
            digest.update(bloque)
    return digest.hexdigest()


def verificar_archivos(directorio, huellas):
    """Comprueba la integridad del artefacto descargado."""
    base = Path(directorio).resolve()

    if not isinstance(huellas, dict) or not huellas:
        raise ValueError("Faltan huellas de integridad")

    for relativo, esperado in huellas.items():
        ruta_relativa = Path(relativo)

        if ruta_relativa.is_absolute() or ".." in ruta_relativa.parts:
            raise ValueError("Ruta de artefacto no segura")

        ruta = (base / ruta_relativa).resolve()

        if not ruta.is_relative_to(base):
            raise ValueError("Ruta fuera del artefacto")

        if not ruta.is_file():
            raise FileNotFoundError(ruta)

        if sha256(ruta) != esperado:
            raise ValueError(
                f"SHA256 no coincide: {relativo}"
            )


def extraer_ejemplo(directorio):
    ruta = Path(directorio) / "input_example.json"

    contenido = json.loads(
        ruta.read_text(encoding="utf-8")
    )

    esperado = list(P3Input.model_fields)

    if contenido.get("columns") != esperado:
        raise ValueError(
            "El ejemplo del modelo no coincide con el contrato API"
        )

    datos = pd.DataFrame(
        contenido["data"],
        columns=esperado,
    ).astype("float64")

    if datos.empty or not np.isfinite(datos.to_numpy()).all():
        raise ValueError("Ejemplo de inferencia invalido")

    # Validacion con el contrato Pydantic antes del empaquetado.
    for fila in datos.to_dict(orient="records"):
        P3Input.model_validate(fila)

    return datos


def ejecutar(nombre, alias, uri, destino=DESTINO):
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    cliente = MlflowClient()

    version = cliente.get_model_version_by_alias(
        nombre, alias
    )

    if version.tags.get("input_semantics") != "raw_p3_features":
        raise ValueError(
            "La version seleccionada no acepta features originales"
        )

    if version.tags.get("base_run_id") != (
        "c30b8d94d8f445e3a8abd1699676bfd9"
    ):
        raise ValueError(
            "El modelo no acredita linaje con P3 historico"
        )

    uri_modelo = f"models:/{nombre}/{version.version}"

    with tempfile.TemporaryDirectory(
        prefix="mibici-p3-",
        dir=str(destino.parent),
    ) as temporal:

        temporal = Path(temporal)
        descarga = temporal / "descarga"
        descarga.mkdir()

        descargado = Path(
            download_artifacts(
                artifact_uri=uri_modelo,
                dst_path=str(descarga),
                tracking_uri=uri,
                registry_uri=uri,
            )
        )

        if not (descargado / "MLmodel").is_file():
            raise ValueError(
                "La descarga no contiene un modelo MLflow valido"
            )

        nuevo = temporal / "nuevo"
        shutil.copytree(descargado, nuevo)

        ejemplo = extraer_ejemplo(nuevo)

        print("=== COMPROBAR MODELO DESCARGADO ===")
        local = mlflow.pyfunc.load_model(str(nuevo))
        remoto = mlflow.pyfunc.load_model(uri_modelo)

        original = np.asarray(remoto.predict(ejemplo))
        recuperado = np.asarray(local.predict(ejemplo))

        if not np.array_equal(original, recuperado):
            raise ValueError(
                "El artefacto local no reproduce el Registry"
            )

        if not np.isin(recuperado, [0, 1, 2, 3]).all():
            raise ValueError("Clusters fuera del rango esperado")

        huellas = {
            str(ruta.relative_to(nuevo)): sha256(ruta)
            for ruta in sorted(nuevo.rglob("*"))
            if ruta.is_file()
        }
        verificar_archivos(nuevo, huellas)

        metadata = {
            "model_name": version.name,
            "model_version": int(version.version),
            "model_alias": alias,
            "model_uri": uri_modelo,
            "run_id": version.run_id,
            "source": version.source,
            "git_commit": version.tags.get("git_commit"),
            "base_run_id": version.tags.get("base_run_id"),
            "input_semantics": version.tags["input_semantics"],
            "training_fingerprint": version.tags.get(
                "huella_scaler_sha256"
            ),
            "feature_names": list(P3Input.model_fields),
            "file_sha256": huellas,
        }

        (nuevo / "deployment_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )

        # Reemplazo con posibilidad de restaurar el modelo anterior.
        anterior = temporal / "anterior"
        habia_modelo = destino.exists()

        if habia_modelo:
            destino.rename(anterior)

        try:
            nuevo.rename(destino)
        except Exception:
            if habia_modelo and anterior.exists():
                anterior.rename(destino)
            raise

        # El nuevo ejemplo siempre corresponde al modelo nuevo.
        EJEMPLO.write_text(
            json.dumps(
                ejemplo.iloc[0].to_dict(),
                indent=2,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )

    print("\n=== SINCRONIZACION COMPLETADA ===")
    print("Nombre:", metadata["model_name"])
    print("Version:", metadata["model_version"])
    print("Alias de origen:", metadata["model_alias"])
    print("Run:", metadata["run_id"])
    print("Features:", len(metadata["feature_names"]))
    print("Archivos verificados:", len(huellas))
    print("Predicciones Registry/local: IDENTICAS")
    print("Modelo guardado en:", destino)
    print("Registry: SIN MODIFICACIONES")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--alias", default="serving")
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get(
            "MLFLOW_TRACKING_URI",
            "http://127.0.0.1:5000",
        ),
    )
    args = parser.parse_args()

    ejecutar(
        nombre=args.model_name,
        alias=args.alias,
        uri=args.tracking_uri,
    )
