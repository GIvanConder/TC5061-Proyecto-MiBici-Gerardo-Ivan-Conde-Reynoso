"""Reconstruir P3 completo sin reentrenar el KMeans historico.

Preflight:
- Reconstruye el scaler exclusivamente con TRAIN.
- Recupera KMeans v1 desde MLflow.
- Compara el ejemplo historico del Registry.
- Exige equivalencia de transformaciones y predicciones.
- Verifica serializacion y recuperacion MLflow.
- No modifica Registry sin --register.
"""
import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from mibici.config import MANIFEST_PATH
from mibici.features import P3_MODEL_FEATURES
from mibici.modeling.p3_clustering import preparar_datos_p3

NOMBRE = "mibici-equipo32-p3-grupos"
URI = "http://127.0.0.1:5000"
EVIDENCIA = Path(
    "docs/evidencias/api-p3/validacion_pipeline_p3.json"
)
EJEMPLO_ORIGINAL = Path(
    "serving/models/p3/input_example.json"
)


def verificar_entrada(datos):
    """Exigir orden, integridad e intensidad no negativa."""
    if list(datos.columns) != P3_MODEL_FEATURES:
        raise ValueError("Esquema u orden de features incompatible")

    valores = datos.to_numpy(dtype=float)
    if not np.isfinite(valores).all():
        raise ValueError("Features no finitas")

    if (datos[["salidas_dia", "llegadas_dia"]] < 0).any().any():
        raise ValueError("Intensidad negativa no permitida")


def construir_pipeline_p3(raw_train, scaler, kmeans):
    """Componer transformaciones y estimador historico ya ajustados."""
    verificar_entrada(raw_train)

    if not hasattr(scaler, "mean_"):
        raise ValueError("StandardScaler no esta ajustado")

    if not hasattr(kmeans, "cluster_centers_"):
        raise ValueError("KMeans historico no esta ajustado")

    if int(scaler.n_features_in_) != len(P3_MODEL_FEATURES):
        raise ValueError("Dimension incorrecta del scaler")

    if int(kmeans.n_features_in_) != len(P3_MODEL_FEATURES):
        raise ValueError("Dimension incorrecta de KMeans")

    preparador = ColumnTransformer(
        transformers=[
            (
                "intensidad",
                FunctionTransformer(
                    np.log1p,
                    feature_names_out="one-to-one",
                ),
                ["salidas_dia", "llegadas_dia"],
            ),
            (
                "resto",
                "passthrough",
                P3_MODEL_FEATURES[2:],
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Solo ajusta la estructura del transformador sin parametros;
    # no vuelve a ajustar ni el scaler ni los centroides.
    preparador.set_output(transform="pandas")
    preparador.fit(raw_train)

    scaler.set_output(transform="pandas")

    return Pipeline([
        ("transformacion", preparador),
        ("scaler", scaler),
        ("modelo", kmeans),
    ])


def ejecutar(registrar=False, uri=URI):
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    cliente = MlflowClient()

    original = cliente.get_model_version_by_alias(
        NOMBRE, "champion"
    )

    if int(original.version) != 1:
        raise ValueError(
            "El champion original ya no es v1: revisar linaje"
        )

    if original.run_id != "c30b8d94d8f445e3a8abd1699676bfd9":
        raise ValueError("Run historico P3 inesperado")

    version_datos = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )["data_version"]

    if original.tags.get("data_version") != version_datos:
        raise ValueError(
            "La version de datos difiere de la registrada"
        )

    print("=== RECUPERAR MODELO ORIGINAL ===")
    modelo_v1 = mlflow.sklearn.load_model(
        f"models:/{NOMBRE}/{original.version}"
    )

    print("Version:", original.version)
    print("Run:", original.run_id)
    print("Datos:", version_datos)

    print("\n=== RECONSTRUIR DATOS Y SCALER ===")
    datos = preparar_datos_p3()

    if datos["contexto"]["data_version"] != version_datos:
        raise ValueError("Datos preparados incompatibles")

    raw_train = datos["features_train"][
        P3_MODEL_FEATURES
    ].copy()
    raw_test = datos["features_test"][
        P3_MODEL_FEATURES
    ].copy()

    x_train = datos["x_train"][P3_MODEL_FEATURES]
    x_test = datos["x_test"][P3_MODEL_FEATURES]

    verificar_entrada(raw_train)
    verificar_entrada(raw_test)

    pipeline = construir_pipeline_p3(
        raw_train,
        datos["scaler"],
        modelo_v1,
    )

    # Primera comprobacion: representacion numerica original.
    transformado_train = pipeline[:-1].transform(raw_train)
    transformado_test = pipeline[:-1].transform(raw_test)

    np.testing.assert_allclose(
        transformado_train.to_numpy(),
        x_train.to_numpy(),
        rtol=0,
        atol=1e-10,
        err_msg="El Pipeline no reproduce el TRAIN original",
    )

    np.testing.assert_allclose(
        transformado_test.to_numpy(),
        x_test.to_numpy(),
        rtol=0,
        atol=1e-10,
        err_msg="El Pipeline no reproduce TEST con el scaler de TRAIN",
    )

    # Segunda comprobacion: ejemplo registrado en MLflow.
    contenido = json.loads(
        EJEMPLO_ORIGINAL.read_text(encoding="utf-8")
    )
    ejemplo = pd.DataFrame(
        contenido["data"],
        columns=contenido["columns"],
    )

    if list(ejemplo.columns) != P3_MODEL_FEATURES:
        raise ValueError("Ejemplo historico incompatible")

    np.testing.assert_allclose(
        x_train.head(len(ejemplo)).to_numpy(),
        ejemplo.to_numpy(),
        rtol=0,
        atol=1e-8,
        err_msg=(
            "El scaler reconstruido NO coincide con el "
            "ejemplo historico del modelo v1"
        ),
    )

    # Tercera comprobacion: mismos clusters en TRAIN y TEST.
    originales_train = modelo_v1.predict(x_train)
    originales_test = modelo_v1.predict(x_test)

    nuevos_train = pipeline.predict(raw_train)
    nuevos_test = pipeline.predict(raw_test)

    if not np.array_equal(originales_train, nuevos_train):
        raise ValueError("Cambian las etiquetas en TRAIN")

    if not np.array_equal(originales_test, nuevos_test):
        raise ValueError("Cambian las etiquetas en TEST")

    # Cuarta comprobacion: serializar y recuperar el pipeline.
    muestra = raw_train.head(5)
    pred_muestra = pipeline.predict(muestra)
    firma = infer_signature(muestra, pred_muestra)

    with tempfile.TemporaryDirectory() as temporal:
        ruta = Path(temporal) / "pipeline"

        mlflow.sklearn.save_model(
            sk_model=pipeline,
            path=str(ruta),
            signature=firma,
            input_example=muestra,
            serialization_format="skops",
        )

        recuperado = mlflow.sklearn.load_model(str(ruta))
        pred_recuperada = recuperado.predict(muestra)

        if not np.array_equal(
            pred_muestra,
            pred_recuperada,
        ):
            raise ValueError(
                "Fallo la reproducibilidad del modelo serializado"
            )

    scaler = datos["scaler"]
    digest = hashlib.sha256()
    digest.update(version_datos.encode())
    digest.update(original.run_id.encode())
    digest.update(
        np.ascontiguousarray(scaler.mean_).tobytes()
    )
    digest.update(
        np.ascontiguousarray(scaler.scale_).tobytes()
    )
    huella = digest.hexdigest()

    resumen = {
        "objetivo": "correccion_preprocesamiento_p3",
        "modelo": NOMBRE,
        "modelo_original_version": int(original.version),
        "modelo_original_run_id": original.run_id,
        "data_version": version_datos,
        "features": P3_MODEL_FEATURES,
        "filas_train": int(len(raw_train)),
        "filas_test": int(len(raw_test)),
        "scaler_reconstruido_con": "TRAIN exclusivamente",
        "transformaciones": [
            "log1p intensidad",
            "StandardScaler original reconstruido",
            "KMeans v1 sin reentrenamiento",
        ],
        "huella_scaler_sha256": huella,
        "diferencia_maxima_transformacion_train": float(
            np.max(np.abs(
                transformado_train.to_numpy()
                - x_train.to_numpy()
            ))
        ),
        "diferencia_maxima_transformacion_test": float(
            np.max(np.abs(
                transformado_test.to_numpy()
                - x_test.to_numpy()
            ))
        ),
        "coincidencia_etiquetas_train": bool(
            np.array_equal(originales_train, nuevos_train)
        ),
        "coincidencia_etiquetas_test": bool(
            np.array_equal(originales_test, nuevos_test)
        ),
        "serializacion_mlflow_skops": "verificada",
        "modelo_reentrenado": False,
    }

    EVIDENCIA.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCIA.write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n=== PREFLIGHT P3 COMPLETO ===")
    print("Estaciones TRAIN:", len(raw_train))
    print("Estaciones TEST:", len(raw_test))
    print("Huella:", huella)
    print("Ejemplo historico: IDENTICO")
    print("Transformacion TRAIN: IDENTICA")
    print("Transformacion TEST: IDENTICA")
    print("Clusters TRAIN: IDENTICOS")
    print("Clusters TEST: IDENTICOS")
    print("Serializacion MLflow: CORRECTA")

    if not registrar:
        print("\nRegistry SIN MODIFICACIONES")
        print("OK: preflight finalizado")
        return

    # El registro definitivo se hara desde main, tras el PR.
    cambios = subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    ).strip()
    if cambios:
        raise RuntimeError(
            "Git debe estar limpio antes del registro"
        )

    versiones = cliente.search_model_versions(
        f"name='{NOMBRE}'"
    )
    existentes = [
        v for v in versiones
        if v.tags.get("huella_scaler_sha256") == huella
        and v.tags.get("base_run_id") == original.run_id
    ]

    if existentes:
        print(
            "Version equivalente existente:",
            existentes[0].version,
        )
        return

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()

    mlflow.set_experiment("mibici-equipo32")

    with mlflow.start_run(
        run_name="p3_pipeline_completo_serving"
    ) as corrida:
        mlflow.set_tags({
            "question": "p3",
            "role": "serving",
            "base_run_id": original.run_id,
            "data_version": version_datos,
            "git_commit": commit,
            "huella_scaler_sha256": huella,
            "model_retrained": "false",
            "scaler_fit_split": "train",
        })

        mlflow.log_dict(
            resumen,
            "serving/validacion_pipeline_p3.json",
        )

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="p3_pipeline_completo",
            registered_model_name=NOMBRE,
            signature=firma,
            input_example=muestra,
            serialization_format="skops",
        )

        run_id = corrida.info.run_id

    candidatas = [
        v for v in cliente.search_model_versions(
            f"name='{NOMBRE}'"
        )
        if v.run_id == run_id
    ]

    if len(candidatas) != 1:
        raise RuntimeError(
            "No se identifico una unica version registrada"
        )

    nueva = candidatas[0]

    cliente.set_model_version_tag(
        NOMBRE, nueva.version,
        "huella_scaler_sha256", huella,
    )
    cliente.set_model_version_tag(
        NOMBRE, nueva.version,
        "base_run_id", original.run_id,
    )
    cliente.set_model_version_tag(
        NOMBRE, nueva.version,
        "git_commit", commit,
    )
    cliente.set_model_version_tag(
        NOMBRE, nueva.version,
        "input_semantics", "raw_p3_features",
    )

    desde_registry = mlflow.sklearn.load_model(
        f"models:/{NOMBRE}/{nueva.version}"
    )

    if not np.array_equal(
        desde_registry.predict(raw_train),
        originales_train,
    ):
        raise RuntimeError(
            "La version registrada no reproduce TRAIN"
        )

    if not np.array_equal(
        desde_registry.predict(raw_test),
        originales_test,
    ):
        raise RuntimeError(
            "La version registrada no reproduce TEST"
        )

    actual = cliente.get_model_version_by_alias(
        NOMBRE, "champion"
    )
    if actual.version != original.version:
        raise RuntimeError("El champion cambio inesperadamente")

    # Alias independiente para despliegue: no desplaza champion.
    cliente.set_registered_model_alias(
        NOMBRE, "serving", nueva.version
    )

    print("\n=== NUEVA VERSION P3 ===")
    print("Version:", nueva.version)
    print("Run:", run_id)
    print("Commit:", commit)
    print("Alias serving:", nueva.version)
    print("Champion conservado:", original.version)
    print("OK: Pipeline completo registrado y verificado")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--tracking-uri", default=URI)
    args = parser.parse_args()
    ejecutar(args.register, args.tracking_uri)
