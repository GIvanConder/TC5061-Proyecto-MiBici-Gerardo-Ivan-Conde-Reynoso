"""Entrenamiento y registro reproducible del challenger P1.

Estrategia:
- Recuperar el champion desde MLflow.
- Clonar su Pipeline completo, incluido el preprocesamiento.
- Reentrenar con enero-octubre de 2025.
- No acceder a TEST durante el entrenamiento o registro.
- Registrar una nueva versión solo cuando se solicite.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error

from mibici.features import P1_FEATURES

BASE = Path("reports/p1/drift_abtest")
NOMBRE_MODELO = "mibici-equipo32-p1-demanda"


def validar_particion(df, nombre, inicio, fin):
    """Valida esquema, calidad, unicidad e intervalo [inicio, fin)."""
    requeridas = {"estacion", "fecha", "viajes_real", *P1_FEATURES}

    if not requeridas.issubset(df.columns):
        raise ValueError(
            f"{nombre}: faltan columnas {requeridas - set(df.columns)}"
        )

    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="raise")

    if df.empty:
        raise ValueError(f"{nombre}: conjunto vacío")

    if not df["fecha"].between(
        pd.Timestamp(inicio),
        pd.Timestamp(fin),
        inclusive="left",
    ).all():
        raise ValueError(f"{nombre}: fechas fuera del periodo autorizado")

    if df.duplicated(["estacion", "fecha"]).any():
        raise ValueError(f"{nombre}: estación-fecha duplicada")

    if df[list(requeridas)].isna().any().any():
        raise ValueError(f"{nombre}: valores faltantes")

    x = df[P1_FEATURES].astype("float64")
    y = df["viajes_real"].astype("float64")

    if not np.isfinite(x.to_numpy()).all():
        raise ValueError(f"{nombre}: features no finitas")

    if not np.isfinite(y.to_numpy()).all() or (y < 0).any():
        raise ValueError(f"{nombre}: objetivo inválido")

    return df.sort_values(["fecha", "estacion"]).reset_index(drop=True)


def preparar_entrenamiento():
    """Lee exclusivamente TRAIN y VALIDATION, nunca TEST."""
    info = json.loads(
        (BASE / "particiones.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        Path("data/manifest.json").read_text(encoding="utf-8")
    )

    if info["data_version"] != manifest["data_version"]:
        raise ValueError("No coincide la versión de datos")

    if info["variables"] != P1_FEATURES:
        raise ValueError("No coincide el esquema de features P1")

    train = validar_particion(
        pd.read_parquet(BASE / "train.parquet"),
        "TRAIN",
        "2025-01-01",
        "2025-10-01",
    )
    val = validar_particion(
        pd.read_parquet(BASE / "val.parquet"),
        "VALIDATION",
        "2025-10-01",
        "2025-11-01",
    )

    combinado = pd.concat([train, val], ignore_index=True)
    if combinado.duplicated(["estacion", "fecha"]).any():
        raise ValueError("Hay observaciones compartidas entre particiones")

    x = combinado[P1_FEATURES].astype("float64")
    y = combinado["viajes_real"].astype("float64")

    digest = hashlib.sha256()
    digest.update(info["data_version"].encode())
    digest.update(
        pd.util.hash_pandas_object(
            combinado[
                ["estacion", "fecha", *P1_FEATURES, "viajes_real"]
            ],
            index=False,
        ).to_numpy().tobytes()
    )

    return x, y, info, digest.hexdigest(), len(train), len(val)


def ejecutar(registrar=False, tracking_uri="http://127.0.0.1:5000"):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = MlflowClient()

    champion = client.get_model_version_by_alias(
        NOMBRE_MODELO, "champion"
    )
    modelo_original = mlflow.sklearn.load_model(
        f"models:/{NOMBRE_MODELO}/{champion.version}"
    )

    if not hasattr(modelo_original, "named_steps"):
        raise TypeError("El champion no contiene el Pipeline esperado")

    original_features = list(modelo_original.feature_names_in_)
    if original_features != P1_FEATURES:
        raise ValueError("El esquema del champion no coincide con P1")

    nuevo = clone(modelo_original)
    parametros = nuevo.named_steps["modelo"].get_params()

    x, y, info, huella, filas_train, filas_val = preparar_entrenamiento()

    versiones = client.search_model_versions(
        f"name='{NOMBRE_MODELO}'"
    )
    repetidas = [
        v for v in versiones
        if v.tags.get("training_fingerprint") == huella
        and v.tags.get("base_run_id") == champion.run_id
    ]

    print("\n=== PREFLIGHT CHALLENGER ===")
    print("Champion:", champion.version, champion.run_id)
    print("Filas TRAIN:", filas_train)
    print("Filas VALIDATION:", filas_val)
    print("Filas combinadas:", len(x))
    print("Features:", len(P1_FEATURES))
    print("Alpha:", parametros["alpha"])
    print("Max iter:", parametros["max_iter"])
    print("Data version:", info["data_version"])
    print("Huella SHA256:", huella)
    print("Versiones equivalentes:", [v.version for v in repetidas])
    print("TEST: no leído ni utilizado")

    if not registrar:
        print("OK: preflight superado. Registry sin modificaciones.")
        return

    if repetidas:
        print("YA EXISTE: no se registrará una versión duplicada.")
        return

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()

    cambios = subprocess.check_output(
        ["git", "status", "--porcelain"], text=True
    ).strip()

    if cambios:
        raise RuntimeError(
            "El repositorio tiene cambios pendientes. "
            "Registrar únicamente desde un commit limpio."
        )

    nuevo.fit(x, y)

    pred_train = nuevo.predict(x)
    mae_train = float(mean_absolute_error(y, pred_train))
    wape_train = float(np.abs(y - pred_train).sum() / y.sum())

    ejemplo = x.head(5)
    firma = infer_signature(ejemplo, nuevo.predict(ejemplo))

    metadata = {
        "model_name": NOMBRE_MODELO,
        "base_model_version": str(champion.version),
        "base_run_id": champion.run_id,
        "data_version": info["data_version"],
        "train_period": "2025-01-01/2025-10-31",
        "excluded_test_period": "2025-11-01/2025-12-31",
        "train_rows": len(x),
        "original_train_rows": filas_train,
        "october_rows": filas_val,
        "feature_names": P1_FEATURES,
        "training_fingerprint": huella,
        "git_commit": commit,
        "hyperparameters": {
            "alpha": parametros["alpha"],
            "max_iter": parametros["max_iter"],
        },
        "selection_rule": (
            "Mismo pipeline e hiperparámetros del champion; "
            "único cambio: incorporar octubre al entrenamiento."
        ),
    }

    mlflow.set_experiment("mibici-equipo32")

    with mlflow.start_run(
        run_name="p1_challenger_reentrenamiento_oct2025"
    ) as run:
        mlflow.set_tags({
            "question": "p1",
            "role": "challenger",
            "data_version": info["data_version"],
            "git_commit": commit,
            "base_run_id": champion.run_id,
            "training_fingerprint": huella,
            "test_used_for_training": "false",
        })

        mlflow.log_params({
            "model": "poisson",
            "alpha": parametros["alpha"],
            "max_iter": parametros["max_iter"],
            "train_rows": len(x),
            "train_end_exclusive": "2025-11-01",
        })

        mlflow.log_metrics({
            "train_mae": mae_train,
            "train_wape": wape_train,
        })

        mlflow.log_dict(
            metadata, "monitoring/challenger_manifest.json"
        )

        mlflow.sklearn.log_model(
            sk_model=nuevo,
            name="p1_poisson_octubre2025",
            registered_model_name=NOMBRE_MODELO,
            signature=firma,
            input_example=ejemplo,
        )

        run_id = run.info.run_id

    creadas = [
        v for v in client.search_model_versions(
            f"name='{NOMBRE_MODELO}'"
        )
        if v.run_id == run_id
    ]

    if len(creadas) != 1:
        raise RuntimeError(
            f"Versiones asociadas al run: {len(creadas)}"
        )

    version = creadas[0]

    client.set_model_version_tag(
        NOMBRE_MODELO, version.version,
        "training_fingerprint", huella
    )
    client.set_model_version_tag(
        NOMBRE_MODELO, version.version,
        "base_run_id", champion.run_id
    )
    client.set_model_version_tag(
        NOMBRE_MODELO, version.version,
        "git_commit", commit
    )

    recuperado = mlflow.sklearn.load_model(
        f"models:/{NOMBRE_MODELO}/{version.version}"
    )

    pred_recuperada = recuperado.predict(ejemplo)

    if not np.allclose(
        pred_recuperada,
        nuevo.predict(ejemplo),
        rtol=0,
        atol=1e-10,
    ):
        raise RuntimeError("El modelo registrado no reproduce predicciones")

    champion_actual = client.get_model_version_by_alias(
        NOMBRE_MODELO, "champion"
    )
    if champion_actual.version != champion.version:
        raise RuntimeError("El alias champion cambió inesperadamente")

    client.set_registered_model_alias(
        NOMBRE_MODELO, "challenger", version.version
    )

    resultado = {
        **metadata,
        "run_id": run_id,
        "registered_version": version.version,
        "train_mae": mae_train,
        "train_wape": wape_train,
    }
    (BASE / "challenger_registry.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== CHALLENGER REGISTRADO ===")
    print("Versión:", version.version)
    print("Run ID:", run_id)
    print("MAE TRAIN+OCT:", round(mae_train, 6))
    print("WAPE TRAIN+OCT:", round(wape_train, 6))
    print("Champion conservado:", champion.version)
    print("Challenger asignado:", version.version)
    print("OK: recuperación y predicciones verificadas")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--register", action="store_true",
        help="Entrenar, registrar y asignar challenger",
    )
    parser.add_argument(
        "--tracking-uri",
        default="http://127.0.0.1:5000",
    )
    args = parser.parse_args()
    ejecutar(
        registrar=args.register,
        tracking_uri=args.tracking_uri,
    )
