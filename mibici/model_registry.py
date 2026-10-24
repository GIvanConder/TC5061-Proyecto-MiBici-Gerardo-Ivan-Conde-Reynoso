"""Registro reproducible de los modelos seleccionados de MiBici."""

from __future__ import annotations

import os

import mlflow
from mlflow import MlflowClient


TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

ALIAS_PRODUCCION = "champion"

MODELOS = {
    "p1": {
        "registered_name": "mibici-equipo32-p1-demanda",
        "model_uri": "models:/m-f1e2eda1d16a4663a7ec1f50d75859d9",
        "run_id": "371bf2379bfa4a74b8effbe02d2ee3ae",
        "description": (
            "Modelo de producción para P1: predicción de viajes iniciados "
            "por estación MiBici para el día siguiente."
        ),
        "version_description": (
            "PoissonRegressor seleccionado por su desempeño frente al "
            "baseline y candidatos, manteniendo una solución simple y "
            "adecuada para una variable objetivo de conteo."
        ),
    },
    "p2": {
        "registered_name": "mibici-equipo32-p2-riesgo",
        "model_uri": "models:/m-8bddd02b531d4d61936d74a3f9cadc93",
        "run_id": "9e6d7264d7a74a68a9f8a02f4be3b03f",
        "description": (
            "Modelo de producción para P2: clasificación del riesgo de que "
            "una estación MiBici se quede sin bicicletas o sin anclajes "
            "en las próximas horas."
        ),
        "version_description": (
            "HistGradientBoostingClassifier tuneado seleccionado mediante "
            "VALIDATION. TEST se utilizó únicamente para la evaluación final."
        ),
    },
    "p3": {
        "registered_name": "mibici-equipo32-p3-grupos",
        "model_uri": "models:/m-7918112016fa487f87ac4e87eb31d359",
        "run_id": "c30b8d94d8f445e3a8abd1699676bfd9",
        "description": (
            "Modelo de producción para P3: segmentación de estaciones MiBici "
            "según patrones de uso y estabilidad temporal."
        ),
        "version_description": (
            "K-Means con k=4 y n_init=20. Seleccionado por el balance entre "
            "calidad interna, estabilidad temporal e histórica y utilidad "
            "interpretativa de los grupos."
        ),
    },
}


def buscar_version_existente(
    client: MlflowClient,
    registered_name: str,
    run_id: str,
):
    """Evita crear una versión duplicada para el mismo run ganador."""
    versiones = client.search_model_versions(
        f"name='{registered_name}'"
    )

    for version in versiones:
        if version.run_id == run_id:
            return version

    return None


def registrar_modelo(
    client: MlflowClient,
    question: str,
    spec: dict,
):
    """Registra una versión, añade linaje y asigna alias champion."""

    registered_name = spec["registered_name"]
    run_id = spec["run_id"]

    run = client.get_run(run_id)

    git_commit = run.data.tags.get(
        "git_commit",
        run.data.tags.get("mlflow.source.git.commit", "desconocido"),
    )

    data_version = run.data.tags.get(
        "data_version",
        run.data.params.get("data_version", "desconocida"),
    )

    existente = buscar_version_existente(
        client,
        registered_name,
        run_id,
    )

    if existente is None:
        print(
            f"\nRegistrando {question.upper()} "
            f"como {registered_name}..."
        )

        version = mlflow.register_model(
            model_uri=spec["model_uri"],
            name=registered_name,
        )
    else:
        version = existente
        print(
            f"\n{question.upper()}: ya existe la versión "
            f"{version.version} para el run seleccionado."
        )

    client.update_registered_model(
        name=registered_name,
        description=spec["description"],
    )

    client.update_model_version(
        name=registered_name,
        version=version.version,
        description=spec["version_description"],
    )

    tags = {
        "team": "equipo32",
        "question": question,
        "source_run_id": run_id,
        "git_commit": git_commit,
        "data_version": data_version,
        "lifecycle_scheme": "aliases",
        "production_alias": ALIAS_PRODUCCION,
        "selection_status": "selected",
    }

    for key, value in tags.items():
        client.set_model_version_tag(
            name=registered_name,
            version=version.version,
            key=key,
            value=str(value),
        )

    client.set_registered_model_tag(
        registered_name,
        "team",
        "equipo32",
    )

    client.set_registered_model_tag(
        registered_name,
        "question",
        question,
    )

    client.set_registered_model_tag(
        registered_name,
        "lifecycle_scheme",
        "aliases",
    )

    client.set_registered_model_alias(
        name=registered_name,
        alias=ALIAS_PRODUCCION,
        version=version.version,
    )

    print(f"Modelo:   {registered_name}")
    print(f"Versión:  {version.version}")
    print(f"Alias:    @{ALIAS_PRODUCCION}")
    print(f"Run ID:   {run_id}")
    print(f"Commit:   {git_commit}")
    print(f"Datos:    {data_version}")

    return version.version


def verificar_carga(registered_name: str):
    """Comprueba que el modelo productivo puede cargarse por alias."""
    uri = f"models:/{registered_name}@{ALIAS_PRODUCCION}"

    modelo = mlflow.pyfunc.load_model(uri)

    print(f"✅ Carga correcta: {uri}")
    print(f"   Tipo pyfunc: {type(modelo).__name__}")


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)

    client = MlflowClient()

    print("=" * 80)
    print("REGISTRO DE MODELOS MIBICI — EQUIPO 32")
    print(f"Tracking/Registry URI: {TRACKING_URI}")
    print("=" * 80)

    for question, spec in MODELOS.items():
        registrar_modelo(
            client,
            question,
            spec,
        )

    print("\n" + "=" * 80)
    print("VERIFICACIÓN DE MODELOS @champion")
    print("=" * 80)

    for spec in MODELOS.values():
        verificar_carga(spec["registered_name"])

    print("\n✅ Model Registry completado correctamente.")


if __name__ == "__main__":
    main()
