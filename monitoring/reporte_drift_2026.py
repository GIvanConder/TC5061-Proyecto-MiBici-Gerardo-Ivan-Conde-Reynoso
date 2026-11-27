"""Drift externo P1 con Evidently 0.7.23."""

import json

import pandas as pd

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

from monitoring.config_2026 import BASE_LOCAL, SALIDA


def guardar_json(ruta, datos):

    ruta.write_text(
        json.dumps(
            datos,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )


def main():

    info = json.loads(
        (SALIDA / "evaluacion_externa_2026.json").read_text()
    )

    manifest = json.loads(
        (SALIDA / "manifest_2026.json").read_text()
    )

    original = json.loads(
        (
            SALIDA.parent
            / "monitoreo_p1"
            / "particiones.json"
        ).read_text()
    )

    assert (
        info["data_version_historica"]
        == original["data_version"]
    )

    assert (
        info["data_version_evaluacion_2026"]
        == manifest["data_version_evaluacion_2026"]
    )

    variables = original["variables"]

    categoricas = [
        "dia_semana",
        "fin_semana",
        "feriado",
    ]

    numericas = [
        c for c in variables
        if c not in categoricas
    ]

    referencia = pd.read_parquet(
        BASE_LOCAL / "referencia_train.parquet"
    )

    actual = pd.read_parquet(
        BASE_LOCAL / "actual_2026.parquet"
    )

    assert len(referencia) == 90230

    assert len(actual) == (
        info["metricas_globales"]["observaciones"]
    )

    for datos in (referencia, actual):

        assert not datos[variables].isna().any().any()

        for columna in categoricas:

            assert (datos[columna] % 1 == 0).all()

            datos[columna] = (
                datos[columna]
                .astype("int64")
                .astype(str)
            )

    definicion = DataDefinition(
        numerical_columns=numericas,
        categorical_columns=categoricas,
    )

    datos_ref = Dataset.from_pandas(
        referencia[variables].copy(),
        data_definition=definicion,
    )

    datos_actuales = Dataset.from_pandas(
        actual[variables].copy(),
        data_definition=definicion,
    )

    reporte = Report(
        [
            DataDriftPreset(),
            DataSummaryPreset(),
        ],
        include_tests=True,
    )

    print("Referencia:", len(referencia))
    print("Periodo externo:", len(actual))
    print("Ejecutando Evidently...", flush=True)

    resultado = reporte.run(
        datos_actuales,
        datos_ref,
    )

    html = SALIDA / "evidently_p1_2026.html"
    ruta_json = SALIDA / "evidently_p1_2026.json"

    resultado.save_html(str(html))
    resultado.save_json(str(ruta_json))

    datos = json.loads(ruta_json.read_text())

    filas = []
    globales = []

    for metrica in datos.get("metrics", []):

        nombre = str(
            metrica.get("metric_name", "")
        )

        config = metrica.get("config") or {}

        if "DriftedColumnsCount" in nombre:

            globales.append({
                "config": config,
                "valor": metrica.get("value"),
            })

        if "ValueDrift" not in nombre:
            continue

        metodo = config["method"]
        umbral = float(config["threshold"])
        valor = float(metrica["value"])

        assert (
            "Wasserstein" in metodo
            or "Jensen-Shannon" in metodo
        ), f"Método inesperado: {metodo}"

        filas.append({
            "variable": config["column"],
            "metodo": metodo,
            "umbral": umbral,
            "valor_drift": valor,
            "drift_detectado": valor >= umbral,
        })

    tabla = pd.DataFrame(filas)

    assert len(tabla) == 12
    assert set(tabla["variable"]) == set(variables)

    cantidad_drift = int(
        tabla["drift_detectado"].sum()
    )

    assert len(globales) == 1

    assert int(
        globales[0]["valor"]["count"]
    ) == cantidad_drift

    tabla.to_csv(
        SALIDA / "resumen_drift_variables_2026.csv",
        index=False,
    )

    estados = {}
    fallidas_por_tipo = {}

    for prueba in datos.get("tests", []):

        estado = str(
            prueba.get("status", "sin_estado")
        ).upper()

        estados[estado] = estados.get(estado, 0) + 1

        if estado == "FAIL":

            tipo = str(
                prueba.get("name")
                or prueba.get("test_name")
                or "no_identificado"
            )

            fallidas_por_tipo[tipo] = (
                fallidas_por_tipo.get(tipo, 0) + 1
            )

    resumen = {
        "referencia": "2025-01/2025-09",
        "actual": "2026-01/2026-10",
        "data_version_historica": (
            manifest["data_version_historica"]
        ),
        "data_version_evaluacion_2026": (
            manifest["data_version_evaluacion_2026"]
        ),
        "filas_referencia": len(referencia),
        "filas_actual": len(actual),
        "variables_analizadas": len(tabla),
        "variables_con_drift": cantidad_drift,
        "variables_detectadas": tabla.loc[
            tabla["drift_detectado"],
            "variable",
        ].tolist(),
        "metricas_globales": globales,
        "estados_pruebas": estados,
        "pruebas_fallidas_por_nombre": fallidas_por_tipo,
        "nota": (
            "Comparación de periodos con diferente "
            "duración y composición temporal. "
            "Las alertas automáticas no equivalen "
            "al número de variables con drift ni "
            "demuestran concept drift."
        ),
    }

    guardar_json(
        SALIDA / "resumen_evidently_2026.json",
        resumen,
    )

    print("\n========== DRIFT POR VARIABLE ==========")

    print(tabla.to_string(index=False))

    print("\n========== RESUMEN EVIDENTLY ==========")

    print(
        json.dumps(
            resumen,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\nHTML generado:", html)


if __name__ == "__main__":
    main()
