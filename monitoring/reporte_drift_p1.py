"""Genera drift y calidad de datos P1 con Evidently."""
import json
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

BASE = Path("reports/p1/drift_abtest")
BASE.mkdir(parents=True, exist_ok=True)

info = json.loads(
    (BASE / "particiones.json").read_text(encoding="utf-8")
)

variables = info["variables"]

categoricas = [
    "dia_semana",
    "fin_semana",
    "feriado",
]
numericas = [c for c in variables if c not in categoricas]

referencia = pd.read_parquet(BASE / "train.parquet")
actual = pd.read_parquet(BASE / "test.parquet")

# Evidently 0.7.23 espera etiquetas categóricas compatibles
# con cadenas; las features originales están en float64.
# La conversión se aplica solo al monitoreo, nunca al modelo.
for nombre, datos in (
    ("TRAIN", referencia),
    ("TEST", actual),
):
    for columna in categoricas:
        serie = datos[columna]

        assert serie.notna().all(), (
            f"{nombre}: nulos en {columna}"
        )
        assert (serie % 1 == 0).all(), (
            f"{nombre}: valores no enteros en {columna}"
        )

        print(
            f"{nombre} | {columna}:",
            sorted(serie.unique().tolist()),
        )

        datos[columna] = serie.astype("int64").astype(str)


assert all(c in referencia for c in variables)
assert all(c in actual for c in variables)
assert not referencia[variables].isna().any().any()
assert not actual[variables].isna().any().any()

print("Referencia:", len(referencia), "filas")
print("Actual:", len(actual), "filas")
print("Variables:", len(variables))

definicion = DataDefinition(
    numerical_columns=numericas,
    categorical_columns=categoricas,
)

datos_referencia = Dataset.from_pandas(
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

resultado = reporte.run(
    datos_actuales,
    datos_referencia,
)

html = BASE / "evidently_p1_drift.html"
json_path = BASE / "evidently_p1_drift.json"

resultado.save_html(str(html))
resultado.save_json(str(json_path))

print("\n=== REPORTES GENERADOS ===")
for ruta in (html, json_path):
    print(ruta, "|", ruta.stat().st_size, "bytes")

resumen = resultado.dict()
print("\nClaves del reporte:", list(resumen.keys()))
print("OK: Evidently ejecutado correctamente")
