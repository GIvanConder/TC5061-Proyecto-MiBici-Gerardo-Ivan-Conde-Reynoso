"""Diagnóstico de alertas y distribución temporal de P1."""
import json
from pathlib import Path

import pandas as pd

BASE = Path("reports/p1/drift_abtest")
reporte = json.loads(
    (BASE / "evidently_p1_drift.json").read_text(
        encoding="utf-8"
    )
)

fallidas = [
    t for t in reporte["tests"]
    if str(t.get("status", "")).upper() == "FAIL"
]

print("\n=== PRUEBAS FALLIDAS ===")
print("Total:", len(fallidas))

for numero, prueba in enumerate(fallidas, 1):
    print(f"\n--- PRUEBA {numero} ---")
    print(
        json.dumps(
            prueba,
            indent=2,
            ensure_ascii=False,
            default=str,
        )[:1100]
    )

print("\n=== DISTRIBUCIONES TRAIN / TEST ===")

columnas = [
    "dias_activa",
    "lag_1",
    "lag_7",
    "media_7d",
    "media_28d",
    "capacidad",
    "viajes_real",
]

for nombre in ("train", "test"):
    datos = pd.read_parquet(BASE / f"{nombre}.parquet")

    print(f"\n--- {nombre.upper()} ---")
    print("Filas:", len(datos))

    for columna in columnas:
        serie = datos[columna]
        print(
            f"{columna:24s} "
            f"media={serie.mean():10.3f} "
            f"mediana={serie.median():10.3f} "
            f"p05={serie.quantile(0.05):10.3f} "
            f"p95={serie.quantile(0.95):10.3f}"
        )

    if "estacion" in datos:
        print(
            "Estaciones distintas:",
            datos["estacion"].nunique()
        )

    for columna in ("fecha", "dia"):
        if columna in datos.columns:
            print(
                f"Periodo ({columna}):",
                datos[columna].min(),
                "a",
                datos[columna].max()
            )

print("\n=== ESTACIONES COMPARTIDAS ===")
train = pd.read_parquet(BASE / "train.parquet")
test = pd.read_parquet(BASE / "test.parquet")

if "estacion" in train and "estacion" in test:
    estaciones_train = set(train["estacion"])
    estaciones_test = set(test["estacion"])
    print("Compartidas:", len(
        estaciones_train & estaciones_test
    ))
    print("Nuevas en TEST:", len(
        estaciones_test - estaciones_train
    ))

print("\nOK: diagnóstico finalizado")
