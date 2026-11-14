"""Extrae resultados verificables del reporte Evidently P1."""
import json
from pathlib import Path

import pandas as pd

BASE = Path("reports/p1/drift_abtest")
REPORTE = json.loads(
    (BASE / "evidently_p1_drift.json").read_text(
        encoding="utf-8"
    )
)

metricas = REPORTE.get("metrics", [])
pruebas = REPORTE.get("tests", [])

print("\n=== ESTRUCTURA EVIDENTLY ===")
print("Metricas:", len(metricas))
print("Pruebas:", len(pruebas))

print("\n=== DRIFT GLOBAL ===")
globales = [
    m for m in metricas
    if "DriftedColumnsCount" in str(
        m.get("metric_name", "")
    )
]
for m in globales:
    print("Metrica:", m.get("metric_name"))
    print("Configuracion:", m.get("config"))
    print("Resultado:", m.get("value"))

print("\n=== DRIFT POR VARIABLE ===")
filas = []

for m in metricas:
    nombre = str(m.get("metric_name", ""))

    if "ValueDrift" not in nombre:
        continue

    config = m.get("config") or {}
    valor = m.get("value")

    filas.append({
        "variable": config.get("column"),
        "metodo": config.get("method"),
        "umbral": config.get("threshold"),
        "valor_drift": valor,
    })

tabla = pd.DataFrame(filas)

if len(tabla) != 12:
    print("No se identificaron las 12 metricas.")
    print("Primeras estructuras disponibles:")
    for m in metricas[:5]:
        print(json.dumps(m, ensure_ascii=False)[:700])
    raise RuntimeError(
        f"Metricas ValueDrift encontradas: {len(tabla)}"
    )

print(tabla.to_string(index=False))

ruta = BASE / "resumen_drift_variables.csv"
tabla.to_csv(ruta, index=False)
print("\nTabla guardada:", ruta)

print("\n=== PRUEBAS DE EVIDENTLY ===")
estados = {}

for prueba in pruebas:
    estado = str(prueba.get("status", "sin_estado"))
    estados[estado] = estados.get(estado, 0) + 1

print("Estados:", estados)

print("\n=== CALIDAD DE DATOS ===")
info = json.loads(
    (BASE / "particiones.json").read_text(
        encoding="utf-8"
    )
)
variables = info["variables"]

for nombre in ("train", "val", "test"):
    datos = pd.read_parquet(BASE / f"{nombre}.parquet")
    x = datos[variables]

    print(
        f"{nombre.upper()}: "
        f"filas={len(datos):,}, "
        f"nulos={int(x.isna().sum().sum())}, "
        f"columnas={len(variables)}"
    )

    if "estacion" in datos.columns:
        print(
            "  estaciones:",
            datos["estacion"].nunique()
        )

print("\nOK: extraccion finalizada")
