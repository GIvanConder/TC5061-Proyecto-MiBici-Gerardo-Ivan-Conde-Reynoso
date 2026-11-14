"""Exporta reporte Evidently y resumen verificable de P1."""
import json
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd

ORIGEN = Path("reports/p1/drift_abtest")
DESTINO = Path("docs/evidencias/monitoreo_p1")
DESTINO.mkdir(parents=True, exist_ok=True)

reporte = json.loads(
    (ORIGEN / "evidently_p1_drift.json").read_text(
        encoding="utf-8"
    )
)
info = json.loads(
    (ORIGEN / "particiones.json").read_text(
        encoding="utf-8"
    )
)

filas = []
for metrica in reporte["metrics"]:
    if "ValueDrift" not in metrica.get("metric_name", ""):
        continue

    config = metrica["config"]
    valor = float(metrica["value"])
    umbral = float(config["threshold"])

    filas.append({
        "variable": config["column"],
        "metodo": config["method"],
        "valor": valor,
        "umbral": umbral,
        "drift": valor > umbral,
    })

tabla = pd.DataFrame(filas)
assert len(tabla) == len(info["variables"])
assert set(tabla["variable"]) == set(info["variables"])
assert tabla["variable"].is_unique

globales = [
    m for m in reporte["metrics"]
    if "DriftedColumnsCount" in m.get("metric_name", "")
]
assert len(globales) == 1
assert int(globales[0]["value"]["count"]) == int(tabla["drift"].sum())

fallidas = [
    t for t in reporte["tests"]
    if t.get("status") == "FAIL"
]

tipos = Counter(
    t.get("metric_config", {})
     .get("params", {})
     .get("type", "")
     .split(":")[-1]
    for t in fallidas
)

train = pd.read_parquet(ORIGEN / "train.parquet")
test = pd.read_parquet(ORIGEN / "test.parquet")

for datos in (train, test):
    assert not datos[info["variables"]].isna().any().any()

tabla = tabla.sort_values(
    ["drift", "valor"], ascending=[False, False]
)

lineas = [
    "# Evidencia de monitoreo P1 con Evidently",
    "",
    f"- Versión de datos: `{info['data_version']}`",
    f"- Referencia: {info['referencia']} ({len(train):,} filas)",
    f"- Periodo actual: {info['actual_evaluacion']} ({len(test):,} filas)",
    f"- Variables analizadas: {len(tabla)}",
    f"- Variables con drift: {int(tabla['drift'].sum())}",
    f"- Pruebas automáticas fallidas: {len(fallidas)}",
    "",
    "## Resultados por variable",
    "",
    "| Variable | Método | Estadístico | Umbral | Drift |",
    "|---|---|---:|---:|---|",
]

for fila in tabla.itertuples():
    lineas.append(
        f"| `{fila.variable}` | {fila.metodo} | "
        f"{fila.valor:.4f} | {fila.umbral:.2f} | "
        f"{'Sí' if fila.drift else 'No'} |"
    )

lineas += [
    "",
    "## Calidad y comprobaciones",
    "",
    f"- TRAIN: {len(train):,} filas, "
    f"{train['estacion'].nunique()} estaciones.",
    f"- TEST: {len(test):,} filas, "
    f"{test['estacion'].nunique()} estaciones.",
    "- Sin nulos en las 12 características analizadas.",
    "- Las particiones se obtuvieron de la preparación "
    "original del modelo P1.",
    "",
    "### Tipos de pruebas fallidas",
    "",
]

for tipo, cantidad in sorted(tipos.items()):
    lineas.append(f"- `{tipo}`: {cantidad}")

lineas += [
    "",
    "## Interpretación",
    "",
    "`dias_activa` es la única variable que supera el "
    "umbral de drift de Evidently. Esta característica "
    "cuenta los días transcurridos en el panel de cada "
    "estación, por lo que su cambio entre periodos "
    "es coherente con el avance temporal.",
    "",
    "Las diferencias de tamaño entre conjuntos y de "
    "valores extremos no equivalen automáticamente "
    "a fallas de calidad ni a degradación predictiva.",
    "",
    "El criterio global de Evidently usa una proporción "
    "de variables con drift de 0.5; debe distinguirse "
    "de las alertas individuales.",
    "",
    "Estos resultados describen data drift, no prueban "
    "concept drift. La recomendación de reentrenamiento "
    "se determinará después de comparar el desempeño "
    "de las versiones sobre el mismo periodo de evaluación.",
    "",
    "## Reproducción",
    "",
    "Desde la raíz del repositorio, con datos procesados "
    "disponibles:",
    "",
    "```bash",
    "./.venv/bin/python -m monitoring.preparar_p1",
    "$HOME/.venvs/mibici-evidently/bin/python "
    "-m monitoring.reporte_drift_p1",
    "./.venv/bin/python -m monitoring.resumir_drift_p1",
    "./.venv/bin/python -m monitoring.exportar_evidencias_p1",
    "```",
    "",
    "El entorno Evidently se prepara con las dependencias "
    "fijadas en `monitoring/requirements.txt`.",
]

(DESTINO / "README.md").write_text(
    "\n".join(lineas) + "\n",
    encoding="utf-8",
)

for nombre in (
    "evidently_p1_drift.html",
    "evidently_p1_drift.json",
    "resumen_drift_variables.csv",
    "particiones.json",
):
    origen = ORIGEN / nombre
    destino = DESTINO / nombre

    if nombre.endswith(".html"):
        contenido = origen.read_text(encoding="utf-8")

        # Normalizar solo líneas vacías generadas por Evidently.
        # El reporte original permanece intacto.
        contenido = "\n".join(
            "" if not linea.strip() else linea
            for linea in contenido.split("\n")
        )

        destino.write_text(contenido, encoding="utf-8")
    else:
        shutil.copy2(origen, destino)

print("=== EVIDENCIAS EXPORTADAS ===")
print("Variables con drift:", int(tabla["drift"].sum()))
print("Pruebas fallidas:", len(fallidas))
print("Tipos:", dict(tipos))
for archivo in sorted(DESTINO.iterdir()):
    print(archivo.name, archivo.stat().st_size, "bytes")
print("OK: evidencias preparadas")
