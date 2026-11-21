"""Resume resultados reales de Locust y recursos de Podman."""

import json
import os
from consolidar_locust import metricas_finales
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/evidencias/operacion_finops/locust"

run = os.getenv("MIBICI_RUN_ID") or (BASE / "ultima_ejecucion.txt").read_text().strip()
ruta = BASE / run

escenarios = ["baja", "media", "alta"]
resultados = []
recursos = []

for escenario in [
    "reposo", "calentamiento", *escenarios
]:
    destino = ruta / escenario

    metadata = json.loads(
        (destino / "escenario.json").read_text()
    )

    consumo = pd.read_csv(destino / "recursos.csv")

    for contenedor, df in consumo.groupby("contenedor"):
        cpu = df["cpu_intervalo_pct"].dropna()
        memoria = df["memoria_bytes"] / 1e6

        assert len(cpu) >= 2
        assert (cpu >= 0).all()
        assert (memoria > 0).all()

        df = df.sort_values("segundos")

        intervalo = (
            df["segundos"].iloc[-1]
            - df["segundos"].iloc[0]
        )

        cpu_ns = (
            df["cpu_acumulada_ns"].iloc[-1]
            - df["cpu_acumulada_ns"].iloc[0]
        )

        assert intervalo > 0
        assert cpu_ns >= 0

        recursos.append({
            "escenario": escenario,
            "usuarios": metadata["usuarios"],
            "servicio": (
                "API" if "_api_" in contenedor else "MLflow"
            ),
            "muestras": len(df),
            "cpu_media_pct": (
                100 * cpu_ns / (intervalo * 1e9)
            ),
            "cpu_max_pct": cpu.max(),
            "memoria_media_mb": memoria.mean(),
            "memoria_max_mb": memoria.max(),
            "intervalo_monitorizado_s": intervalo,
        })

    if escenario not in escenarios:
        continue

    stats = pd.read_csv(destino / "locust_stats.csv")

    filas = stats[
        (stats["Type"] == "POST")
        & (stats["Name"] == "/predict")
    ]

    assert len(filas) == 1, (
        f"No se identifico /predict en {escenario}"
    )


    fila = filas.iloc[0]
    final = metricas_finales(destino)

    total = final["solicitudes"]
    fallos = final["fallos"]
    rps = final["rps"]

    resultados.append({
        "escenario": escenario,
        "usuarios": metadata["usuarios"],
        "solicitudes": total,
        "exitos": total - fallos,
        "fallos": fallos,
        "tasa_error_pct": 100 * fallos / total,
        "rps": rps,
        "rps_exitosas": rps * (total - fallos) / total,
        "latencia_media_ms": final["latencia_media_ms"],
        "p50_ms": final["p50_ms"],
        "p95_ms": final["p95_ms"],
        "p99_ms": final["p99_ms"],
        "solicitudes_csv_periodico": int(fila["Request Count"]),
        "fuente_metricas": final["fuente"],
        "duracion_programada_s": metadata["duracion_programada_s"],
        "duracion_observada_s": metadata["duracion_observada_s"],
    })

rendimiento = pd.DataFrame(resultados)
consumo = pd.DataFrame(recursos)

rendimiento.to_csv(
    ruta / "resumen_rendimiento.csv",
    index=False,
    float_format="%.6f",
)

consumo.to_csv(
    ruta / "resumen_recursos.csv",
    index=False,
    float_format="%.6f",
)

informe = {
    "ejecucion": run,
    "rendimiento": rendimiento.to_dict("records"),
    "recursos": consumo.to_dict("records"),
}

(ruta / "resumen_experimental.json").write_text(
    json.dumps(informe, indent=2) + "\n"
)

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def guardar(figura, nombre):
    figura.tight_layout()
    figura.savefig(
        ruta / nombre,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figura)

fig, ax = plt.subplots(figsize=(8, 4.5))

for campo, etiqueta in [
    ("p50_ms", "p50"),
    ("p95_ms", "p95"),
    ("p99_ms", "p99"),
]:
    ax.plot(
        rendimiento["usuarios"],
        rendimiento[campo],
        marker="o",
        label=etiqueta,
    )

ax.set_title("Latencia de inferencia P3")
ax.set_xlabel("Usuarios concurrentes")
ax.set_ylabel("Latencia (ms)")
ax.set_xticks(rendimiento["usuarios"])
ax.grid(alpha=0.25)
ax.legend()

guardar(fig, "01_latencia.png")

fig, ax = plt.subplots(figsize=(8, 4.5))

ax.bar(
    rendimiento["usuarios"].astype(str),
    rendimiento["rps_exitosas"],
)

ax.set_title("Rendimiento de inferencias exitosas")
ax.set_xlabel("Usuarios concurrentes")
ax.set_ylabel("Inferencias por segundo")
ax.grid(axis="y", alpha=0.25)

guardar(fig, "02_rendimiento.png")

fig, ax = plt.subplots(figsize=(8, 4.5))

ax.bar(
    rendimiento["usuarios"].astype(str),
    rendimiento["tasa_error_pct"],
)

ax.set_title("Tasa de fallos por nivel de carga")
ax.set_xlabel("Usuarios concurrentes")
ax.set_ylabel("Errores (%)")
ax.set_ylim(
    0,
    max(1, rendimiento["tasa_error_pct"].max() * 1.2),
)
ax.grid(axis="y", alpha=0.25)

guardar(fig, "03_errores.png")

orden = ["reposo", "baja", "media", "alta"]

filtrado = consumo[consumo["escenario"].isin(orden)]

for columna, titulo, ylabel, archivo in [
    (
        "cpu_media_pct",
        "Consumo medio de CPU",
        "CPU (%) - 100% equivale a un nucleo",
        "04_cpu.png",
    ),
    (
        "memoria_media_mb",
        "Consumo medio de memoria",
        "Memoria (MB)",
        "05_memoria.png",
    ),
]:
    tabla = filtrado.pivot(
        index="escenario",
        columns="servicio",
        values=columna,
    ).reindex(orden)

    ax = tabla.plot.bar(
        figsize=(8, 4.5),
        rot=0,
    )

    ax.set_title(titulo)
    ax.set_xlabel("Escenario")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)

    guardar(ax.get_figure(), archivo)

print("\n========== RENDIMIENTO ==========")
print(rendimiento.round(4).to_string(index=False))

print("\n========== RECURSOS ==========")
print(consumo.round(4).to_string(index=False))

print(f"\nResultados y graficas disponibles en: {ruta}")
