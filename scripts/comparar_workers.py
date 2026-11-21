"""Regenera la comparacion historica de configuraciones Uvicorn."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def comparar(base):
    original = (
        base / "ejecucion_original.txt"
    ).read_text().strip()

    optimizada = (
        base / "ejecucion_optimizada.txt"
    ).read_text().strip()

    def cargar(run):
        rendimiento = pd.read_csv(
            base / run / "resumen_rendimiento.csv"
        )

        recursos = pd.read_csv(
            base / run / "resumen_recursos.csv"
        )

        api = recursos[
            (recursos["servicio"] == "API")
            & recursos["escenario"].isin(
                ["baja", "media", "alta"]
            )
        ]

        return rendimiento.merge(
            api[[
                "escenario",
                "cpu_media_pct",
                "memoria_media_mb",
                "memoria_max_mb",
            ]],
            on="escenario",
            validate="one_to_one",
        )

    comparacion = cargar(original).merge(
        cargar(optimizada),
        on=["escenario", "usuarios"],
        suffixes=("_1worker", "_2workers"),
        validate="one_to_one",
    )

    for nueva, anterior in [
        ("variacion_rps_pct", "rps_exitosas"),
        ("variacion_p95_pct", "p95_ms"),
        ("variacion_memoria_pct", "memoria_media_mb"),
    ]:
        comparacion[nueva] = 100 * (
            comparacion[anterior + "_2workers"]
            / comparacion[anterior + "_1worker"] - 1
        )

    comparacion.to_csv(
        base / "comparacion_workers.csv",
        index=False,
        float_format="%.6f",
    )

    (base / "comparacion_workers.json").write_text(
        json.dumps({
            "ejecucion_original": original,
            "ejecucion_optimizada": optimizada,
            "comparacion": comparacion.to_dict("records"),
        }, indent=2, allow_nan=False) + "\n"
    )

    for metrica, titulo, unidad, archivo in [
        (
            "rps_exitosas",
            "Rendimiento por configuracion",
            "Inferencias exitosas por segundo",
            "06_comparacion_rps.png",
        ),
        (
            "p95_ms",
            "Latencia p95 por configuracion",
            "Milisegundos",
            "07_comparacion_p95.png",
        ),
        (
            "memoria_media_mb",
            "Memoria de FastAPI por configuracion",
            "MB",
            "08_comparacion_memoria.png",
        ),
        (
            "cpu_media_pct",
            "CPU de FastAPI por configuracion",
            "CPU (%)",
            "09_comparacion_cpu.png",
        ),
    ]:
        tabla = comparacion.set_index("usuarios")[[
            metrica + "_1worker",
            metrica + "_2workers",
        ]]

        tabla.columns = ["1 worker", "2 workers"]

        ax = tabla.plot.bar(
            figsize=(8, 4.5),
            rot=0,
        )

        ax.set_title(titulo)
        ax.set_xlabel("Usuarios concurrentes")
        ax.set_ylabel(unidad)
        ax.grid(axis="y", alpha=0.25)

        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(base / archivo, dpi=220)
        plt.close(fig)

    print("Comparacion de workers regenerada")
    print(comparacion[[
        "usuarios",
        "variacion_rps_pct",
        "variacion_p95_pct",
        "variacion_memoria_pct",
    ]].to_string(index=False))

    return comparacion


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base",
        type=Path,
        default=ROOT / "docs/evidencias/operacion_finops/locust",
    )

    args = parser.parse_args()

    comparar(args.base.resolve())
