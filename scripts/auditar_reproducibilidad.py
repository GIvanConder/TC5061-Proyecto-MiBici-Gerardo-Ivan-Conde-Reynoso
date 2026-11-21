"""Reproduce resultados versionados sin repetir pruebas de carga."""

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REL = Path("docs/evidencias/operacion_finops")


def ejecutar(script, cwd, env=None):
    subprocess.run(
        [sys.executable, str(script)],
        cwd=cwd,
        env=env,
        check=True,
        stdin=subprocess.DEVNULL,
    )


def auditar():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()

    with tempfile.TemporaryDirectory(
        prefix="mibici-auditoria-"
    ) as temporal:

        temp = Path(temporal)
        archivo = temp / "repositorio.tar"
        copia = temp / "repo"

        copia.mkdir()

        with archivo.open("wb") as salida:
            subprocess.run(
                ["git", "archive", "HEAD"],
                cwd=ROOT,
                stdout=salida,
                stdin=subprocess.DEVNULL,
                check=True,
            )

        with tarfile.open(archivo) as tar:
            tar.extractall(copia, filter="data")

        base = copia / REL
        locust = base / "locust"
        finops = base / "finops"

        tablas = [
            Path("finops/mediciones_consolidadas.csv"),
            Path("finops/escenarios_finops.csv"),
            Path("finops/costo_demanda_fija.csv"),
            Path("locust/comparacion_workers.csv"),
        ]

        runs = []

        for marcador in [
            "ejecucion_original.txt",
            "ejecucion_optimizada.txt",
        ]:
            run = (locust / marcador).read_text().strip()
            runs.append(run)

            for nombre in [
                "resumen_rendimiento.csv",
                "resumen_recursos.csv",
            ]:
                tablas.append(
                    Path("locust") / run / nombre
                )

        # Conservar las tablas originales de Git antes
        # de regenerarlas en la copia temporal.
        originales = {
            str(ruta): pd.read_csv(base / ruta)
            for ruta in tablas
        }


        # Eliminar graficas exclusivamente de la copia temporal.
        # Esto demuestra su regeneracion efectiva.
        graficas_originales = list(locust.rglob("*.png"))
        graficas_originales += list(finops.glob("*.png"))

        assert len(graficas_originales) == 16

        for grafica in graficas_originales:
            assert grafica.resolve().is_relative_to(copia.resolve())
            grafica.unlink()

        print("16 graficas originales retiradas de la copia temporal")

        for run in runs:
            ejecutar(
                copia / "scripts/analizar_carga.py",
                copia,
                {
                    **os.environ,
                    "MIBICI_RUN_ID": run,
                },
            )

        ejecutar(
            copia / "scripts/analizar_finops.py",
            copia,
        )

        subprocess.run(
            [
                sys.executable,
                str(copia / "scripts/comparar_workers.py"),
                "--base",
                str(locust),
            ],
            cwd=copia,
            stdin=subprocess.DEVNULL,
            check=True,
        )

        verificadas = []

        for ruta in tablas:
            regenerada = pd.read_csv(base / ruta)

            pd.testing.assert_frame_equal(
                originales[str(ruta)],
                regenerada,
                check_exact=False,
                rtol=1e-8,
                atol=1e-10,
            )

            verificadas.append(str(ruta))

            print(f"REPRODUCIDO: {ruta}")

        resultado = json.loads(
            (finops / "resumen_finops.json").read_text()
        )

        assert resultado["solicitudes_evaluadas"] == 90239
        assert resultado["inferencias_exitosas"] == 90239
        assert resultado["fallos"] == 0

        escenarios = pd.read_csv(
            finops / "escenarios_finops.csv"
        )

        assert len(escenarios) == 24

        graficas = list(locust.rglob("*.png"))
        graficas += list(finops.glob("*.png"))

        assert len(graficas) == 16
        assert all(p.stat().st_size > 0 for p in graficas)

        return {
            "fecha_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "commit_auditado": commit,
            "experimentos": runs,
            "tablas_verificadas": verificadas,
            "total_tablas": len(verificadas),
            "graficas_regeneradas": len(graficas),
            "inferencias_exitosas": 90239,
            "fallos": 0,
            "escenarios_finops": len(escenarios),
            "metodo": (
                "git archive y regeneracion "
                "con dependencias instaladas"
            ),
            "resultado": "APROBADO",
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output", type=Path)

    args = parser.parse_args()

    resultado = auditar()

    contenido = json.dumps(
        resultado,
        indent=2,
        ensure_ascii=False,
    ) + "\n"

    if args.output:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.output.write_text(
            contenido,
            encoding="utf-8",
        )

    print(contenido)
    print("REPRODUCIBILIDAD APROBADA")
