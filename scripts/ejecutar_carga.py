"""Ejecuta Locust y registra recursos de Podman por escenario."""

import csv
import json
import os
import platform
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/evidencias/operacion_finops/locust"

LOCUST = os.environ["LOCUST_BIN"]
HOST = "http://127.0.0.1:8000"

CONTENEDORES = [
    os.getenv("MIBICI_API_CONTAINER", "mibici-finops_api_1"),
    "mibici-finops_mlflow_1",
]

ESCENARIOS = [
    ("reposo", 0, 30),
    ("calentamiento", 2, 30),
    ("baja", 5, 120),
    ("media", 20, 180),
    ("alta", 50, 180),
]


def ahora():
    return datetime.now(timezone.utc).isoformat()


def comando(args):
    return subprocess.check_output(
        args,
        cwd=ROOT,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=20,
    ).strip()


def salud():
    with urlopen(HOST + "/health", timeout=10) as r:
        datos = json.load(r)

    assert datos["status"] == "ok"
    assert datos["model_version"] == 2

    return datos


def memoria_bytes(texto):
    uso = texto.split("/")[0].strip()

    match = re.fullmatch(
        r"([0-9.]+)\s*([A-Za-z]+)",
        uso,
    )
    if not match:
        raise ValueError(f"Memoria no interpretable: {texto}")

    cantidad = float(match.group(1))
    unidad = match.group(2).upper()

    factores = {
        "B": 1,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "TB": 1000**4,
        "KIB": 1024,
        "MIB": 1024**2,
        "GIB": 1024**3,
        "TIB": 1024**4,
    }

    return int(cantidad * factores[unidad])


def muestra_podman():
    formato = "{{.Name}}|{{.CPUNano}}|{{.MemUsage}}"

    antes = time.monotonic()

    salida = comando([
        "podman", "stats", "--no-stream",
        "--format", formato,
        *CONTENEDORES,
    ])

    despues = time.monotonic()
    instante = (antes + despues) / 2

    registros = []

    for linea in salida.splitlines():
        nombre, cpu, memoria = linea.strip().split("|")

        registros.append({
            "contenedor": nombre,
            "cpu_ns": int(cpu),
            "memoria_bytes": memoria_bytes(memoria),
        })

    assert {
        r["contenedor"] for r in registros
    } == set(CONTENEDORES)

    return instante, registros


def monitorear(ruta, evento, inicio, errores):
    previos = {}

    campos = [
        "fecha_utc",
        "segundos",
        "contenedor",
        "cpu_acumulada_ns",
        "cpu_intervalo_pct",
        "memoria_bytes",
    ]

    try:
        with ruta.open("w", newline="") as archivo:
            writer = csv.DictWriter(
                archivo,
                fieldnames=campos,
            )
            writer.writeheader()

            while not evento.is_set():
                instante, registros = muestra_podman()

                for registro in registros:
                    nombre = registro["contenedor"]
                    cpu = registro["cpu_ns"]
                    porcentaje = ""

                    if nombre in previos:
                        anterior_t, anterior_cpu = previos[nombre]

                        delta = cpu - anterior_cpu
                        intervalo = instante - anterior_t

                        assert delta >= 0
                        assert intervalo > 0

                        porcentaje = round(
                            100 * delta / (intervalo * 1e9),
                            4,
                        )

                    previos[nombre] = (instante, cpu)

                    writer.writerow({
                        "fecha_utc": ahora(),
                        "segundos": round(instante - inicio, 3),
                        "contenedor": nombre,
                        "cpu_acumulada_ns": cpu,
                        "cpu_intervalo_pct": porcentaje,
                        "memoria_bytes": registro["memoria_bytes"],
                    })

                archivo.flush()
                evento.wait(2)

    except Exception as exc:
        errores.append(repr(exc))
        evento.set()


def ejecutar_escenario(base, nombre, usuarios, segundos):
    destino = base / nombre
    destino.mkdir()

    print(
        f"\nEscenario: {nombre} | "
        f"{usuarios} usuarios | {segundos} s",
        flush=True,
    )

    salud_previa = salud()
    inicio = time.monotonic()
    inicio_utc = ahora()

    evento = threading.Event()
    errores = []

    monitor = threading.Thread(
        target=monitorear,
        args=(
            destino / "recursos.csv",
            evento,
            inicio,
            errores,
        ),
        daemon=True,
    )

    monitor.start()
    codigo = 0

    try:
        if usuarios:
            args = [
                LOCUST,
                "-f",
                str(ROOT / "load_testing/locustfile.py"),
                "--headless",
                "--host", HOST,
                "--users", str(usuarios),
                "--spawn-rate", "10",
                "--run-time", f"{segundos}s",
                "--stop-timeout", "10",
                "--only-summary",
                "--csv", str(destino / "locust"),
                "--csv-full-history",
                "--html", str(destino / "reporte.html"),
                "--exit-code-on-error", "0",
            ]

            with (destino / "ejecucion.log").open("w") as log:
                resultado = subprocess.run(
                    args,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    timeout=segundos + 90,
                    env={
                        **os.environ,
                        "MIBICI_LOCUST_FINAL_JSON": str(destino / "locust_final.json"),
                    },
                )

            codigo = resultado.returncode

            assert codigo == 0, (
                f"Locust regreso codigo {codigo}"
            )

            for archivo in [
                "locust_final.json",
                "locust_stats.csv",
                "locust_stats_history.csv",
                "locust_failures.csv",
                "reporte.html",
            ]:
                assert (destino / archivo).is_file(), (
                    f"Falta evidencia: {archivo}"
                )
        else:
            time.sleep(segundos)

    finally:
        evento.set()
        monitor.join(timeout=30)

    assert not monitor.is_alive()
    assert not errores, errores

    salud_posterior = salud()

    metadata = {
        "escenario": nombre,
        "usuarios": usuarios,
        "duracion_programada_s": segundos,
        "duracion_observada_s": round(
            time.monotonic() - inicio, 3
        ),
        "inicio_utc": inicio_utc,
        "fin_utc": ahora(),
        "codigo_locust": codigo,
        "salud_previa": salud_previa,
        "salud_posterior": salud_posterior,
    }

    (destino / "escenario.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    if usuarios:
        with (destino / "ejecucion.log").open() as archivo:
            print("".join(archivo.readlines()[-15:]))

    print(f"Escenario terminado: {nombre}", flush=True)


def main():
    assert salud()["model_version"] == 2

    instante, muestra = muestra_podman()
    print("Monitoreo inicial:", muestra)

    identificador = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    base = BASE / identificador
    base.mkdir(parents=True)

    metadata = {
        "identificador": identificador,
        "fecha_utc": ahora(),
        "commit_base": comando(["git", "rev-parse", "HEAD"]),
        "rama": comando(["git", "branch", "--show-current"]),
        "sistema": platform.system(),
        "arquitectura": platform.machine(),
        "kernel": platform.release(),
        "cpu_logicas_visibles": os.cpu_count(),
        "memoria_visible_bytes": (
            os.sysconf("SC_PAGE_SIZE")
            * os.sysconf("SC_PHYS_PAGES")
        ),
        "podman": comando(["podman", "--version"]),
        "locust": comando([LOCUST, "--version"]),
        "modelo": salud(),
        "contenedores": CONTENEDORES,
        "imagen": comando([
            "podman",
            "inspect",
            "--format", "{{.Image}}",
            CONTENEDORES[0],
        ]),
        "metodo": "Carga cerrada con espera de 0.1 a 0.3 s",
        "configuracion": os.getenv("MIBICI_EXPERIMENT_LABEL", "1_worker"),
    }

    (base / "entorno.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    BASE.joinpath("ultima_ejecucion.txt").write_text(
        identificador + "\n"
    )

    print(json.dumps(metadata, indent=2))

    for nombre, usuarios, segundos in ESCENARIOS:
        ejecutar_escenario(
            base, nombre, usuarios, segundos
        )

        if nombre != "alta":
            time.sleep(5)

    print(f"EXPERIMENTO COMPLETADO: {base}")


if __name__ == "__main__":
    main()
