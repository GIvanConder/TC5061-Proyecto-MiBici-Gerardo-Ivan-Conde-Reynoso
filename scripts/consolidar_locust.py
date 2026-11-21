"""Recupera las metricas finales sin alterar los CSV originales."""

import json
import re
from pathlib import Path


def leer_consola(ruta):
    texto = Path(ruta).read_text(encoding="utf-8")

    patron = (
        r"^\s*POST\s+/predict\s+"
        r"(\d+)\s+(\d+)\s*\([^)]*\)\s*\|\s*"
        r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
        r"\s*\|\s*([\d.]+)"
    )

    coincidencias = list(
        re.finditer(patron, texto, re.MULTILINE)
    )

    assert coincidencias, "Resumen final no encontrado"

    datos = coincidencias[-1].groups()

    total = int(datos[0])
    fallos = int(datos[1])

    percentiles = []

    for linea in texto.splitlines():
        partes = linea.split()

        if (
            len(partes) == 14
            and partes[:2] == ["POST", "/predict"]
        ):
            valores = partes[2:]

            if all(v.isdigit() for v in valores):
                percentiles.append(
                    [int(v) for v in valores]
                )

    assert percentiles, "Percentiles no encontrados"

    valores = percentiles[-1]

    assert valores[-1] == total
    assert 0 <= fallos <= total
    assert total > 0

    return {
        "solicitudes": total,
        "fallos": fallos,
        "rps": float(datos[6]),
        "latencia_media_ms": float(datos[2]),
        "p50_ms": valores[0],
        "p95_ms": valores[5],
        "p99_ms": valores[7],
    }


def metricas_finales(destino):
    destino = Path(destino)

    consola = leer_consola(
        destino / "ejecucion.log"
    )

    ruta = destino / "locust_final.json"

    if ruta.is_file():
        datos = json.loads(ruta.read_text())

        for campo in (
            "solicitudes", "fallos",
            "p50_ms", "p95_ms", "p99_ms",
        ):
            assert datos[campo] == consola[campo], (
                f"Diferencia entre JSON y consola: {campo}"
            )

        datos["fuente"] = "JSON final de Locust"

    else:
        datos = consola
        datos["fuente"] = "Resumen final de consola"

    assert datos["rps"] > 0

    return datos
