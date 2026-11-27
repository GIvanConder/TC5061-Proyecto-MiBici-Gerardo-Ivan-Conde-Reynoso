"""Adquiere datos externos sin sobrescribir el histórico de 2025."""

import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

from mibici.dataset import (
    COLUMNAS_VIAJES,
    _normalizar_nombre,
)


FUENTES = Path("monitoring/fuentes_2026.json")
RAW = Path("data/monitoring_2026/raw")
MANIFEST = Path(
    "data/monitoring_2026/manifest_descarga.json"
)

REQUERIDAS = {
    "viaje_id",
    "inicio_viaje",
    "fin_viaje",
    "origen_id",
    "destino_id",
}


def verificar_cabecera(ruta):

    for codificacion in ("utf-8-sig", "latin-1"):

        try:

            columnas = pd.read_csv(
                ruta,
                nrows=0,
                encoding=codificacion,
            ).columns

        except UnicodeDecodeError:
            continue

        normalizadas = [
            _normalizar_nombre(c)
            for c in columnas
        ]

        esquema = [
            COLUMNAS_VIAJES.get(c, c)
            for c in normalizadas
        ]

        faltantes = REQUERIDAS - set(esquema)

        if faltantes:
            raise ValueError(
                f"Columnas faltantes: {faltantes}"
            )

        return {
            "codificacion_cabecera": codificacion,
            "columnas_requeridas_presentes": True,
        }

    raise ValueError("No se pudo leer el encabezado")


def sha256(ruta):

    with ruta.open("rb") as archivo:

        return hashlib.file_digest(
            archivo,
            "sha256",
        ).hexdigest()


def main():

    RAW.mkdir(parents=True, exist_ok=True)

    fuentes = json.loads(
        FUENTES.read_text()
    )

    resultados = {}
    pendientes = {}

    sesion = requests.Session()

    sesion.headers.update({
        "User-Agent": "Mozilla/5.0 (proyecto academico TC5061)",
        "Accept": "text/csv,*/*;q=0.8",
    })

    for mes, url in fuentes["meses"].items():

        nombre = Path(urlparse(url).path).name

        ruta = RAW / nombre
        parcial = ruta.with_suffix(".csv.part")

        print(f"\n--- {mes} ---", flush=True)

        try:

            if not ruta.exists():

                print("Descargando:", url, flush=True)

                with sesion.get(
                    url,
                    stream=True,
                    timeout=(20, 120),
                ) as respuesta:

                    respuesta.raise_for_status()

                    with parcial.open("wb") as archivo:

                        for bloque in respuesta.iter_content(
                            chunk_size=1024 * 1024
                        ):

                            if bloque:
                                archivo.write(bloque)

                verificar_cabecera(parcial)

                parcial.replace(ruta)

            else:

                print(
                    "Archivo local existente; verificando."
                )

            cabecera = verificar_cabecera(ruta)

            resultados[mes] = {
                "archivo": nombre,
                "url": url,
                "bytes": ruta.stat().st_size,
                "sha256": sha256(ruta),
                **cabecera,
            }

            print(
                f"OK: {ruta.stat().st_size / 1e6:.2f} MB"
            )

            print(
                "SHA-256:",
                resultados[mes]["sha256"],
            )

        except Exception as error:

            pendientes[mes] = str(error)

            print("PENDIENTE:", error)

            # Solo se elimina una descarga incompleta.
            # Los CSV existentes nunca se eliminan.
            parcial.unlink(missing_ok=True)

    manifest = {
        "fuente": fuentes["fuente"],
        "periodo": "2026-01/2026-10",
        "archivos": resultados,
        "pendientes": pendientes,
        "descarga_completa": not pendientes,
        "validacion": (
            "Cabeceras y hashes; contenido y cobertura "
            "temporal pendientes de validacion integral."
        ),
    }

    MANIFEST.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ) + "\n"
    )

    print("\n========== RESUMEN DE DESCARGAS ==========")

    print("Meses verificados:", len(resultados))
    print("Meses pendientes:", len(pendientes))

    print(
        "Volumen total MB:",
        round(
            sum(
                v["bytes"] for v in resultados.values()
            ) / 1e6,
            2,
        ),
    )

    print("Manifest independiente:", MANIFEST)

    if pendientes:

        for mes, error in pendientes.items():
            print(mes, error)

        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
