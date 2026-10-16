"""Ingesta reproducible de datos de MiBici — Equipo 32.

Descarga los meses definidos en config.py, normaliza columnas y
codificación, guarda Parquet y versiona los datos con un manifiesto
de hashes. Ejecutable sin pasos manuales:

    python -m mibici.dataset
"""

import hashlib
import json
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import typer
from loguru import logger

from mibici.config import (
    ESTACIONES_PARQUET,
    MANIFEST_PATH,
    NOMENCLATURA_URL,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    VIAJES_PARQUET,
    VIAJES_URLS,
)

app = typer.Typer()

HEADERS = {"User-Agent": "Mozilla/5.0 (proyecto academico TC5061)"}

# Mapeo de nombres de columna: los archivos de MiBici cambian
# nombres y acentos entre años; todo se normaliza a este esquema.
COLUMNAS_VIAJES = {
    "viaje_id": "viaje_id",
    "usuario_id": "usuario_id",
    "genero": "genero",
    "ano_de_nacimiento": "anio_nacimiento",
    "anio_de_nacimiento": "anio_nacimiento",
    "a_o_de_nacimiento": "anio_nacimiento",
    "inicio_del_viaje": "inicio_viaje",
    "fin_del_viaje": "fin_viaje",
    "origen_id": "origen_id",
    "destino_id": "destino_id",
}


def _normalizar_nombre(col: str) -> str:
    """Quita acentos, pasa a minúsculas y deja solo [a-z0-9_]."""
    col = unicodedata.normalize("NFKD", col).encode("ascii", "ignore").decode()
    col = col.strip().lower().replace(" ", "_")
    return "".join(c if c.isalnum() or c == "_" else "_" for c in col)


def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def descargar(url: str, destino: Path) -> Path:
    """Descarga un archivo si no existe ya en data/raw (idempotente)."""
    if destino.exists():
        logger.info(f"Ya existe, no se descarga: {destino.name}")
        return destino
    logger.info(f"Descargando {url}")
    r = requests.get(url, headers=HEADERS, timeout=120)
    r.raise_for_status()
    destino.write_bytes(r.content)
    return destino


def leer_csv(ruta: Path) -> pd.DataFrame:
    """Lee un CSV tolerando las codificaciones usadas por MiBici."""
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(ruta, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"No se pudo decodificar {ruta}")


def normalizar_viajes(df: pd.DataFrame, origen: str) -> pd.DataFrame:
    """Normaliza nombres de columna, tipos de fecha y agrega el mes origen."""
    df.columns = [_normalizar_nombre(c) for c in df.columns]
    df = df.rename(columns={c: COLUMNAS_VIAJES[c] for c in df.columns if c in COLUMNAS_VIAJES})
    df["inicio_viaje"] = pd.to_datetime(df["inicio_viaje"], errors="coerce")
    df["fin_viaje"] = pd.to_datetime(df["fin_viaje"], errors="coerce")
    df["mes_archivo"] = origen
    return df


@app.command()
def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifiesto = {
        "generado": datetime.now(timezone.utc).isoformat(),
        "archivos": {},
    }

    # ---- 1. Viajes ----
    partes = []
    for mes, url in VIAJES_URLS.items():
        ruta = descargar(url, RAW_DATA_DIR / f"viajes_{mes}.csv")
        df = normalizar_viajes(leer_csv(ruta), origen=mes)
        manifiesto["archivos"][ruta.name] = {
            "url": url,
            "sha256": _sha256(ruta),
            "registros": len(df),
        }
        partes.append(df)
    viajes = pd.concat(partes, ignore_index=True)

    # ---- 2. Nomenclatura de estaciones ----
    ruta_nom = descargar(NOMENCLATURA_URL, RAW_DATA_DIR / "nomenclatura.csv")
    estaciones = leer_csv(ruta_nom)
    estaciones.columns = [_normalizar_nombre(c) for c in estaciones.columns]
    manifiesto["archivos"][ruta_nom.name] = {
        "url": NOMENCLATURA_URL,
        "sha256": _sha256(ruta_nom),
        "registros": len(estaciones),
    }

    # ---- 3. Guardar en formato eficiente ----
    viajes.to_parquet(VIAJES_PARQUET, index=False)
    estaciones.to_parquet(ESTACIONES_PARQUET, index=False)

    # ---- 4. Versión de los datos ----
    hashes = sorted(v["sha256"] for v in manifiesto["archivos"].values())
    manifiesto["data_version"] = hashlib.sha256("".join(hashes).encode()).hexdigest()[:12]
    manifiesto["total_viajes"] = len(viajes)
    MANIFEST_PATH.write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False))

    logger.success(
        f"Ingesta completa: {len(viajes):,} viajes, {len(estaciones)} estaciones, "
        f"data_version={manifiesto['data_version']}"
    )


if __name__ == "__main__":
    app()