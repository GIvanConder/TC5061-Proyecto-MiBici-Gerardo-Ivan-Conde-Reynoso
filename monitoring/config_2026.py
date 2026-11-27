"""Configuración independiente para monitoreo temporal P1."""

from pathlib import Path

BASE_LOCAL = Path("data/monitoring_2026")
RAW = BASE_LOCAL / "raw"

SALIDA = Path("docs/evidencias/monitoreo_final")

MANIFEST_DESCARGA = BASE_LOCAL / "manifest_descarga.json"

INICIO_HISTORICO = "2025-01-01"
INICIO_ACTUAL = "2026-01-01"
FIN_ACTUAL = "2026-11-01"

MESES = [f"2026-{m:02d}" for m in range(1, 11)]

# Misma definición del calendario histórico:
# descansos obligatorios + Jueves y Viernes Santo.
FERIADOS_2026 = [
    "2026-01-01",
    "2026-02-02",
    "2026-03-16",
    "2026-04-02",
    "2026-04-03",
    "2026-05-01",
    "2026-09-16",
]

FUENTES_CALENDARIO = [
    "https://www.gob.mx/profedet/articulos/"
    "sabes-cuales-son-los-dias-de-descanso-obligatorio-para-este-2026",
    "https://www.gob.mx/profedet/articulos/"
    "la-semana-santa-tiene-dias-de-descanso-obligatorio",
]
