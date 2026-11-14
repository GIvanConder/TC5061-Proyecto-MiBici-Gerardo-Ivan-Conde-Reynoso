"""Prepara las particiones originales de P1 para drift y A/B."""
import json
from pathlib import Path

from mibici.features import P1_FEATURES
from mibici.modeling.p1_regresion import preparar_datos_p1

SALIDA = Path("reports/p1/drift_abtest")
SALIDA.mkdir(parents=True, exist_ok=True)

particiones, metadata, _ = preparar_datos_p1()

nombres = ("train", "val", "test")
conteos = {}

for nombre, (x, y) in zip(nombres, particiones[:3]):
    assert x.index.equals(y.index)
    assert list(x.columns) == P1_FEATURES
    assert not x.isna().any().any()
    assert not y.isna().any()

    datos = x.copy()
    datos["viajes_real"] = y

    ruta = SALIDA / f"{nombre}.parquet"
    datos.reset_index().to_parquet(ruta, index=False)

    conteos[nombre] = len(datos)
    print(f"{nombre}: {len(datos):,} filas -> {ruta}")

manifiesto = json.loads(
    Path("data/manifest.json").read_text(encoding="utf-8")
)

info = {
    "data_version": manifiesto["data_version"],
    "modelo": "mibici-equipo32-p1-demanda",
    "version_champion": 1,
    "referencia": "enero-septiembre 2025",
    "validacion": "octubre 2025",
    "actual_evaluacion": "noviembre-diciembre 2025",
    "variables": P1_FEATURES,
    "objetivo": "viajes_real",
    "filas": conteos,
    "metadata_original": metadata,
}

(SALIDA / "particiones.json").write_text(
    json.dumps(info, indent=2, ensure_ascii=False, default=str),
    encoding="utf-8",
)

print("OK: particiones y trazabilidad preparadas")
