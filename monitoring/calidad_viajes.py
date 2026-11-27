"""Controles de calidad para viajes de evaluación externa."""

import pandas as pd


CAMPOS_EVENTO = [
    "viaje_id",
    "inicio_viaje",
    "fin_viaje",
    "origen_id",
    "destino_id",
]


def deduplicar_eventos(datos: pd.DataFrame):
    """Conserva una copia por evento operacional.

    No modifica el DataFrame recibido.

    Rechaza identificadores nulos y aquellos asociados
    a eventos operacionalmente distintos.
    """

    faltantes = set(CAMPOS_EVENTO) - set(datos.columns)

    if faltantes:
        raise ValueError(
            f"Columnas faltantes: {faltantes}"
        )

    if datos["viaje_id"].isna().any():
        raise ValueError(
            "Existen identificadores de viaje nulos."
        )

    unicos = datos.drop_duplicates(
        subset=CAMPOS_EVENTO,
        keep="first",
    ).copy()

    conflictos = unicos["viaje_id"].duplicated(
        keep=False
    )

    if conflictos.any():

        cantidad = unicos.loc[
            conflictos, "viaje_id"
        ].nunique()

        raise ValueError(
            f"Existen {cantidad} IDs asociados "
            "a eventos operacionalmente distintos."
        )

    eliminados = len(datos) - len(unicos)

    return unicos, int(eliminados)
