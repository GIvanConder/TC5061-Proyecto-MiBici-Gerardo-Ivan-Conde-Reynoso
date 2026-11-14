"""Pruebas de controles temporales del challenger."""
import pandas as pd
import pytest

from mibici.features import P1_FEATURES
from monitoring.registrar_challenger_p1 import validar_particion


def datos_ejemplo():
    valores = {col: [1.0, 2.0] for col in P1_FEATURES}
    return pd.DataFrame({
        "estacion": [1, 1],
        "fecha": ["2025-10-01", "2025-10-02"],
        "viajes_real": [10.0, 11.0],
        **valores,
    })


def validar(df):
    return validar_particion(
        df, "VALIDATION", "2025-10-01", "2025-11-01"
    )


def test_particion_valida():
    assert len(validar(datos_ejemplo())) == 2


def test_rechaza_fuga_desde_test():
    datos = datos_ejemplo()
    datos.loc[1, "fecha"] = "2025-11-01"

    with pytest.raises(ValueError, match="fuera del periodo"):
        validar(datos)


def test_rechaza_observaciones_duplicadas():
    datos = datos_ejemplo()
    datos.loc[1, "fecha"] = datos.loc[0, "fecha"]

    with pytest.raises(ValueError, match="duplicada"):
        validar(datos)


def test_rechaza_valores_faltantes():
    datos = datos_ejemplo()
    datos.loc[0, "lag_7"] = None

    with pytest.raises(ValueError, match="faltantes"):
        validar(datos)


def test_rechaza_objetivo_negativo():
    datos = datos_ejemplo()
    datos.loc[0, "viajes_real"] = -1

    with pytest.raises(ValueError, match="objetivo"):
        validar(datos)


def test_rechaza_columna_faltante():
    datos = datos_ejemplo().drop(columns=["lag_14"])

    with pytest.raises(ValueError, match="faltan columnas"):
        validar(datos)


def test_rechaza_valor_infinito():
    datos = datos_ejemplo()
    datos.loc[0, "lag_7"] = float("inf")

    with pytest.raises(ValueError, match="no finitas"):
        validar(datos)


def test_rechaza_objetivo_infinito():
    datos = datos_ejemplo()
    datos.loc[0, "viajes_real"] = float("inf")

    with pytest.raises(ValueError, match="objetivo"):
        validar(datos)
