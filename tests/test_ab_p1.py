"""Validaciones del esquema y del análisis A/B."""
import numpy as np
import pandas as pd
import pytest

from mibici.features import P1_FEATURES
from monitoring.comparar_ab_p1 import (
    bootstrap_pareado,
    metricas,
    validar_test,
)


def test_metricas_identicas():
    y = [1., 2., 3.]
    m = metricas(y, y)
    assert m["mae"] == 0
    assert m["wape"] == 0


def test_rechaza_predicciones_negativas():
    with pytest.raises(ValueError, match="inválidas"):
        metricas([1., 2.], [-1., 2.])


def test_rechaza_predicciones_no_finitas():
    with pytest.raises(ValueError, match="inválidas"):
        metricas([1., 2.], [np.inf, 2.])


def test_bootstrap_reproducible():
    diario = pd.DataFrame({
        "fecha": pd.date_range("2025-11-01", periods=14),
        "n": [5] * 14,
        "abs_a": [10.] * 14,
        "abs_b": [8.] * 14,
    })
    a = bootstrap_pareado(diario, 200, 7, 42)
    b = bootstrap_pareado(diario, 200, 7, 42)
    assert a == b
    assert np.isclose(a["ic95_inferior"], 0.4)
    assert np.isclose(a["ic95_superior"], 0.4)


def test_rechaza_fuga_temporal_test():
    df = pd.DataFrame({
        "estacion": [1],
        "fecha": ["2025-10-31"],
        "viajes_real": [10.],
        **{c: [1.] for c in P1_FEATURES},
    })
    with pytest.raises(ValueError, match="fuera del periodo"):
        validar_test(df, {"filas": {"test": 1}})


def test_rechaza_test_duplicado():
    df = pd.DataFrame({
        "estacion": [1, 1],
        "fecha": ["2025-11-01", "2025-11-01"],
        "viajes_real": [10., 11.],
        **{c: [1., 2.] for c in P1_FEATURES},
    })
    with pytest.raises(ValueError, match="duplicada"):
        validar_test(df, {"filas": {"test": 2}})
