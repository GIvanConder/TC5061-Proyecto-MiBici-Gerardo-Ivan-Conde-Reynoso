"""Controles de integridad para los resultados publicados del A/B."""
import pandas as pd
import pytest

from monitoring.exportar_evidencias_ab_p1 import verificar_consistencia


@pytest.fixture
def evidencia():
    fechas = pd.date_range("2025-11-01", "2025-12-31")

    diario = pd.DataFrame({
        "fecha": fechas,
        "n": [10] * len(fechas),
        "abs_a": [50.0] * len(fechas),
        "abs_b": [55.0] * len(fechas),
    })
    diario["mae_a"] = diario["abs_a"] / diario["n"]
    diario["mae_b"] = diario["abs_b"] / diario["n"]
    diario["mejora_mae"] = diario["mae_a"] - diario["mae_b"]

    mensual = pd.DataFrame({
        "mes": ["2025-11", "2025-12"],
        "filas": [300, 310],
        "mae_champion": [5.0, 5.0],
        "mae_challenger": [5.5, 5.5],
    })

    resumen = {
        "tipo": "comparacion_ab_retrospectiva_pareada",
        "aleatorizacion_en_produccion": False,
        "champion": {"version": 1, "metricas": {"mae": 5.0}},
        "challenger": {"version": 2, "metricas": {"mae": 5.5}},
        "dias": 61,
        "observaciones": 610,
        "diferencia_mae_champion_menos_challenger": -0.5,
    }

    return resumen, diario, mensual


def test_evidencias_coherentes(evidencia):
    assert verificar_consistencia(*evidencia) == pytest.approx(-0.5)


def test_rechaza_numero_observaciones_incorrecto(evidencia):
    resumen, diario, mensual = evidencia
    resumen["observaciones"] = 611

    with pytest.raises(ValueError, match="observaciones inconsistente"):
        verificar_consistencia(resumen, diario, mensual)


def test_rechaza_mae_publicado_incorrecto(evidencia):
    resumen, diario, mensual = evidencia
    resumen["challenger"]["metricas"]["mae"] = 5.1

    with pytest.raises(ValueError, match="MAE inconsistente"):
        verificar_consistencia(resumen, diario, mensual)


def test_rechaza_fechas_incompletas(evidencia):
    resumen, diario, mensual = evidencia
    diario = diario.iloc[:-1].copy()

    with pytest.raises(ValueError, match="Fechas diarias incompletas"):
        verificar_consistencia(resumen, diario, mensual)


def test_rechaza_mae_mensual_adulterado(evidencia):
    resumen, diario, mensual = evidencia
    mensual.loc[1, "mae_challenger"] = 4.0

    with pytest.raises(ValueError, match="MAE mensual inconsistente"):
        verificar_consistencia(resumen, diario, mensual)
