"""Pruebas del motor de alertas y gobernanza P1."""

import numpy as np
import pandas as pd
import pytest

from mibici.features import P1_FEATURES

from monitoring.politica_ct_p1 import (
    evaluar_meses,
    validar_panel,
    evaluar_semanas,
)


def tabla_mensual(maes):
    return pd.DataFrame({
        "periodo": pd.period_range(
            "2026-01",
            periods=len(maes),
            freq="M",
        ).astype(str),
        "mae": maes,
    })


def test_alerta_aislada_no_solicita_reentrenamiento():

    resultado = evaluar_meses(
        tabla_mensual([10, 12, 10]),
        10,
    )

    assert resultado["alerta_desempeno"].tolist() == [
        False, True, False,
    ]

    assert not resultado["nueva_solicitud_evaluacion"].any()


def test_persistencia_genera_una_solicitud():

    resultado = evaluar_meses(
        tabla_mensual([10, 12, 13, 14]),
        10,
    )

    assert resultado["meses_consecutivos_alerta"].tolist() == [
        0, 1, 2, 3,
    ]

    assert resultado["nueva_solicitud_evaluacion"].tolist() == [
        False, False, True, False,
    ]

    assert not resultado["promocion_autorizada"].any()


def test_recuperacion_reinicia_contador():

    resultado = evaluar_meses(
        tabla_mensual([12, 10, 12, 13]),
        10,
    )

    assert resultado["meses_consecutivos_alerta"].tolist() == [
        1, 0, 1, 2,
    ]


def test_umbral_inclusivo():

    resultado = evaluar_meses(
        tabla_mensual([10 * 1.15]),
        10,
    )

    assert resultado["alerta_desempeno"].iloc[0]


def test_rechaza_meses_faltantes():

    datos = pd.DataFrame({
        "periodo": ["2026-01", "2026-03"],
        "mae": [12, 13],
    })

    with pytest.raises(ValueError, match="meses faltantes"):
        evaluar_meses(datos, 10)


def test_rechaza_mae_no_finito():

    with pytest.raises(ValueError, match="MAE mensual inválido"):
        evaluar_meses(tabla_mensual([np.nan]), 10)


def panel_prueba():

    datos = pd.DataFrame({
        c: [1.0, 1.0] for c in P1_FEATURES
    })

    datos["estacion"] = [1, 2]

    datos["fecha"] = pd.to_datetime([
        "2026-01-01",
        "2026-01-01",
    ])

    datos["viajes_real"] = [10, 20]

    return datos


def test_features_iguales_con_claves_distintas_son_validas():

    datos = panel_prueba()

    assert datos.duplicated(P1_FEATURES).sum() == 1

    resultado = validar_panel(datos)

    assert len(resultado) == 2


def test_rechaza_claves_repetidas():

    datos = panel_prueba()

    datos["estacion"] = 1

    with pytest.raises(
        ValueError,
        match="estación-fecha duplicadas",
    ):
        validar_panel(datos)


def test_rechaza_features_no_finitas():

    datos = panel_prueba()

    datos.loc[0, "lag_1"] = np.inf

    with pytest.raises(
        ValueError,
        match="Características no finitas",
    ):
        validar_panel(datos)


def test_semanas_parciales_no_generan_alerta_extrapolacion():

    datos = panel_prueba()

    datos["dias_activa"] = 400

    umbrales = {
        "fuera_soporte_fraccion": 0.1,
        "capacidad_no_positiva_alerta": 1,
    }

    resultado = evaluar_semanas(
        datos,
        28,
        272,
        umbrales,
    )

    assert len(resultado) == 1
    assert not resultado["semana_completa"].iloc[0]
    assert pd.isna(resultado["alerta_extrapolacion"].iloc[0])
