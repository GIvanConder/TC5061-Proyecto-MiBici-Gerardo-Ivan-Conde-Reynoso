"""Pruebas de calidad y monitoreo temporal del proyecto final."""

import numpy as np
import pandas as pd
import pytest

from mibici.features import (
    p1_construir_panel_diario,
    p1_agregar_features,
)

from monitoring.calidad_viajes import deduplicar_eventos

from monitoring.impacto_drift_p1 import (
    clasificar_soporte,
    metricas,
)

from monitoring.config_2026 import FERIADOS_2026


def viajes_prueba():

    return pd.DataFrame({
        "viaje_id": [1, 1, 2],
        "inicio_viaje": pd.to_datetime([
            "2026-09-01",
            "2026-09-01",
            "2026-09-02",
        ]),
        "fin_viaje": pd.to_datetime([
            "2026-09-01 00:05:00",
            "2026-09-01 00:05:00",
            "2026-09-02 00:05:00",
        ]),
        "origen_id": [10, 10, 20],
        "destino_id": [11, 11, 21],
    })


def test_deduplicacion_operacional():

    datos = viajes_prueba()

    original = datos.copy(deep=True)

    resultado, eliminados = deduplicar_eventos(datos)

    assert len(resultado) == 2

    assert eliminados == 1

    assert resultado["viaje_id"].is_unique

    pd.testing.assert_frame_equal(datos, original)


def test_deduplicacion_idempotente():

    datos, _ = deduplicar_eventos(viajes_prueba())

    resultado, eliminados = deduplicar_eventos(datos)

    assert eliminados == 0

    pd.testing.assert_frame_equal(resultado, datos)


def test_rechaza_ids_conflictivos():

    datos = viajes_prueba()

    datos.loc[1, "destino_id"] = 99

    with pytest.raises(
        ValueError,
        match="eventos operacionalmente distintos",
    ):
        deduplicar_eventos(datos)


def test_rechaza_ids_nulos():

    datos = viajes_prueba()

    datos["viaje_id"] = datos["viaje_id"].astype("Int64")

    datos.loc[0, "viaje_id"] = pd.NA

    with pytest.raises(
        ValueError,
        match="nulos",
    ):
        deduplicar_eventos(datos)


def test_rechaza_esquema_incompleto():

    datos = viajes_prueba().drop(
        columns="destino_id"
    )

    with pytest.raises(
        ValueError,
        match="Columnas faltantes",
    ):
        deduplicar_eventos(datos)


def test_limites_soporte():

    valores = pd.Series([
        27, 28, 100, 272, 273,
    ])

    assert clasificar_soporte(
        valores, 28, 272,
    ).tolist() == [
        "inferior_train",
        "dentro_train",
        "dentro_train",
        "dentro_train",
        "superior_train",
    ]


def test_metricas_predictivas():

    resultado = metricas(
        [10, 20],
        [12, 18],
    )

    assert resultado["mae"] == 2
    assert resultado["rmse"] == 2
    assert resultado["sesgo"] == 0

    assert np.isclose(
        resultado["wape"],
        4 / 30,
    )


def test_segmentos_vacios():

    assert metricas([], [])["mae"] is None

    assert metricas([0], [1])["wape"] is None


def test_rechaza_predicciones_invalidas():

    with pytest.raises(ValueError):

        metricas([1], [float("nan")])

    with pytest.raises(ValueError):

        metricas([1], [-1])


def construir_panel_prueba():

    fechas = pd.date_range(
        "2025-12-01",
        "2026-01-05",
        freq="D",
    )

    viajes = pd.DataFrame({
        "inicio_viaje": fechas,
        "origen_id": 1,
    })

    return p1_construir_panel_diario(
        viajes,
        "2025-12-01",
        "2026-01-06",
    )


def test_continuidad_entre_anios():

    panel = construir_panel_prueba()

    datos = p1_agregar_features(
        panel,
        pd.Series({1: 20}),
        FERIADOS_2026,
        [1, 2, 7, 14],
        [7, 28],
        1,
    )

    anterior = datos.loc[
        (1, pd.Timestamp("2025-12-31"))
    ]

    actual = datos.loc[
        (1, pd.Timestamp("2026-01-01"))
    ]

    assert actual["dias_activa"] == (
        anterior["dias_activa"] + 1
    )

    assert actual["lag_1"] == anterior["viajes"]

    assert actual["media_28d"] == 1

    assert actual["feriado"] == 1


def test_features_no_utilizan_objetivo_del_dia():

    panel = construir_panel_prueba()

    alterado = panel.copy()

    indice = (
        1,
        pd.Timestamp("2026-01-01"),
    )

    alterado.loc[indice, "viajes"] = 999

    argumentos = (
        pd.Series({1: 20}),
        FERIADOS_2026,
        [1, 2, 7, 14],
        [7, 28],
        1,
    )

    original = p1_agregar_features(
        panel, *argumentos
    )

    nuevo = p1_agregar_features(
        alterado, *argumentos
    )

    pd.testing.assert_series_equal(
        original.loc[indice].drop("viajes"),
        nuevo.loc[indice].drop("viajes"),
    )
