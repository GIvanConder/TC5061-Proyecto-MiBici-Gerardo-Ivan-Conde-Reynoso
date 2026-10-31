"""Pruebas de construcción de etiquetas de riesgo para P2."""

import pandas as pd
import pytest

from mibici.features import p2_agregar_etiqueta, p2_agregar_features


@pytest.fixture
def panel_p2():
    horas = pd.date_range(
        "2025-01-01 08:00", periods=5, freq="h"
    )
    indice = pd.MultiIndex.from_product(
        [[1, 2], horas],
        names=["estacion", "hora"],
    )
    return pd.DataFrame(
        {
            "neto": [
                0, -1, -1, 0, 0,
                0, 1, 1, 0, 0,
            ]
        },
        index=indice,
    )


def test_etiquetas_de_riesgo_y_umbral(panel_p2):
    resultado = p2_agregar_etiqueta(
        panel_p2,
        horizonte=2,
        capacidad=pd.Series({1: 10, 2: 10}),
        fraccion=0.2,
    )

    hora = pd.Timestamp("2025-01-01 08:00")

    assert resultado.loc[(1, hora), "neto_futuro"] == -2
    assert resultado.loc[(1, hora), "riesgo"] == 1
    assert resultado.loc[(1, hora), "tipo_riesgo"] == "sin_bicis"

    assert resultado.loc[(2, hora), "neto_futuro"] == 2
    assert resultado.loc[(2, hora), "riesgo"] == 1
    assert resultado.loc[(2, hora), "tipo_riesgo"] == "sin_anclajes"

    hora_normal = pd.Timestamp("2025-01-01 09:00")

    for estacion in [1, 2]:
        assert resultado.loc[(estacion, hora_normal), "riesgo"] == 0
        assert resultado.loc[
            (estacion, hora_normal), "tipo_riesgo"
        ] == "normal"


def test_descarta_ventanas_incompletas_y_capacidad_cero(panel_p2):
    resultado = p2_agregar_etiqueta(
        panel_p2,
        horizonte=2,
        capacidad=pd.Series({1: 10, 2: 0}),
        fraccion=0.2,
    )

    assert set(resultado.index.get_level_values("estacion")) == {1}
    assert len(resultado) == 3
    assert resultado.loc[1].index.max() == pd.Timestamp(
        "2025-01-01 10:00"
    )

    # La función no debe alterar el DataFrame original.
    assert "riesgo" not in panel_p2.columns


@pytest.mark.parametrize("horizonte", [0, 24])
def test_horizonte_p2_fuera_de_rango_lanza_error(panel_p2, horizonte):
    """Rechaza horizontes incompatibles con los rezagos temporales."""
    with pytest.raises(
        ValueError,
        match="El horizonte de P2 debe estar entre 1 y 23 horas",
    ):
        p2_agregar_features(panel_p2, horizonte=horizonte)
