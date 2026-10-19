"""Pruebas de las variables de P1: el panel es contiguo y no hay fuga de información."""

import pandas as pd
import pytest

from mibici.features import LeakageError, p1_agregar_features, p1_construir_panel_diario


@pytest.fixture
def viajes():
    # Estación 1: 1 viaje el día d (d = 1..20). Estación 2 empieza el día 10.
    filas = []
    for d in range(1, 21):
        filas += [{"inicio_viaje": pd.Timestamp(2025, 1, d, 8), "origen_id": 1}] * d
        if d >= 10:
            filas.append({"inicio_viaje": pd.Timestamp(2025, 1, d, 9), "origen_id": 2})
    return pd.DataFrame(filas)


def test_panel_contiguo_desde_primera_actividad(viajes):
    panel = p1_construir_panel_diario(viajes, "2025-01-01", "2025-01-21")
    assert panel.loc[1].shape[0] == 20
    assert panel.loc[2].index.min() == pd.Timestamp("2025-01-10")
    assert panel.loc[(1, pd.Timestamp("2025-01-05")), "viajes"] == 5


def test_rezagos_solo_usan_dias_anteriores(viajes):
    panel = p1_construir_panel_diario(viajes, "2025-01-01", "2025-01-21")
    df = p1_agregar_features(panel, pd.Series({1: 20, 2: 15}), [], [1, 7], [7])
    fila = df.loc[(1, pd.Timestamp("2025-01-15"))]
    assert fila["lag_1"] == 14
    assert fila["lag_7"] == 8
    assert fila["media_7d"] == pytest.approx(sum(range(8, 15)) / 7)  # días 8..14, no incluye el 15


def test_rezago_menor_al_horizonte_lanza_error(viajes):
    panel = p1_construir_panel_diario(viajes, "2025-01-01", "2025-01-21")
    with pytest.raises(LeakageError):
        p1_agregar_features(panel, pd.Series(dtype=float), [], [0, 7], [7], horizonte=1)
