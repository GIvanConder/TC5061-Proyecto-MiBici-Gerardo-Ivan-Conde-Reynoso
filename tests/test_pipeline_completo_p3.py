"""Pruebas independientes de la composicion del Pipeline P3."""
import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from mibici.features import P3_MODEL_FEATURES
from serving.registrar_pipeline_p3 import (
    construir_pipeline_p3,
    verificar_entrada,
)


def conjunto_sintetico():
    n = 20
    indices = np.arange(n, dtype=float)
    datos = {
        "salidas_dia": 1.0 + indices,
        "llegadas_dia": 2.0 + indices,
        "balance_relativo": np.linspace(-0.5, 0.5, n),
    }
    for i, columna in enumerate(P3_MODEL_FEATURES[3:]):
        datos[columna] = np.mod(indices + i, 7) / 7.0

    return pd.DataFrame(datos)[P3_MODEL_FEATURES]


@pytest.fixture
def modelos():
    raw = conjunto_sintetico()
    transformado = raw.copy()

    for col in ("salidas_dia", "llegadas_dia"):
        transformado[col] = np.log1p(transformado[col])

    scaler = StandardScaler()
    x = pd.DataFrame(
        scaler.fit_transform(transformado),
        columns=P3_MODEL_FEATURES,
    )

    km = KMeans(
        n_clusters=2,
        random_state=42,
        n_init=10,
    ).fit(x)

    centroides = km.cluster_centers_.copy()
    pipeline = construir_pipeline_p3(raw, scaler, km)

    return raw, x, km, pipeline, centroides


def test_pipeline_reproduce_transformacion_original(modelos):
    raw, x, _, pipeline, _ = modelos
    calculado = pipeline[:-1].transform(raw)

    np.testing.assert_allclose(
        calculado.to_numpy(),
        x.to_numpy(),
        rtol=0,
        atol=1e-10,
    )


def test_pipeline_reproduce_clusters(modelos):
    raw, x, km, pipeline, _ = modelos
    np.testing.assert_array_equal(
        pipeline.predict(raw),
        km.predict(x),
    )


def test_pipeline_no_reentrena_kmeans(modelos):
    _, _, km, pipeline, centroides = modelos
    np.testing.assert_array_equal(
        km.cluster_centers_,
        centroides,
    )
    assert pipeline.named_steps["modelo"] is km


def test_rechaza_intensidad_negativa():
    datos = conjunto_sintetico()
    datos.loc[0, "salidas_dia"] = -2

    with pytest.raises(ValueError, match="Intensidad negativa"):
        verificar_entrada(datos)


def test_rechaza_esquema_incompleto():
    datos = conjunto_sintetico().drop(
        columns=["llegadas_dia"]
    )

    with pytest.raises(ValueError, match="Esquema"):
        verificar_entrada(datos)


def test_rechaza_valores_infinitos():
    datos = conjunto_sintetico()
    datos.loc[0, "balance_relativo"] = np.inf

    with pytest.raises(ValueError, match="no finitas"):
        verificar_entrada(datos)
