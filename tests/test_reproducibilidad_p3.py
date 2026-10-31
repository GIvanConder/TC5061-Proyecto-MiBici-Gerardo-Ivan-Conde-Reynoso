"""Pruebas de configuración y reproducibilidad del entrenamiento P3."""

import numpy as np

from mibici.config import P3_N_INIT, SEED
from mibici.modeling.p3_clustering import construir_kmeans


def test_constructor_kmeans_respeta_configuracion():
    modelo = construir_kmeans(n_clusters=3)

    assert modelo.n_clusters == 3
    assert modelo.n_init == P3_N_INIT
    assert modelo.random_state == SEED


def test_kmeans_reproducible_con_misma_semilla():
    # Tres grupos sintéticos claramente diferenciados.
    datos = np.array([
        [0.0, 0.0],
        [0.2, 0.1],
        [-0.1, 0.2],
        [0.1, -0.2],
        [10.0, 10.0],
        [10.2, 10.1],
        [9.8, 10.0],
        [10.1, 9.9],
        [-10.0, 10.0],
        [-10.2, 10.1],
        [-9.8, 9.9],
        [-10.1, 10.2],
    ])

    modelo_a = construir_kmeans(n_clusters=3)
    modelo_b = construir_kmeans(n_clusters=3)

    modelo_a.fit(datos)
    modelo_b.fit(datos)

    # Las asignaciones deben ser idénticas.
    np.testing.assert_array_equal(
        modelo_a.labels_,
        modelo_b.labels_,
    )

    # Los centroides calculados deben coincidir.
    np.testing.assert_allclose(
        modelo_a.cluster_centers_,
        modelo_b.cluster_centers_,
        rtol=0,
        atol=0,
    )

    # La métrica de entrenamiento debe ser idéntica.
    np.testing.assert_allclose(modelo_a.inertia_, modelo_b.inertia_, rtol=0, atol=1e-12)

    # Las predicciones también deben coincidir.
    np.testing.assert_array_equal(
        modelo_a.predict(datos),
        modelo_b.predict(datos),
    )
