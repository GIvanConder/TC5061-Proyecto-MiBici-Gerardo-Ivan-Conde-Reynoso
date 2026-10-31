"""Pruebas de las reglas de integridad del catálogo MiBici."""

import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import pandas as pd

from validation import deepchecks_mibici as val


class TestReglasEstaciones(unittest.TestCase):

    def setUp(self):
        self.estaciones = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "name": ["Estación A", "Estación B", "Estación C", "ex-Estación D"],
            "latitude": [20.67, 20.68, 20.69, 20.70],
            "longitude": [-103.35, -103.36, -103.37, -103.38],
            "dpcapacity": [20, 25, 30, 0],
        })

    def ejecutar(self, datos):
        with patch.object(val.pd, "read_parquet", return_value=datos):
            with redirect_stdout(StringIO()):
                val.validar_estaciones()

    def verificar_rechazo(self, columna, valor, mensaje):
        datos = self.estaciones.copy()
        datos.loc[0, columna] = valor

        with self.assertRaisesRegex(ValueError, mensaje):
            self.ejecutar(datos)

    def test_latitud_invalida(self):
        self.verificar_rechazo(
            "latitude", 100, "atributos inválidos"
        )

    def test_longitud_invalida(self):
        self.verificar_rechazo(
            "longitude", -200, "atributos inválidos"
        )

    def test_capacidad_negativa(self):
        self.verificar_rechazo(
            "dpcapacity", -1, "atributos inválidos"
        )

    def test_nombre_vacio(self):
        self.verificar_rechazo(
            "name", "  ", "atributos inválidos"
        )

    def test_identificador_duplicado(self):
        self.verificar_rechazo(
            "id", 2, "duplicados en estaciones"
        )

    def test_capacidad_cero_permitida(self):
        # Capacidad cero es advertencia, no error.
        self.ejecutar(self.estaciones)


if __name__ == "__main__":
    unittest.main()
