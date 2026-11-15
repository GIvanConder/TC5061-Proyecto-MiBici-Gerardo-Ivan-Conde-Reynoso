"""Pruebas de proteccion y promocion explicita de alias."""
from types import SimpleNamespace

import pytest
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    RESOURCE_DOES_NOT_EXIST,
)

from mibici import model_registry
from mibici.registry_aliases import (
    asegurar_alias_inicial,
    obtener_alias,
    promover_alias,
)


class RegistryFalso:
    def __init__(self, champion=1):
        self.aliases = {"serving": 2}

        if champion is not None:
            self.aliases["champion"] = champion

        self.cambios_alias = []
        self.error_consulta = None

    def get_registered_model(self, nombre):
        if self.error_consulta is not None:
            raise self.error_consulta

        return SimpleNamespace(
            aliases={
                alias: str(version)
                for alias, version in self.aliases.items()
            }
        )

    def get_model_version_by_alias(self, nombre, alias):
        if self.error_consulta is not None:
            raise self.error_consulta

        if alias not in self.aliases:
            raise MlflowException(
                "Alias no encontrado",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        return SimpleNamespace(
            version=str(self.aliases[alias])
        )

    def set_registered_model_alias(
        self, name, alias, version
    ):
        self.aliases[alias] = int(version)
        self.cambios_alias.append(
            (name, alias, int(version))
        )

    def get_model_version(self, nombre, version):
        if int(version) not in (1, 2):
            raise MlflowException(
                "Version no encontrada",
                error_code=RESOURCE_DOES_NOT_EXIST,
            )

        return SimpleNamespace(
            version=str(version),
            status="READY",
        )

    def get_run(self, run_id):
        return SimpleNamespace(
            data=SimpleNamespace(
                tags={
                    "git_commit": "commit-test",
                    "data_version": "datos-test",
                },
                params={},
            )
        )

    def search_model_versions(self, filtro):
        return [
            SimpleNamespace(
                version="1",
                run_id="run-original",
            )
        ]

    def update_registered_model(self, **kwargs):
        pass

    def update_model_version(self, **kwargs):
        pass

    def set_model_version_tag(self, **kwargs):
        pass

    def set_registered_model_tag(self, *args):
        pass


NOMBRE = "mibici-prueba-p3"


def test_inicializa_champion_ausente():
    client = RegistryFalso(champion=None)

    resultado = asegurar_alias_inicial(
        client, NOMBRE, "1"
    )

    assert resultado["accion"] == "inicializado"
    assert client.aliases["champion"] == 1
    assert len(client.cambios_alias) == 1


def test_registro_repetido_es_idempotente():
    client = RegistryFalso(champion=1)

    for _ in range(3):
        asegurar_alias_inicial(client, NOMBRE, "1")

    assert client.aliases["champion"] == 1
    assert client.cambios_alias == []


def test_no_revierte_promocion_existente():
    client = RegistryFalso(champion=2)

    resultado = asegurar_alias_inicial(
        client, NOMBRE, "1"
    )

    assert resultado["accion"] == "conservado"
    assert resultado["version"] == "2"
    assert client.aliases["champion"] == 2
    assert client.cambios_alias == []


def test_serving_permanece_independiente():
    client = RegistryFalso(champion=1)

    asegurar_alias_inicial(client, NOMBRE, "1")

    assert client.aliases == {
        "champion": 1,
        "serving": 2,
    }


def test_no_oculta_errores_de_registry():
    client = RegistryFalso()
    client.error_consulta = MlflowException(
        "Fallo de conexion",
        error_code=INTERNAL_ERROR,
    )

    with pytest.raises(MlflowException):
        obtener_alias(client, NOMBRE)


def test_promocion_sin_confirmacion_no_escribe():
    client = RegistryFalso(champion=1)

    resultado = promover_alias(
        client, NOMBRE, 2, 1
    )

    assert resultado is False
    assert client.aliases["champion"] == 1
    assert client.cambios_alias == []


def test_promocion_rechaza_version_esperada_incorrecta():
    client = RegistryFalso(champion=2)

    with pytest.raises(RuntimeError, match="cambio"):
        promover_alias(
            client, NOMBRE, 1, 1,
            confirmar=True,
        )

    assert client.cambios_alias == []


def test_promocion_rechaza_destino_ausente():
    client = RegistryFalso(champion=1)

    with pytest.raises(MlflowException):
        promover_alias(
            client, NOMBRE, 99, 1,
            confirmar=True,
        )

    assert client.cambios_alias == []


def test_promocion_explicita_verificada():
    client = RegistryFalso(champion=1)

    resultado = promover_alias(
        client, NOMBRE, 2, 1,
        confirmar=True,
    )

    assert resultado is True
    assert client.aliases["champion"] == 2
    assert client.cambios_alias == [
        (NOMBRE, "champion", 2)
    ]


def test_registro_general_no_desplaza_champion(
    monkeypatch,
):
    """Integracion: registrar el ganador no revierte v2."""
    client = RegistryFalso(champion=2)

    def prohibir_registro(*args, **kwargs):
        raise AssertionError(
            "No debe registrar una version duplicada"
        )

    monkeypatch.setattr(
        model_registry.mlflow,
        "register_model",
        prohibir_registro,
    )

    spec = {
        "registered_name": NOMBRE,
        "run_id": "run-original",
        "model_uri": "models:/ejemplo",
        "description": "Modelo de prueba",
        "version_description": "Version historica",
    }

    version = model_registry.registrar_modelo(
        client,
        "p3",
        spec,
    )

    assert str(version) == "1"
    assert client.aliases["champion"] == 2
    assert client.aliases["serving"] == 2
    assert client.cambios_alias == []


def test_codigos_reales_de_excepcion_mlflow():
    """Verifica la semantica de MlflowException."""
    ausente = MlflowException(
        "Alias inexistente",
        error_code=RESOURCE_DOES_NOT_EXIST,
    )
    interno = MlflowException(
        "Error del servidor",
        error_code=INTERNAL_ERROR,
    )

    assert ausente.error_code == "RESOURCE_DOES_NOT_EXIST"
    assert interno.error_code == "INTERNAL_ERROR"


def test_serving_ausente_devuelve_none():
    """P1 o P2 no necesitan tener alias serving."""
    client = RegistryFalso(champion=1)
    del client.aliases["serving"]

    resultado = obtener_alias(
        client, NOMBRE, "serving"
    )

    assert resultado is None
    assert client.aliases["champion"] == 1
    assert client.cambios_alias == []


def test_modelo_inexistente_no_se_confunde_con_alias_ausente():
    """Un modelo inexistente debe generar un error real."""
    client = RegistryFalso(champion=None)

    client.error_consulta = MlflowException(
        "Modelo no encontrado",
        error_code=RESOURCE_DOES_NOT_EXIST,
    )

    with pytest.raises(MlflowException):
        obtener_alias(client, NOMBRE)

    assert client.cambios_alias == []
