"""Proteccion de alias de MLflow Model Registry.

El registro ordinario nunca reasigna un alias existente.
Las promociones requieren una accion explicita.
"""
import argparse
import os

import mlflow
from mlflow import MlflowClient


def obtener_alias(client, nombre, alias="champion"):
    """Obtener un alias sin confundir ausencia con fallas del Registry.

    Consulta primero el modelo registrado, cuya propiedad aliases
    contiene las asociaciones existentes.
    """
    registrado = client.get_registered_model(nombre)

    aliases = registrado.aliases
    if not isinstance(aliases, dict):
        raise TypeError(
            "El Registry no devolvio un diccionario de alias"
        )

    version = aliases.get(alias)

    if version is None:
        return None

    return client.get_model_version(nombre, str(version))


def asegurar_alias_inicial(
    client, nombre, version, alias="champion"
):
    """Inicializa un alias ausente; nunca reemplaza uno existente."""
    actual = obtener_alias(client, nombre, alias)

    if actual is not None:
        return {
            "accion": "conservado",
            "version": str(actual.version),
        }

    client.set_registered_model_alias(
        name=nombre,
        alias=alias,
        version=version,
    )

    confirmado = obtener_alias(client, nombre, alias)

    if confirmado is None or (
        str(confirmado.version) != str(version)
    ):
        raise RuntimeError(
            "No se pudo verificar la inicializacion del alias"
        )

    return {
        "accion": "inicializado",
        "version": str(confirmado.version),
    }


def promover_alias(
    client,
    nombre,
    version_destino,
    version_actual_esperada,
    confirmar=False,
    alias="champion",
):
    """Promocion controlada con comprobacion de version previa."""
    if alias != "champion":
        raise ValueError(
            "La promocion explicita solo administra champion"
        )

    destino = int(version_destino)
    esperada = int(version_actual_esperada)

    if destino < 1 or esperada < 1:
        raise ValueError("Las versiones deben ser positivas")

    actual = obtener_alias(client, nombre, alias)

    if actual is None:
        raise RuntimeError(
            "No existe champion: inicializar mediante registro"
        )

    if int(actual.version) != esperada:
        raise RuntimeError(
            f"Champion cambio: se esperaba v{esperada}, "
            f"pero apunta a v{actual.version}"
        )

    if destino == esperada:
        raise ValueError(
            "La version de destino ya es champion"
        )

    candidata = client.get_model_version(nombre, str(destino))

    if str(candidata.version) != str(destino):
        raise RuntimeError(
            "El Registry devolvio una version inesperada"
        )

    estado = getattr(candidata, "status", None)

    if estado not in (None, "READY"):
        raise RuntimeError(
            f"La version de destino no esta lista: {estado}"
        )

    print(
        f"Promocion solicitada: {nombre} "
        f"champion v{esperada} -> v{destino}"
    )

    if not confirmar:
        print("SIMULACION: no se modifico el Registry")
        return False

    client.set_registered_model_alias(
        name=nombre,
        alias=alias,
        version=str(destino),
    )

    posterior = obtener_alias(client, nombre, alias)

    if posterior is None or int(posterior.version) != destino:
        raise RuntimeError(
            "La promocion no pudo verificarse"
        )

    print(f"Promocion confirmada: champion -> v{destino}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Auditar o promover explicitamente champion"
    )

    parser.add_argument("--model", required=True)
    parser.add_argument("--to-version", type=int)
    parser.add_argument(
        "--expected-current-version", type=int
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Autoriza la modificacion real del alias",
    )
    parser.add_argument(
        "--tracking-uri",
        default=os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://127.0.0.1:5000",
        ),
    )

    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_registry_uri(args.tracking_uri)

    client = MlflowClient()

    if args.to_version is None:
        if args.confirm or (
            args.expected_current_version is not None
        ):
            parser.error(
                "--confirm y --expected-current-version "
                "requieren --to-version"
            )

        actual = obtener_alias(client, args.model)

        print("Modelo:", args.model)
        print(
            "Champion:",
            "ausente" if actual is None
            else f"v{actual.version}",
        )
        print("Operacion: solo lectura")
        return

    if args.expected_current_version is None:
        parser.error(
            "Es obligatorio --expected-current-version"
        )

    promover_alias(
        client=client,
        nombre=args.model,
        version_destino=args.to_version,
        version_actual_esperada=args.expected_current_version,
        confirmar=args.confirm,
    )


if __name__ == "__main__":
    main()
