"""Validaciones de calidad de datos para el proyecto MiBici."""

from pathlib import Path

import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import DataDuplicates, PercentOfNulls


PROJ_ROOT = Path(__file__).resolve().parents[1]
ESTACIONES_PATH = PROJ_ROOT / "data/processed/estaciones.parquet"


def validar_estaciones():
    """Verifica estructura, faltantes y duplicados del catálogo."""
    estaciones = pd.read_parquet(ESTACIONES_PATH)

    columnas_esperadas = {
        "id", "name", "latitude", "longitude", "dpcapacity"
    }

    if set(estaciones.columns) != columnas_esperadas:
        raise ValueError("El esquema de estaciones no coincide con el esperado.")

    dataset = Dataset(estaciones, cat_features=["name"])

    faltantes = PercentOfNulls().run(dataset).value
    duplicados = DataDuplicates().run(dataset).value
    ids_duplicados = int(estaciones["id"].duplicated().sum())

    print("=== INTEGRIDAD DE ESTACIONES ===")
    print("Registros:", len(estaciones))
    print("\nPorcentaje de valores faltantes:")
    print(faltantes)
    print("\nProporción de filas duplicadas:", duplicados)
    print("IDs duplicados:", ids_duplicados)

    if faltantes.iloc[:, 0].gt(0).any():
        raise ValueError("Se detectaron valores faltantes en estaciones.")

    if duplicados > 0 or ids_duplicados > 0:
        raise ValueError("Se detectaron duplicados en estaciones.")

    # Reglas complementarias de integridad del catálogo.
    tipos_validos = (
        pd.api.types.is_integer_dtype(estaciones["id"])
        and pd.api.types.is_integer_dtype(estaciones["dpcapacity"])
        and pd.api.types.is_numeric_dtype(estaciones["latitude"])
        and pd.api.types.is_numeric_dtype(estaciones["longitude"])
    )

    if not tipos_validos:
        raise ValueError("Tipos numéricos inválidos en estaciones.")

    coordenadas_invalidas = int((
        ~estaciones["latitude"].between(-90, 90)
        | ~estaciones["longitude"].between(-180, 180)
    ).sum())

    capacidades_negativas = int(
        (estaciones["dpcapacity"] < 0).sum()
    )
    capacidades_cero = int(
        (estaciones["dpcapacity"] == 0).sum()
    )
    nombres_vacios = int(
        estaciones["name"].astype("string").str.strip().eq("").sum()
    )

    print("\n=== VALIDACIONES COMPLEMENTARIAS ===")
    print("Coordenadas inválidas:", coordenadas_invalidas)
    print("Capacidades negativas:", capacidades_negativas)
    print("Nombres vacíos:", nombres_vacios)
    print("Capacidades cero:", capacidades_cero)

    if coordenadas_invalidas or capacidades_negativas or nombres_vacios:
        raise ValueError("Se detectaron atributos inválidos en estaciones.")

    if capacidades_cero:
        print("ADVERTENCIA: estaciones con capacidad cero.")
        print("Requieren interpretación según el catálogo de origen.")

    print("\nResultado: VALIDACIÓN APROBADA")


def validar_viajes():
    """Audita la integridad del dataset completo de viajes."""
    import json

    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    ruta_viajes = PROJ_ROOT / "data/processed/viajes.parquet"
    ruta_manifest = PROJ_ROOT / "data/manifest.json"

    manifest = json.loads(ruta_manifest.read_text(encoding="utf-8"))
    archivo = pq.ParquetFile(ruta_viajes)

    esquema_esperado = {
        "viaje_id": "int64",
        "inicio_viaje": "timestamp[us]",
        "fin_viaje": "timestamp[us]",
        "origen_id": "int64",
        "destino_id": "int64",
        "mes_archivo": "large_string",
    }

    esquema_actual = {
        campo.name: str(campo.type)
        for campo in archivo.schema_arrow
    }

    if esquema_actual != esquema_esperado:
        raise ValueError("El esquema de viajes no coincide con el esperado.")

    total = 0
    nulos = {columna: 0 for columna in esquema_esperado}
    fechas_invertidas = 0
    duracion_cero = 0

    for lote in archivo.iter_batches(batch_size=100_000):
        df = lote.to_pandas()
        total += len(df)

        conteo_nulos = df.isna().sum()
        for columna in nulos:
            nulos[columna] += int(conteo_nulos[columna])

        fechas_invertidas += int(
            (df["fin_viaje"] < df["inicio_viaje"]).sum()
        )
        duracion_cero += int(
            (df["fin_viaje"] == df["inicio_viaje"]).sum()
        )

    tabla_ids = pq.read_table(ruta_viajes, columns=["viaje_id"])
    ids_unicos = pc.count_distinct(tabla_ids["viaje_id"]).as_py()
    ids_repetidos = total - ids_unicos

    print("\n=== INTEGRIDAD COMPLETA DE VIAJES ===")
    print("Versión de datos:", manifest["data_version"])
    print("Registros evaluados:", total)
    print("Nulos por columna:", nulos)
    print("IDs repetidos:", ids_repetidos)
    print("Fechas invertidas:", fechas_invertidas)
    print("Duración cero:", duracion_cero)

    if total != manifest["total_viajes"]:
        raise ValueError("El total no coincide con el manifiesto.")

    if total != archivo.metadata.num_rows:
        raise ValueError("El total no coincide con los metadatos Parquet.")

    if any(nulos.values()):
        raise ValueError("Se detectaron valores faltantes en viajes.")

    if ids_repetidos > 0:
        raise ValueError("Se detectaron identificadores duplicados.")

    if fechas_invertidas > 0:
        raise ValueError("Se detectaron fechas invertidas.")

    if duracion_cero > 0:
        print("ADVERTENCIA: existen viajes de duración cero.")
        print("P3 los excluye mediante su filtrado específico.")

    print("Resultado: INTEGRIDAD APROBADA")


def validar_muestra_viajes():
    """Ejecuta DeepChecks sobre una muestra reproducible de viajes."""
    import pyarrow.parquet as pq

    ruta_viajes = PROJ_ROOT / "data/processed/viajes.parquet"
    archivo = pq.ParquetFile(ruta_viajes)

    muestras = []

    for numero, lote in enumerate(
        archivo.iter_batches(batch_size=100_000)
    ):
        df = lote.to_pandas()
        muestra_lote = df.sample(
            n=min(200, len(df)),
            random_state=42 + numero,
        )
        muestras.append(muestra_lote)

    if not muestras:
        raise ValueError("No se encontraron viajes para validar.")

    muestra = pd.concat(muestras, ignore_index=True)

    dataset = Dataset(
        muestra,
        cat_features=["mes_archivo"],
    )

    faltantes = PercentOfNulls().run(dataset).value
    duplicados = DataDuplicates().run(dataset).value

    print("\n=== DEEPCHECKS: MUESTRA DE VIAJES ===")
    print("Registros muestreados:", len(muestra))
    print("Meses representados:",
          muestra["inicio_viaje"].dt.to_period("M").nunique())
    print("Valores faltantes:")
    print(faltantes)
    print("Proporción de duplicados:", duplicados)

    if faltantes.iloc[:, 0].gt(0).any():
        raise ValueError("DeepChecks detectó valores faltantes.")

    if duplicados > 0:
        raise ValueError("DeepChecks detectó filas duplicadas.")

    print("Resultado: DEEPCHECKS APROBADO")

    return muestra


def validar_drift(muestra: pd.DataFrame):
    """Compara distribuciones temporales sin seleccionar modelos."""
    from deepchecks.tabular.checks import FeatureDrift

    from mibici.config import (
        P2_TRAIN_START,
        P2_VAL_START,
        P2_TEST_START,
        P2_TEST_END,
    )

    datos = muestra.copy()

    datos["duracion_min"] = (
        datos["fin_viaje"] - datos["inicio_viaje"]
    ).dt.total_seconds() / 60

    datos["hora_inicio"] = datos["inicio_viaje"].dt.hour

    fechas = datos["inicio_viaje"]
    columnas = ["duracion_min", "hora_inicio"]

    train = datos.loc[
        (fechas >= P2_TRAIN_START)
        & (fechas < P2_VAL_START),
        columnas,
    ]

    test = datos.loc[
        (fechas >= P2_TEST_START)
        & (fechas < P2_TEST_END),
        columnas,
    ]

    if train.empty or test.empty:
        raise ValueError("TRAIN o TEST no contiene registros.")

    train_ds = Dataset(
        train, cat_features=["hora_inicio"]
    )
    test_ds = Dataset(
        test, cat_features=["hora_inicio"]
    )

    resultado = FeatureDrift().run(train_ds, test_ds)

    print("\n=== DEEPCHECKS: FEATURE DRIFT ===")
    print("TRAIN:", len(train), "registros")
    print("TEST:", len(test), "registros")

    for variable, valores in resultado.value.items():
        print(
            f"{variable}: "
            f"score={valores['Drift score']:.6f}, "
            f"metodo={valores['Method']}"
        )

    print("Resultado: ANALISIS DE DRIFT EJECUTADO")
    print("Los scores requieren interpretacion contextual.")


def main():
    validar_estaciones()
    validar_viajes()
    muestra = validar_muestra_viajes()
    validar_drift(muestra)


if __name__ == "__main__":
    main()
