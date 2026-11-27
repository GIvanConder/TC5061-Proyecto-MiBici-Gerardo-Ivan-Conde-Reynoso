"""Evaluación retrospectiva externa del champion P1 congelado."""

import hashlib
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mibici.config import (
    ESTACIONES_PARQUET,
    VIAJES_PARQUET,
    P1_FERIADOS,
    P1_LAGS_DIAS,
    P1_VENTANAS_MEDIA_DIAS,
    P1_HORIZONTE_DIAS,
    P1_MIN_DURATION_SECONDS,
    P1_MAX_DURATION_SECONDS,
    P1_TRAIN_START,
    P1_VAL_START,
    P1_TEST_START,
    P1_TEST_END,
)

from mibici.dataset import leer_csv, normalizar_viajes

from mibici.features import (
    P1_FEATURES,
    p1_construir_panel_diario,
    p1_agregar_features,
)

from mibici.modeling.p1_regresion import (
    cargar_viajes,
    cargar_capacidad,
    preparar_datos_p1,
    cortar,
)

from monitoring.config_2026 import (
    BASE_LOCAL,
    RAW,
    SALIDA,
    MANIFEST_DESCARGA,
    INICIO_HISTORICO,
    INICIO_ACTUAL,
    FIN_ACTUAL,
    MESES,
    FERIADOS_2026,
    FUENTES_CALENDARIO,
)

from monitoring.calidad_viajes import deduplicar_eventos

from monitoring.impacto_drift_p1 import (
    cargar_champion,
    clasificar_soporte,
    obtener_coeficiente,
    resumir_grupo,
    metricas,
    RUN_ID,
    MODEL_ID,
    MODEL_NAME,
    MODEL_VERSION,
    DATA_VERSION,
    MAE_TEST_ORIGINAL,
)


def sha256(ruta):
    with ruta.open("rb") as archivo:
        return hashlib.file_digest(
            archivo, "sha256"
        ).hexdigest()


def guardar_json(ruta, datos):
    ruta.write_text(
        json.dumps(
            datos,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )


def cargar_actual():

    manifest = json.loads(
        MANIFEST_DESCARGA.read_text()
    )

    assert manifest["descarga_completa"]
    assert set(manifest["archivos"]) == set(MESES)

    partes = []
    ids_viajes = []
    controles = []

    archivos = {}

    for mes in MESES:

        metadata = manifest["archivos"][mes]
        ruta = RAW / metadata["archivo"]

        assert ruta.is_file()
        assert sha256(ruta) == metadata["sha256"]

        print(f"\n--- Validando {mes} ---", flush=True)

        df = normalizar_viajes(
            leer_csv(ruta),
            origen=mes.replace("-", "_"),
        )

        requeridas = [
            "viaje_id",
            "inicio_viaje",
            "fin_viaje",
            "origen_id",
            "destino_id",
        ]

        assert set(requeridas) <= set(df.columns)
        assert not df.columns.duplicated().any()

        df = df[requeridas].copy()

        total = len(df)
        assert total > 0

        # Tratamiento exclusivo del conjunto externo.
        # Se preservan los CSV originales.
        df, redundantes = deduplicar_eventos(df)

        total_unicos = len(df)

        print(
            f"Deduplicación {mes}: "
            f"{redundantes} copias redundantes retiradas"
        )

        df["origen_id"] = pd.to_numeric(
            df["origen_id"],
            errors="coerce",
        )

        fechas_invalidas = int(
            df[["inicio_viaje", "fin_viaje"]]
            .isna()
            .any(axis=1)
            .sum()
        )

        origen_invalido = (
            df["origen_id"].isna()
            | (df["origen_id"] <= 0)
            | (df["origen_id"] % 1 != 0)
        )

        ids_viajes.append(
            df["viaje_id"].dropna().copy()
        )

        fechas = df["inicio_viaje"]

        mes_inicio = fechas.dt.strftime("%Y-%m")

        fuera_mes = int(
            (fechas.notna() & mes_inicio.ne(mes)).sum()
        )

        en_periodo = fechas.between(
            INICIO_ACTUAL,
            FIN_ACTUAL,
            inclusive="left",
        )

        duracion = (
            df["fin_viaje"] - df["inicio_viaje"]
        ).dt.total_seconds()

        duracion_valida = duracion.between(
            P1_MIN_DURATION_SECONDS,
            P1_MAX_DURATION_SECONDS,
        )

        validos = (
            en_periodo
            & duracion_valida
            & ~origen_invalido
        )

        seleccion = df.loc[
            validos,
            [
                "inicio_viaje",
                "fin_viaje",
                "origen_id",
            ],
        ].copy()

        seleccion["origen_id"] = (
            seleccion["origen_id"].astype("int64")
        )

        control = {
            "mes_archivo": mes,
            "filas_originales": total,
            "copias_operacionales_redundantes": redundantes,
            "filas_unicas": total_unicos,
            "fechas_invalidas": fechas_invalidas,
            "origen_invalido": int(origen_invalido.sum()),
            "viaje_id_nulo": int(df["viaje_id"].isna().sum()),
            "inicio_fuera_mes_archivo": fuera_mes,
            "inicio_fuera_periodo_2026": int(
                (fechas.notna() & ~en_periodo).sum()
            ),
            "duracion_no_valida": int(
                (~duracion_valida).sum()
            ),
            "filas_utilizables_p1": len(seleccion),
            "filas_excluidas_por_calidad_p1": (
                total_unicos - len(seleccion)
            ),
            "filas_excluidas_p1": total - len(seleccion),
        }

        controles.append(control)

        archivos[mes] = {
            **metadata,
            "registros": total,
            "copias_operacionales_redundantes": redundantes,
            "registros_unicos": total_unicos,
            "registros_utilizables_p1": len(seleccion),
        }

        partes.append(seleccion)

        print(json.dumps(control, indent=2))

        del df

    calidad = pd.DataFrame(controles)

    calidad.to_csv(
        SALIDA / "calidad_ingesta_2026.csv",
        index=False,
    )

    ids = pd.concat(
        ids_viajes,
        ignore_index=True,
    )

    duplicados = int(ids.duplicated().sum())

    print("\nIDs de viaje duplicados:", duplicados)

    # Si existen duplicados, se investigan antes de evaluar.
    # No se eliminan silenciosamente.
    if duplicados:
        raise ValueError(
            "Existen IDs de viaje repetidos en 2026. "
            "Revisar origen y cobertura antes de continuar."
        )

    total_filas = int(
        calidad["filas_originales"].sum()
    )

    proporcion_fechas_invalidas = (
        calidad["fechas_invalidas"]
        / calidad["filas_originales"]
    )

    # Guardia preliminar de ingestión, no umbral
    # calibrado de Continuous Training.
    if (proporcion_fechas_invalidas > 0.01).any():
        raise ValueError(
            "Al menos un mes supera 1% de fechas inválidas."
        )

    actual = pd.concat(
        partes,
        ignore_index=True,
    )

    cobertura = (
        actual["inicio_viaje"]
        .dt.normalize()
        .value_counts()
        .sort_index()
        .rename("viajes_validos")
    )

    dias_esperados = pd.date_range(
        INICIO_ACTUAL,
        FIN_ACTUAL,
        inclusive="left",
        freq="D",
    )

    cobertura = cobertura.reindex(
        dias_esperados,
        fill_value=0,
    )

    cobertura.index.name = "fecha"

    cobertura.to_csv(
        SALIDA / "cobertura_diaria_2026.csv"
    )

    if (cobertura == 0).any():
        raise ValueError(
            "Existen días sin viajes válidos en 2026."
        )

    hashes = sorted(
        a["sha256"]
        for a in archivos.values()
    )

    version = hashlib.sha256(
        "".join(hashes).encode()
    ).hexdigest()[:12]

    evidencia = {
        "fuente": manifest["fuente"],
        "data_version_historica": DATA_VERSION,
        "data_version_evaluacion_2026": version,
        "inicio_inclusivo": INICIO_ACTUAL,
        "fin_exclusivo": FIN_ACTUAL,
        "archivos": archivos,
        "total_registros": total_filas,
        "total_registros_unicos": int(
            calidad["filas_unicas"].sum()
        ),
        "copias_operacionales_redundantes_retiradas": int(
            calidad["copias_operacionales_redundantes"].sum()
        ),
        "viajes_utilizables_p1": len(actual),
        "politica_duplicados": (
            "Conservar una copia por ID, inicio, fin, "
            "origen y destino. Rechazar IDs conflictivos "
            "o repetidos entre archivos después del tratamiento."
        ),
        "csv_originales_modificados": False,
        "ids_duplicados": duplicados,
        "dias_cubiertos": int(len(cobertura)),
        "feriados_2026": FERIADOS_2026,
        "fuentes_calendario": FUENTES_CALENDARIO,
    }

    ruta_manifest = SALIDA / "manifest_2026.json"

    if ruta_manifest.exists():

        anterior = json.loads(
            ruta_manifest.read_text()
        )

        assert anterior == evidencia, (
            "La fuente de evaluación cambió respecto "
            "a una ejecución anterior."
        )

    guardar_json(ruta_manifest, evidencia)

    print("\n========== CALIDAD GLOBAL 2026 ==========")
    print("Registros:", total_filas)
    print("Viajes utilizables P1:", len(actual))
    print("Días cubiertos:", len(cobertura))
    print("Data version externa:", version)

    return actual, evidencia


def preparar_panel(actual):

    historico = cargar_viajes(VIAJES_PARQUET)

    capacidad = cargar_capacidad(
        ESTACIONES_PARQUET
    )

    assert capacidad.index.is_unique

    viajes = pd.concat(
        [historico, actual],
        ignore_index=True,
    )

    panel = p1_construir_panel_diario(
        viajes,
        INICIO_HISTORICO,
        FIN_ACTUAL,
    )

    df = p1_agregar_features(
        panel,
        capacidad,
        P1_FERIADOS + FERIADOS_2026,
        P1_LAGS_DIAS,
        P1_VENTANAS_MEDIA_DIAS,
        P1_HORIZONTE_DIAS,
    )

    # Comparación exacta contra la preparación original.
    originales, _, _ = preparar_datos_p1()

    limites = [
        ("train", P1_TRAIN_START, P1_VAL_START),
        ("val", P1_VAL_START, P1_TEST_START),
        ("test", P1_TEST_START, P1_TEST_END),
    ]

    for (nombre, inicio, fin), (x, y) in zip(
        limites,
        originales[:3],
    ):

        ampliado = cortar(
            df,
            inicio,
            fin,
        ).dropna(subset=P1_FEATURES)

        pd.testing.assert_frame_equal(
            ampliado[P1_FEATURES].astype("float64"),
            x,
            check_exact=True,
        )

        pd.testing.assert_series_equal(
            ampliado["viajes"],
            y,
            check_exact=True,
        )

        print(
            f"INVARIANCIA HISTORICA OK: "
            f"{nombre} | {len(x):,} observaciones"
        )

    actual_bruto = cortar(
        df,
        INICIO_ACTUAL,
        FIN_ACTUAL,
    )

    actual_panel = actual_bruto.dropna(
        subset=P1_FEATURES
    )

    estaciones_2026 = set(
        actual["origen_id"].unique()
    )

    estaciones_2025 = set(
        historico["origen_id"].unique()
    )

    estaciones_panel = set(
        actual_panel.index.get_level_values("estacion")
    )

    sin_catalogo = (
        estaciones_2026 - set(capacidad.index)
    )

    info = {
        "filas_panel_antes_dropna": len(actual_bruto),
        "filas_panel_evaluables": len(actual_panel),
        "filas_sin_historia_suficiente": (
            len(actual_bruto) - len(actual_panel)
        ),
        "estaciones_con_viajes_2025": len(estaciones_2025),
        "estaciones_con_viajes_2026": len(estaciones_2026),
        "estaciones_nuevas_observadas": len(
            estaciones_2026 - estaciones_2025
        ),
        "estaciones_historicas_sin_viajes_2026": len(
            estaciones_2025 - estaciones_2026
        ),
        "estaciones_en_panel_evaluable": len(
            estaciones_panel
        ),
        "estaciones_2026_sin_catalogo": len(
            sin_catalogo
        ),
        "filas_capacidad_cero": int(
            actual_panel["capacidad"].eq(0).sum()
        ),
        "limitacion_capacidad": (
            "Se conserva el catálogo histórico de septiembre "
            "de 2026. No hay reconstrucción de capacidades "
            "vigentes para cada fecha de evaluación."
        ),
        "limitacion_actividad": (
            "Los días sin salidas se representan como cero. "
            "No se distingue falta de demanda de suspensión "
            "operativa o retiro de una estación."
        ),
    }

    guardar_json(
        SALIDA / "calidad_panel_2026.json",
        info,
    )

    print("\n========== COBERTURA DEL PANEL ==========")
    print(json.dumps(info, indent=2, ensure_ascii=False))

    return actual_panel, originales, capacidad, estaciones_2025


def evaluar(actual, originales, capacidad, estaciones_2025):

    modelo = cargar_champion()

    train_x, train_y = originales[0]
    test_x, test_y = originales[2]

    minimo = float(train_x["dias_activa"].min())
    maximo = float(train_x["dias_activa"].max())

    mae_test = metricas(
        test_y,
        modelo.predict(test_x),
    )["mae"]

    assert abs(
        mae_test - MAE_TEST_ORIGINAL
    ) < 1e-4

    x = actual[P1_FEATURES].astype("float64")
    y = actual["viajes"].astype("float64")

    assert x.index.equals(y.index)
    assert x.index.is_unique
    assert len(x) > 0
    assert np.isfinite(x.to_numpy()).all()
    assert (y >= 0).all()

    pred = modelo.predict(x)

    x_recorte = x.copy()

    x_recorte["dias_activa"] = (
        x_recorte["dias_activa"].clip(
            minimo, maximo
        )
    )

    pred_recorte = modelo.predict(x_recorte)

    soporte = clasificar_soporte(
        x["dias_activa"],
        minimo,
        maximo,
    )

    estaciones = x.index.get_level_values(
        "estacion"
    )

    observaciones = pd.DataFrame(
        {
            "real": y.to_numpy(),
            "pred": pred,
            "pred_recorte": pred_recorte,
            "lag_7": x["lag_7"].to_numpy(),
            "dias_activa": x["dias_activa"].to_numpy(),
            "soporte": soporte,
            "estacion_nueva": ~estaciones.isin(
                estaciones_2025
            ),
            "sin_catalogo": ~estaciones.isin(
                capacidad.index
            ),
            "capacidad_cero": (
                x["capacidad"].to_numpy() == 0
            ),
        },
        index=x.index,
    ).reset_index()

    observaciones["fecha"] = pd.to_datetime(
        observaciones["fecha"]
    )

    mensual = []
    segmentos = []

    globales = resumir_grupo(
        "total",
        "2026-01/2026-10",
        observaciones,
    )

    for mes, grupo in observaciones.groupby(
        observaciones["fecha"].dt.to_period("M")
    ):

        fila = resumir_grupo(
            "total",
            str(mes),
            grupo,
        )

        fila["fuera_train_pct"] = float(
            100 * (
                grupo["soporte"] != "dentro_train"
            ).mean()
        )

        fila["mae_variacion_vs_test_2025_pct"] = float(
            100 * (
                fila["mae"] / mae_test - 1
            )
        )

        fila["mejora_vs_baseline_pct"] = (
            100 * (
                1 - fila["mae"]
                / fila["mae_baseline_lag7"]
            )
            if fila["mae_baseline_lag7"] > 0
            else None
        )

        mensual.append(fila)

    mascaras = {
        "dentro_train": soporte == "dentro_train",
        "inferior_train": soporte == "inferior_train",
        "superior_train": soporte == "superior_train",
        "estaciones_historicas": (
            ~observaciones["estacion_nueva"]
        ),
        "estaciones_nuevas": (
            observaciones["estacion_nueva"]
        ),
        "estaciones_sin_catalogo": (
            observaciones["sin_catalogo"]
        ),
        "capacidad_cero": (
            observaciones["capacidad_cero"]
        ),
    }

    for nombre, mascara in mascaras.items():

        segmentos.append(
            resumir_grupo(
                nombre,
                "2026-01/2026-10",
                observaciones.loc[mascara],
            )
        )

    tabla = pd.DataFrame(mensual)

    assert tabla["periodo"].tolist() == MESES
    assert tabla["observaciones"].sum() == len(x)

    tabla.to_csv(
        SALIDA / "desempeno_mensual_2026.csv",
        index=False,
    )

    pd.DataFrame(segmentos).to_csv(
        SALIDA / "impacto_por_segmento_2026.csv",
        index=False,
    )

    rango = {
        "minimo_train": minimo,
        "maximo_train": maximo,
        "minimo_actual": float(
            x["dias_activa"].min()
        ),
        "maximo_actual": float(
            x["dias_activa"].max()
        ),
        "mediana_actual": float(
            x["dias_activa"].median()
        ),
        "fuera_train": int(
            (soporte != "dentro_train").sum()
        ),
        "fuera_train_pct": float(
            100 * (soporte != "dentro_train").mean()
        ),
    }

    resultado = {
        "modelo": MODEL_NAME,
        "version": MODEL_VERSION,
        "run_id": RUN_ID,
        "logged_model_id": MODEL_ID,
        "data_version_historica": DATA_VERSION,
        "data_version_evaluacion_2026": json.loads(
            (SALIDA / "manifest_2026.json").read_text()
        )["data_version_evaluacion_2026"],
        "evaluacion": "retrospectiva_temporal_externa",
        "entrenamiento_realizado": False,
        "promocion_realizada": False,
        "mae_test_2025_reproducido": mae_test,
        "metricas_globales": globales,
        "soporte_dias_activa": rango,
        "sensibilidad_modelo": obtener_coeficiente(
            modelo
        ),
        "nota": (
            "Evaluación diaria con rezagos observados "
            "hasta D-1. No es un pronóstico multihorizonte "
            "emitido al comienzo de 2026."
        ),
    }

    guardar_json(
        SALIDA / "evaluacion_externa_2026.json",
        resultado,
    )

    # Intercambio local entre los dos entornos Python.
    # Solo se exportan características y objetivo.
    referencia = train_x.copy()
    referencia["viajes_real"] = train_y

    referencia.reset_index().to_parquet(
        BASE_LOCAL / "referencia_train.parquet",
        index=False,
    )

    datos_actuales = x.copy()
    datos_actuales["viajes_real"] = y

    datos_actuales.reset_index().to_parquet(
        BASE_LOCAL / "actual_2026.parquet",
        index=False,
    )

    print("\n========== EVALUACION EXTERNA ==========")

    print(
        tabla[
            [
                "periodo",
                "observaciones",
                "mae",
                "wape",
                "sesgo",
                "mae_baseline_lag7",
                "fuera_train_pct",
                "mae_variacion_vs_test_2025_pct",
            ]
        ].to_string(index=False)
    )

    print("\nResultados globales:")
    print(
        json.dumps(
            resultado,
            indent=2,
            ensure_ascii=False,
        )
    )

    return tabla


def graficas(tabla):

    fig, ax = plt.subplots(figsize=(10, 4.8))

    ax.plot(
        tabla["periodo"],
        tabla["mae"],
        marker="o",
        label="Champion P1 v1",
    )

    ax.plot(
        tabla["periodo"],
        tabla["mae_baseline_lag7"],
        marker="s",
        linestyle="--",
        label="Baseline semanal",
    )

    ax.axhline(
        MAE_TEST_ORIGINAL,
        linestyle=":",
        label="MAE TEST 2025",
    )

    ax.set_title("P1: desempeño temporal externo 2026")
    ax.set_ylabel("MAE (viajes por estación-día)")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        SALIDA / "01_desempeno_temporal_2026.png",
        dpi=180,
    )

    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.bar(
        tabla["periodo"],
        tabla["fuera_train_pct"],
    )

    ax.set_title(
        "P1: observaciones de dias_activa fuera de TRAIN"
    )

    ax.set_ylabel("Observaciones fuera de rango (%)")
    ax.set_ylim(0, 105)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        SALIDA / "02_extrapolacion_2026.png",
        dpi=180,
    )

    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.bar(
        tabla["periodo"],
        tabla["sesgo"],
    )

    ax.axhline(0, color="black", linewidth=1)

    ax.set_title("P1: sesgo mensual de las predicciones")
    ax.set_ylabel("Predicción - valor observado")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        SALIDA / "03_sesgo_mensual_2026.png",
        dpi=180,
    )

    plt.close(fig)


def main():

    SALIDA.mkdir(parents=True, exist_ok=True)

    actual, _ = cargar_actual()

    panel, originales, capacidad, estaciones = (
        preparar_panel(actual)
    )

    tabla = evaluar(
        panel,
        originales,
        capacidad,
        estaciones,
    )

    graficas(tabla)

    print("\nEVALUACION TEMPORAL COMPLETADA")
    print("Particiones históricas preservadas.")
    print("Modelo champion sin modificaciones.")


if __name__ == "__main__":
    main()
