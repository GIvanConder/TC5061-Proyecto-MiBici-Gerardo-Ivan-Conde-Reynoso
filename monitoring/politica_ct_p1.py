"""Política ejecutable de monitoreo P1.

Genera alertas y solicitudes de evaluación.
No entrena modelos ni escribe en MLflow.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mibici.features import P1_FEATURES

from monitoring.config_2026 import BASE_LOCAL, SALIDA


CONFIG = Path("monitoring/politica_p1.json")


def cargar_politica():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def guardar_json(ruta, contenido):
    ruta.write_text(
        json.dumps(
            contenido,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )


def validar_panel(datos):
    """Controles bloqueantes sobre la unidad estación-día."""

    requeridas = {
        "estacion",
        "fecha",
        "viajes_real",
        *P1_FEATURES,
    }

    faltantes = requeridas - set(datos.columns)

    if faltantes:
        raise ValueError(f"Columnas faltantes: {faltantes}")

    if datos.empty:
        raise ValueError("Panel vacío")

    df = datos.copy()

    df["fecha"] = pd.to_datetime(
        df["fecha"],
        errors="raise",
    )

    if df[["estacion", "fecha"]].isna().any().any():
        raise ValueError("Claves estación-fecha nulas")

    if df.duplicated(["estacion", "fecha"]).any():
        raise ValueError("Claves estación-fecha duplicadas")

    x = df[P1_FEATURES].to_numpy(dtype=float)
    y = df["viajes_real"].to_numpy(dtype=float)

    if not np.isfinite(x).all():
        raise ValueError("Características no finitas")

    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Objetivos inválidos")

    return df


def evaluar_meses(
    mensual,
    mae_referencia,
    incremento=0.15,
    consecutivos=2,
):
    """Aplica una regla de persistencia sobre meses completos.

    Los datos de entrada deben contener meses consecutivos.
    """

    if not np.isfinite(mae_referencia) or mae_referencia <= 0:
        raise ValueError("Referencia MAE inválida")

    if not np.isfinite(incremento) or incremento <= 0:
        raise ValueError("Incremento inválido")

    if consecutivos < 1 or int(consecutivos) != consecutivos:
        raise ValueError("Persistencia inválida")

    tabla = mensual.copy()

    if tabla.empty:
        raise ValueError("No hay meses evaluables")

    periodos = pd.PeriodIndex(
        tabla["periodo"],
        freq="M",
    )

    if periodos.hasnans or not periodos.is_unique:
        raise ValueError("Periodos duplicados o inválidos")

    tabla["_periodo"] = periodos

    tabla = tabla.sort_values("_periodo").reset_index(drop=True)

    periodos = pd.PeriodIndex(
        tabla["_periodo"],
        freq="M",
    )

    esperados = pd.period_range(
        periodos.min(),
        periodos.max(),
        freq="M",
    )

    if not periodos.equals(esperados):
        raise ValueError("Existen meses faltantes")

    maes = tabla["mae"].to_numpy(dtype=float)

    if not np.isfinite(maes).all() or (maes < 0).any():
        raise ValueError("MAE mensual inválido")

    umbral = mae_referencia * (1 + incremento)

    racha = 0
    filas = []

    for _, fila in tabla.iterrows():

        mae = float(fila["mae"])

        alerta = mae >= umbral

        racha = racha + 1 if alerta else 0

        solicitud_nueva = racha == consecutivos

        if racha >= consecutivos:
            estado = "evaluacion_reentrenamiento_requerida"
        elif alerta:
            estado = "alerta_desempeno"
        else:
            estado = "monitoreo_habitual"

        filas.append({
            "periodo": str(fila["_periodo"]),
            "mae": mae,
            "mae_referencia": float(mae_referencia),
            "umbral_mae": float(umbral),
            "incremento_mae_pct": float(
                100 * (mae / mae_referencia - 1)
            ),
            "alerta_desempeno": bool(alerta),
            "meses_consecutivos_alerta": racha,
            "estado": estado,
            "nueva_solicitud_evaluacion": solicitud_nueva,
            "promocion_autorizada": False,
        })

    return pd.DataFrame(filas)


def evaluar_semanas(datos, minimo, maximo, umbrales):
    """Monitoreo semanal de calidad y extrapolación."""

    df = validar_panel(datos)

    periodos = df["fecha"].dt.to_period("W-SUN")

    resultados = []

    for periodo, grupo in df.groupby(periodos, sort=True):

        dias = grupo["fecha"].dt.normalize().nunique()

        fuera = (
            grupo["dias_activa"].lt(minimo)
            | grupo["dias_activa"].gt(maximo)
        )

        proporcion = float(fuera.mean())

        capacidad_invalida = int(
            grupo["capacidad"].le(0).sum()
        )

        completa = dias == 7

        resultados.append({
            "inicio_semana": str(periodo.start_time.date()),
            "fin_semana": str(periodo.end_time.date()),
            "dias_observados": int(dias),
            "semana_completa": completa,
            "observaciones": len(grupo),
            "estaciones": int(grupo["estacion"].nunique()),
            "fuera_train": int(fuera.sum()),
            "fuera_train_pct": 100 * proporcion,
            "filas_capacidad_no_positiva": capacidad_invalida,
            "alerta_extrapolacion": (
                proporcion >= umbrales["fuera_soporte_fraccion"]
            ) if completa else None,
            "alerta_catalogo": (
                capacidad_invalida
                >= umbrales["capacidad_no_positiva_alerta"]
            ),
            "duplicados_clave_estacion_fecha": 0,
        })

    return pd.DataFrame(resultados)


def main():

    SALIDA.mkdir(parents=True, exist_ok=True)

    politica = cargar_politica()

    umbrales = politica["umbrales"]

    evaluacion = json.loads(
        (SALIDA / "evaluacion_externa_2026.json").read_text()
    )

    referencia = politica["referencia_desempeno"]["valor"]

    assert evaluacion["modelo"] == politica["modelo"]

    assert (
        evaluacion["version"]
        == politica["version_referencia"]
    )

    assert np.isclose(
        evaluacion["mae_test_2025_reproducido"],
        referencia,
        atol=1e-4,
        rtol=0,
    )

    soporte = evaluacion["soporte_dias_activa"]

    actual = validar_panel(
        pd.read_parquet(BASE_LOCAL / "actual_2026.parquet")
    )

    assert len(actual) == (
        evaluacion["metricas_globales"]["observaciones"]
    )

    assert actual["fecha"].dt.normalize().nunique() == 304

    # Evidently recibe solo características, no claves.
    duplicados_features = actual.duplicated(
        subset=P1_FEATURES
    )

    repetidos = actual.duplicated(
        subset=P1_FEATURES,
        keep=False,
    )

    grupos_repetidos = (
        actual.loc[repetidos, P1_FEATURES]
        .drop_duplicates()
    )

    calidad_panel = {
        "observaciones": len(actual),
        "claves_estacion_fecha_duplicadas": 0,
        "copias_excedentes_vectores_features": int(
            duplicados_features.sum()
        ),
        "filas_con_vectores_features_repetidos": int(
            repetidos.sum()
        ),
        "grupos_features_repetidos": len(grupos_repetidos),
        "estaciones_involucradas": int(
            actual.loc[repetidos, "estacion"].nunique()
        ),
        "filas_eliminadas": 0,
        "decision": (
            "Conservar observaciones estación-día únicas. "
            "La igualdad de las 12 características no demuestra "
            "que dos observaciones correspondan al mismo evento."
        ),
    }

    guardar_json(
        SALIDA / "diagnostico_features_repetidas_2026.json",
        calidad_panel,
    )

    semanal = evaluar_semanas(
        actual,
        soporte["minimo_train"],
        soporte["maximo_train"],
        umbrales,
    )

    semanal.to_csv(
        SALIDA / "monitoreo_semanal_2026.csv",
        index=False,
    )

    mensual_original = pd.read_csv(
        SALIDA / "desempeno_mensual_2026.csv"
    )

    mensual = evaluar_meses(
        mensual_original,
        referencia,
        umbrales["incremento_mae_fraccion"],
        umbrales["meses_consecutivos"],
    )

    mensual.to_csv(
        SALIDA / "politica_ct_mensual_2026.csv",
        index=False,
    )

    drift = pd.read_csv(
        SALIDA / "resumen_drift_variables_2026.csv"
    )

    assert drift["variable"].is_unique
    assert set(drift["variable"]) == set(P1_FEATURES)

    assert np.isfinite(
        drift["valor_drift"].to_numpy(dtype=float)
    ).all()

    # Los métodos utilizados son distancias:
    # Wasserstein normalizada y Jensen-Shannon.
    metodos_validos = drift["metodo"].str.contains(
        "Wasserstein|Jensen-Shannon",
        regex=True,
    )

    assert metodos_validos.all()

    assert np.allclose(
        drift["umbral"],
        umbrales["drift_individual_distancia"],
    )

    alertas_drift = (
        drift["valor_drift"] >= drift["umbral"]
    )

    assert alertas_drift.eq(
        drift["drift_detectado"]
    ).all()

    proporcion_drift = float(alertas_drift.mean())

    solicitudes = mensual.loc[
        mensual["nueva_solicitud_evaluacion"],
        "periodo",
    ].tolist()

    meses_alerta = mensual.loc[
        mensual["alerta_desempeno"],
        "periodo",
    ].tolist()

    semanas_completas = semanal.loc[
        semanal["semana_completa"]
    ]

    resultado = {
        "politica_version": politica["politica_version"],
        "modelo": politica["modelo"],
        "version_modelo": politica["version_referencia"],
        "data_version_evaluacion": (
            evaluacion["data_version_evaluacion_2026"]
        ),
        "tipo": "aplicacion_retrospectiva_de_politica_inicial",
        "validacion_prospectiva": False,
        "umbral_mae": float(
            referencia * (
                1 + umbrales["incremento_mae_fraccion"]
            )
        ),
        "meses_con_alerta_desempeno": meses_alerta,
        "solicitudes_evaluacion_reentrenamiento": solicitudes,
        "semanas_completas": len(semanas_completas),
        "semanas_parciales": (
            len(semanal) - len(semanas_completas)
        ),
        "semanas_completas_alerta_extrapolacion": int(
            semanas_completas["alerta_extrapolacion"]
            .eq(True).sum()
        ),
        "semanas_alerta_catalogo": int(
            semanal["alerta_catalogo"].sum()
        ),
        "variables_con_drift": int(alertas_drift.sum()),
        "proporcion_variables_drift": proporcion_drift,
        "alerta_drift_global": (
            proporcion_drift
            >= umbrales["drift_global_fraccion"]
        ),
        "alcance_evidently": (
            "Evaluación agregada enero-octubre 2026; "
            "frecuencia semanal propuesta, no automatizada."
        ),
        "entrenamiento_ejecutado": False,
        "promocion_ejecutada": False,
        "decision": (
            "Conservar champion P1 v1 y solicitar evaluación "
            "supervisada de un candidato cuando corresponda."
        ),
        "limitacion": (
            "Umbrales iniciales definidos después de conocer "
            "la evaluación externa. La referencia de TEST 2025 "
            "no controla completamente estacionalidad ni expansión."
        ),
    }

    guardar_json(
        SALIDA / "resultado_politica_ct_2026.json",
        resultado,
    )

    print("\n========== CALIDAD DEL PANEL ==========")
    print(json.dumps(calidad_panel, indent=2))

    print("\n========== DIAGNOSTICO SEMANAL ==========")
    print(semanal.to_string(index=False))

    print("\n========== POLITICA MENSUAL ==========")
    print(mensual.to_string(index=False))

    print("\n========== DECISION OPERATIVA ==========")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))

    print("\nNo se entrenaron ni promovieron modelos.")


if __name__ == "__main__":
    main()
