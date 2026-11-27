"""Diagnóstico descriptivo del impacto predictivo de dias_activa.

Recupera el champion histórico sin entrenar, registrar ni promover
modelos. Conserva las particiones originales de P1.
"""

import hashlib
import json
from pathlib import Path

import mlflow.sklearn
import numpy as np
import pandas as pd

from mibici.features import P1_FEATURES
from mibici.modeling.p1_regresion import preparar_datos_p1


RUN_ID = "371bf2379bfa4a74b8effbe02d2ee3ae"
MODEL_ID = "m-f1e2eda1d16a4663a7ec1f50d75859d9"
MODEL_NAME = "mibici-equipo32-p1-demanda"
MODEL_VERSION = 1
DATA_VERSION = "7aa5e436d887"

MODELO = (
    Path("mlartifacts/1/models")
    / MODEL_ID
    / "artifacts"
)

SALIDA = Path("docs/evidencias/monitoreo_final")

VARIABLE = "dias_activa"

MAE_TEST_ORIGINAL = 5.409785669092729

FILAS_HISTORICAS = {
    "train": 90230,
    "val": 11439,
    "test": 22565,
}


def metricas(y, pred):

    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)

    if y.shape != pred.shape:
        raise ValueError("Dimensiones incompatibles")

    if not np.isfinite(y).all():
        raise ValueError("Objetivos no finitos")

    if not np.isfinite(pred).all():
        raise ValueError("Predicciones no finitas")

    if (y < 0).any() or (pred < 0).any():
        raise ValueError("Conteos negativos")

    if len(y) == 0:
        return {
            "mae": None,
            "rmse": None,
            "wape": None,
            "sesgo": None,
        }

    error = pred - y

    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "wape": (
            float(np.abs(error).sum() / y.sum())
            if y.sum() > 0 else None
        ),
        "sesgo": float(error.mean()),
    }


def clasificar_soporte(serie, minimo, maximo):

    return np.select(
        [
            serie < minimo,
            serie > maximo,
        ],
        [
            "inferior_train",
            "superior_train",
        ],
        default="dentro_train",
    )


def cargar_champion():

    manifiesto = json.loads(
        Path("data/manifest.json").read_text()
    )

    assert manifiesto["data_version"] == DATA_VERSION

    assert (MODELO / "MLmodel").is_file()

    # Artefacto local original generado por el proyecto.
    # No se consulta el MLflow aislado de Compose.
    modelo = mlflow.sklearn.load_model(
        str(MODELO.resolve())
    )

    assert list(modelo.feature_names_in_) == P1_FEATURES

    return modelo


def obtener_coeficiente(modelo):

    prep = modelo.named_steps["prep"]
    estimador = modelo.named_steps["modelo"]

    columnas = next(
        columnas
        for nombre, _, columnas in prep.transformers_
        if nombre == "num"
    )

    posicion = list(columnas).index(VARIABLE)

    escalador = prep.named_transformers_["num"]

    inicio = prep.output_indices_["num"].start

    coeficiente = float(
        estimador.coef_[inicio + posicion]
    )

    escala = float(escalador.scale_[posicion])

    assert escala > 0

    coef_dia = coeficiente / escala

    return {
        "coeficiente_estandarizado": coeficiente,
        "media_scaler": float(
            escalador.mean_[posicion]
        ),
        "escala_scaler": escala,
        "coeficiente_por_dia": coef_dia,
        "factor_prediccion_30_dias": float(
            np.exp(coef_dia * 30)
        ),
        "variacion_prediccion_30_dias_pct": float(
            100 * np.expm1(coef_dia * 30)
        ),
        "factor_prediccion_365_dias": float(
            np.exp(coef_dia * 365)
        ),
    }


def resumir_grupo(nombre, periodo, grupo):

    n = len(grupo)

    resultado = {
        "periodo": periodo,
        "segmento": nombre,
        "observaciones": n,
        "estaciones": int(
            grupo["estacion"].nunique()
        ),
        "dias_activa_min": (
            float(grupo[VARIABLE].min())
            if n else None
        ),
        "dias_activa_max": (
            float(grupo[VARIABLE].max())
            if n else None
        ),
        "demanda_real_media": (
            float(grupo["real"].mean())
            if n else None
        ),
        "prediccion_media": (
            float(grupo["pred"].mean())
            if n else None
        ),
        **metricas(grupo["real"], grupo["pred"]),
        "mae_baseline_lag7": metricas(
            grupo["real"], grupo["lag_7"]
        )["mae"],
        "mae_sensibilidad_recorte": metricas(
            grupo["real"], grupo["pred_recorte"]
        )["mae"],
        "cambio_absoluto_prediccion_medio": (
            float(
                np.abs(
                    grupo["pred_recorte"] - grupo["pred"]
                ).mean()
            )
            if n else None
        ),
    }

    return resultado


def main():

    SALIDA.mkdir(parents=True, exist_ok=True)

    modelo = cargar_champion()

    particiones, _, _ = preparar_datos_p1()

    datos = {}

    for nombre, (x, y) in zip(
        ("train", "val", "test"),
        particiones[:3],
    ):

        assert len(x) == FILAS_HISTORICAS[nombre]
        assert x.index.equals(y.index)
        assert list(x.columns) == P1_FEATURES
        assert np.isfinite(x.to_numpy()).all()

        datos[nombre] = (x, y)

    train = datos["train"][0]

    minimo = float(train[VARIABLE].min())
    maximo = float(train[VARIABLE].max())

    filas_rangos = []
    filas_metricas = []
    filas_mensuales = []

    for nombre, (x, y) in datos.items():

        pred = np.asarray(
            modelo.predict(x),
            dtype=float,
        )

        x_recorte = x.copy()

        x_recorte[VARIABLE] = (
            x_recorte[VARIABLE].clip(
                lower=minimo,
                upper=maximo,
            )
        )

        pred_recorte = np.asarray(
            modelo.predict(x_recorte),
            dtype=float,
        )

        soporte = clasificar_soporte(
            x[VARIABLE],
            minimo,
            maximo,
        )

        observaciones = pd.DataFrame(
            {
                "real": y.to_numpy(dtype=float),
                "pred": pred,
                "pred_recorte": pred_recorte,
                "lag_7": x["lag_7"].to_numpy(),
                VARIABLE: x[VARIABLE].to_numpy(),
                "soporte": soporte,
            },
            index=x.index,
        ).reset_index()

        observaciones["fecha"] = pd.to_datetime(
            observaciones["fecha"]
        )

        fuera = soporte != "dentro_train"

        filas_rangos.append({
            "particion": nombre,
            "observaciones": len(x),
            "minimo": float(x[VARIABLE].min()),
            "p05": float(x[VARIABLE].quantile(0.05)),
            "mediana": float(x[VARIABLE].median()),
            "p95": float(x[VARIABLE].quantile(0.95)),
            "maximo": float(x[VARIABLE].max()),
            "inferior_train": int(
                (soporte == "inferior_train").sum()
            ),
            "superior_train": int(
                (soporte == "superior_train").sum()
            ),
            "fuera_train": int(fuera.sum()),
            "fuera_train_pct": float(
                100 * fuera.mean()
            ),
        })

        filas_metricas.append(
            resumir_grupo(
                "total",
                nombre,
                observaciones,
            )
        )

        for segmento in [
            "dentro_train",
            "inferior_train",
            "superior_train",
        ]:

            grupo = observaciones.loc[
                observaciones["soporte"] == segmento
            ]

            filas_metricas.append(
                resumir_grupo(
                    segmento,
                    nombre,
                    grupo,
                )
            )

        if nombre == "test":

            mae = metricas(
                observaciones["real"],
                observaciones["pred"],
            )["mae"]

            assert abs(
                mae - MAE_TEST_ORIGINAL
            ) < 1e-4, (
                "No se reproduce el MAE historico"
            )

        for mes, grupo in observaciones.groupby(
            observaciones["fecha"].dt.to_period("M")
        ):

            fila = resumir_grupo(
                "total",
                str(mes),
                grupo,
            )

            fila["particion"] = nombre

            fila["fuera_train_pct"] = float(
                100 * (
                    grupo["soporte"] != "dentro_train"
                ).mean()
            )

            filas_mensuales.append(fila)

    coeficientes = obtener_coeficiente(modelo)

    rangos = pd.DataFrame(filas_rangos)
    resultados = pd.DataFrame(filas_metricas)
    mensual = pd.DataFrame(filas_mensuales)

    rangos.to_csv(
        SALIDA / "rangos_dias_activa_2025.csv",
        index=False,
    )

    resultados.to_csv(
        SALIDA / "impacto_predictivo_2025.csv",
        index=False,
    )

    mensual.to_csv(
        SALIDA / "impacto_mensual_2025.csv",
        index=False,
    )

    huellas = {}

    for archivo in sorted(MODELO.rglob("*")):

        if archivo.is_file():

            with archivo.open("rb") as f:

                huellas[
                    archivo.relative_to(MODELO).as_posix()
                ] = hashlib.file_digest(
                    f,
                    "sha256",
                ).hexdigest()

    resumen = {
        "modelo": MODEL_NAME,
        "version": MODEL_VERSION,
        "run_id": RUN_ID,
        "logged_model_id": MODEL_ID,
        "data_version": DATA_VERSION,
        "variable": VARIABLE,
        "minimo_train": minimo,
        "maximo_train": maximo,
        "sensibilidad_modelo": coeficientes,
        "huellas_artefacto": huellas,
        "mae_historico_reproducido": True,
        "entrenamiento_realizado": False,
        "modificaciones_registry": False,
        "metodologia": (
            "Diagnostico descriptivo y sensibilidad "
            "contrafactual de una sola variable. "
            "El recorte al rango TRAIN no es una "
            "correccion validada ni una politica "
            "de inferencia. Las diferencias entre "
            "segmentos no demuestran causalidad."
        ),
    }

    (SALIDA / "impacto_drift_p1.json").write_text(
        json.dumps(
            resumen,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
    )

    print("\n========== RANGO DE DIAS_ACTIVA ==========")

    print(rangos.to_string(index=False))

    print("\n========== IMPACTO PREDICTIVO ==========")

    print(
        resultados[
            [
                "periodo",
                "segmento",
                "observaciones",
                "mae",
                "rmse",
                "wape",
                "sesgo",
                "mae_baseline_lag7",
                "mae_sensibilidad_recorte",
            ]
        ].to_string(index=False)
    )

    print("\n========== RESULTADOS MENSUALES ==========")

    print(
        mensual[
            [
                "periodo",
                "observaciones",
                "fuera_train_pct",
                "mae",
                "wape",
                "sesgo",
            ]
        ].to_string(index=False)
    )

    print("\n========== SENSIBILIDAD DEL MODELO ==========")

    print(json.dumps(coeficientes, indent=2))

    print("\nMAE original de TEST reproducido.")
    print("Sin entrenamiento ni modificaciones de modelos.")
    print("Evidencias:", SALIDA)


if __name__ == "__main__":
    main()
