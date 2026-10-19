"""P1: regresión de viajes iniciados por estación para el día siguiente."""

import json
from pathlib import Path
import subprocess

from loguru import logger
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_poisson_deviance,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import typer

from mibici.config import (
    MANIFEST_PATH,
    P1_MAX_DURATION_SECONDS,
    P1_MIN_DURATION_SECONDS,
    P1_MLFLOW_BASELINE_RUN_NAME,
    P1_MLFLOW_CANDIDATE_RUN_NAME,
    P1_MLFLOW_EXPERIMENT,
    P1_MLFLOW_FINAL_RUN_NAME,
    P1_OUTPUT_DIR,
    P1_QUESTION,
    P1_TEST_END,
    P1_TEST_START,
    P1_TRAIN_START,
    P1_VAL_START,
    PROJ_ROOT,
    SEED,
    VIAJES_PARQUET,
)
from mibici.features import (
    P1_FEATURES,
    p1_agregar_features_y_target,
    p1_construir_panel_diario,
)

app = typer.Typer()

SKOPS_TRUSTED = [
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"
]


def obtener_git_commit() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJ_ROOT,
            text=True,
        ).strip()
        estado = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=PROJ_ROOT,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "desconocido", True

    return commit, bool(estado)


def cargar_viajes(ruta: Path) -> pd.DataFrame:
    viajes = pd.read_parquet(
        ruta,
        columns=[
            "inicio_viaje",
            "fin_viaje",
            "origen_id",
            "destino_id",
        ],
    )

    duracion = (
        viajes["fin_viaje"] - viajes["inicio_viaje"]
    ).dt.total_seconds()

    return viajes.loc[
        duracion.between(
            P1_MIN_DURATION_SECONDS,
            P1_MAX_DURATION_SECONDS,
        )
    ].dropna()


def construir_baseline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "modelo",
                PoissonRegressor(
                    alpha=0.1,
                    max_iter=500,
                ),
            ),
        ]
    )


def construir_candidato() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="poisson",
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=SEED,
    )


def metricas_regresion(y, pred, prefijo: str) -> dict:
    y = np.asarray(y, dtype="float64")
    pred = np.clip(
        np.asarray(pred, dtype="float64"),
        1e-9,
        None,
    )

    return {
        f"{prefijo}_mae": float(mean_absolute_error(y, pred)),
        f"{prefijo}_rmse": float(
            np.sqrt(mean_squared_error(y, pred))
        ),
        f"{prefijo}_r2": float(r2_score(y, pred)),
        f"{prefijo}_wape": float(
            np.abs(y - pred).sum() / max(y.sum(), 1.0)
        ),
        f"{prefijo}_poisson_deviance": float(
            mean_poisson_deviance(y, pred)
        ),
        f"{prefijo}_media_real": float(y.mean()),
        f"{prefijo}_media_pred": float(pred.mean()),
    }


def preparar_xy(df: pd.DataFrame):
    x = df[P1_FEATURES].astype("float64")
    y = df["target_salidas"].astype("float64")

    meta = df.reset_index()[
        ["estacion", "fecha_objetivo"]
    ]

    return x, y, meta


def guardar_artefactos(
    nombre: str,
    meta: pd.DataFrame,
    y,
    pred,
    particion: str,
) -> list[Path]:
    salida = P1_OUTPUT_DIR / nombre
    salida.mkdir(parents=True, exist_ok=True)

    pred = np.clip(
        np.asarray(pred, dtype="float64"),
        0,
        None,
    )

    tabla = meta.copy()
    tabla["real"] = np.asarray(y, dtype="float64")
    tabla["prediccion"] = pred
    tabla["residuo"] = tabla["real"] - tabla["prediccion"]

    ruta_pred = salida / f"{particion}_predicciones.csv"
    tabla.to_csv(ruta_pred, index=False)

    diario = (
        tabla.groupby("fecha_objetivo", as_index=False)[
            ["real", "prediccion"]
        ]
        .sum()
        .sort_values("fecha_objetivo")
    )

    ruta_serie = salida / f"{particion}_demanda_total.png"

    plt.figure(figsize=(10, 4.5))
    plt.plot(
        diario["fecha_objetivo"],
        diario["real"],
        label="Real",
    )
    plt.plot(
        diario["fecha_objetivo"],
        diario["prediccion"],
        label="Predicción",
    )
    plt.title(
        f"{nombre} · demanda diaria total · {particion.upper()}"
    )
    plt.xlabel("Fecha objetivo")
    plt.ylabel("Viajes iniciados")
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(
        ruta_serie,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()

    ruta_residuos = salida / f"{particion}_residuos.png"

    plt.figure(figsize=(7, 4.5))
    plt.hist(tabla["residuo"], bins=50)
    plt.title(
        f"{nombre} · residuos · {particion.upper()}"
    )
    plt.xlabel("Real - predicción")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(
        ruta_residuos,
        dpi=160,
        bbox_inches="tight",
    )
    plt.close()

    return [
        ruta_pred,
        ruta_serie,
        ruta_residuos,
    ]


def registrar_run(
    nombre_run,
    etapa,
    modelo,
    train,
    evaluacion,
    nombre_evaluacion,
    contexto,
) -> dict:
    x_train, y_train, _ = preparar_xy(train)
    x_eval, y_eval, meta_eval = preparar_xy(evaluacion)

    modelo.fit(x_train, y_train)

    pred_train = modelo.predict(x_train)
    pred_eval = modelo.predict(x_eval)

    metricas = {
        **metricas_regresion(
            y_train,
            pred_train,
            "train",
        ),
        **metricas_regresion(
            y_eval,
            pred_eval,
            nombre_evaluacion,
        ),
    }

    artefactos = guardar_artefactos(
        nombre_run,
        meta_eval,
        y_eval,
        pred_eval,
        nombre_evaluacion,
    )

    with mlflow.start_run(run_name=nombre_run):
        mlflow.set_tags(
            {
                "problema": "P1",
                "tipo": "regresion",
                "etapa": etapa,
                "pregunta": P1_QUESTION,
                "git_commit": contexto["git_commit"],
                "git_sucio": str(contexto["git_sucio"]),
                "data_version": contexto["data_version"],
            }
        )

        mlflow.log_params(
            {
                "modelo": contexto.get(
                    "modelo_nombre",
                    type(modelo).__name__,
                ),
                "unidad_analisis": "estacion-dia",
                "target": (
                    "viajes iniciados en la estacion "
                    "el dia siguiente"
                ),
                "horizonte_dias": 1,
                "features": len(P1_FEATURES),
                "train_inicio": P1_TRAIN_START,
                "val_inicio": P1_VAL_START,
                "test_inicio": P1_TEST_START,
                "test_fin": P1_TEST_END,
                "random_state": SEED,
                "data_version": contexto["data_version"],
                **contexto["params_datos"],
            }
        )

        mlflow.log_metrics(metricas)

        mlflow.sklearn.log_model(
            modelo,
            name="modelo",
            input_example=x_eval.head(5),
            skops_trusted_types=SKOPS_TRUSTED,
        )

        for ruta in artefactos:
            mlflow.log_artifact(
                str(ruta),
                artifact_path="evaluacion",
            )

    logger.info(
        f"{nombre_run}: "
        f"{nombre_evaluacion}_mae="
        f"{metricas[f'{nombre_evaluacion}_mae']:.3f} "
        f"{nombre_evaluacion}_rmse="
        f"{metricas[f'{nombre_evaluacion}_rmse']:.3f} "
        f"{nombre_evaluacion}_r2="
        f"{metricas[f'{nombre_evaluacion}_r2']:.3f}"
    )

    return metricas


@app.command()
def main(
    viajes_path: Path = VIAJES_PARQUET,
    experiment_name: str = P1_MLFLOW_EXPERIMENT,
):
    logger.info("Iniciando pipeline P1")

    P1_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    viajes = cargar_viajes(viajes_path)

    panel = p1_construir_panel_diario(
        viajes,
        P1_TRAIN_START,
        P1_TEST_END,
    )

    df = p1_agregar_features_y_target(panel)

    fecha_obj = df["fecha_objetivo"]

    desarrollo = df.loc[
        (fecha_obj < pd.Timestamp(P1_TEST_START))
    ].copy()

    test = df.loc[
        (fecha_obj >= pd.Timestamp(P1_TEST_START))
        & (fecha_obj < pd.Timestamp(P1_TEST_END))
    ].copy()

    train = desarrollo.loc[
        desarrollo["fecha_objetivo"]
        < pd.Timestamp(P1_VAL_START)
    ].copy()

    val = desarrollo.loc[
        desarrollo["fecha_objetivo"]
        >= pd.Timestamp(P1_VAL_START)
    ].copy()

    estaciones_train = (
        train.index.get_level_values("estacion").unique()
    )

    val = val.loc[
        val.index.get_level_values("estacion").isin(
            estaciones_train
        )
    ]

    estaciones_desarrollo = (
        desarrollo.index.get_level_values("estacion").unique()
    )

    test = test.loc[
        test.index.get_level_values("estacion").isin(
            estaciones_desarrollo
        )
    ]

    antes = {
        "train": len(train),
        "val": len(val),
        "desarrollo": len(desarrollo),
        "test": len(test),
    }

    train, val, desarrollo, test = (
        d.dropna(
            subset=P1_FEATURES + ["target_salidas"]
        )
        for d in (
            train,
            val,
            desarrollo,
            test,
        )
    )

    for nombre, d in (
        ("train", train),
        ("val", val),
        ("desarrollo", desarrollo),
        ("test", test),
    ):
        logger.info(
            f"{nombre}: filas={len(d):,} "
            f"(descartadas por lags="
            f"{antes[nombre] - len(d):,}) "
            f"estaciones="
            f"{d.index.get_level_values('estacion').nunique()} "
            f"media_target="
            f"{d['target_salidas'].mean():.2f}"
        )

    logger.info(
        "Partición temporal externa: "
        "desarrollo=292 días (80.00%) | "
        "test=73 días (20.00%)"
    )

    git_commit, git_sucio = obtener_git_commit()

    if git_sucio:
        logger.warning(
            "Hay cambios sin commit: "
            "el run no será reproducible desde git_commit"
        )

    contexto = {
        "git_commit": git_commit,
        "git_sucio": git_sucio,
        "data_version": json.loads(
            MANIFEST_PATH.read_text()
        )["data_version"],
        "params_datos": {
            "filas_train": len(train),
            "filas_val": len(val),
            "filas_desarrollo": len(desarrollo),
            "filas_test": len(test),
            "estaciones_train": (
                train.index
                .get_level_values("estacion")
                .nunique()
            ),
            "estaciones_test": (
                test.index
                .get_level_values("estacion")
                .nunique()
            ),
        },
    }

    mlflow.set_tracking_uri(
        f"sqlite:///{PROJ_ROOT / 'mlflow.db'}"
    )
    mlflow.set_experiment(experiment_name)

    contexto["modelo_nombre"] = "PoissonRegressor"

    metricas_baseline = registrar_run(
        P1_MLFLOW_BASELINE_RUN_NAME,
        "baseline",
        construir_baseline(),
        train,
        val,
        "val",
        contexto,
    )

    contexto["modelo_nombre"] = (
        "HistGradientBoostingRegressor"
    )

    metricas_candidato = registrar_run(
        P1_MLFLOW_CANDIDATE_RUN_NAME,
        "candidato",
        construir_candidato(),
        train,
        val,
        "val",
        contexto,
    )

    if (
        metricas_candidato["val_mae"]
        <= metricas_baseline["val_mae"]
    ):
        constructor_final = construir_candidato
        seleccionado = "HistGradientBoostingRegressor"
    else:
        constructor_final = construir_baseline
        seleccionado = "PoissonRegressor"

    logger.info(
        "Modelo seleccionado por menor MAE "
        f"en validación: {seleccionado}"
    )

    contexto["modelo_nombre"] = seleccionado
    contexto["params_datos"][
        "modelo_seleccionado"
    ] = seleccionado

    metricas_final = registrar_run(
        P1_MLFLOW_FINAL_RUN_NAME,
        "final",
        constructor_final(),
        desarrollo,
        test,
        "test",
        contexto,
    )

    logger.success(
        "Pipeline P1 terminado: "
        f"test_mae={metricas_final['test_mae']:.3f} "
        f"test_rmse={metricas_final['test_rmse']:.3f} "
        f"test_r2={metricas_final['test_r2']:.3f}"
    )


if __name__ == "__main__":
    app()
