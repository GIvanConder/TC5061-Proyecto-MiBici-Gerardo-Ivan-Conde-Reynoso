"""P1: regresión de viajes iniciados por estación y día (Johanna Camargo).

Pregunta: ¿cuántos viajes se iniciarán en cada estación el día siguiente?

Pipeline reproducible de extremo a extremo:

1. Lee ``data/processed/viajes.parquet`` (salida de ``python -m mibici.dataset``)
   y descarta viajes de menos de 60 s o más de 24 h.
2. Construye el panel diario estación × fecha y las variables de
   ``mibici.features`` (rezagos >= 1 día, medias móviles, calendario).
3. Partición temporal: TRAIN ene-sep 2025, VAL oct 2025, TEST nov-dic 2025.
4. Entrena y registra en MLflow (experimento común ``mibici-equipo32``):
   baseline ingenuo semanal, candidato lineal Poisson y candidato
   HistGradientBoosting con pérdida Poisson.

Ejecución::

    python -m mibici.modeling.p1_regresion
"""

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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
import typer

from mibici.config import (
    ESTACIONES_PARQUET,
    MANIFEST_PATH,
    P1_BASELINE_LAG_DIAS,
    P1_FERIADOS,
    P1_HGB_PARAMS,
    P1_HORIZONTE_DIAS,
    P1_LAGS_DIAS,
    P1_LINEAL_ALPHA,
    P1_MAX_DURATION_SECONDS,
    P1_MIN_DURATION_SECONDS,
    P1_MLFLOW_BASELINE_RUN_NAME,
    P1_MLFLOW_CANDIDATE_RUN_NAME,
    P1_MLFLOW_EXPERIMENT,
    P1_MLFLOW_LINEAL_RUN_NAME,
    P1_OUTPUT_DIR,
    P1_QUESTION,
    P1_TEST_END,
    P1_TEST_START,
    P1_TRAIN_START,
    P1_VAL_START,
    P1_VENTANAS_MEDIA_DIAS,
    PROJ_ROOT,
    SEED,
    TEAM_ID,
    VIAJES_PARQUET,
)
from mibici.features import (
    P1_FEATURES,
    p1_agregar_features,
    p1_construir_panel_diario,
)

app = typer.Typer()

SKOPS_TRUSTED = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]
DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]


class BaselineNaiveSemanal:
    """Predice para el día D los viajes del mismo día de la semana anterior (``lag_7``)."""

    def __init__(self, columna: str):
        self.columna = columna

    def fit(self, x, y=None):
        return self

    def predict(self, x):
        return x[self.columna].to_numpy(dtype="float64")


def obtener_git_commit() -> tuple[str, bool]:
    """Devuelve el commit actual y si hay cambios sin confirmar."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT, text=True).strip()
        estado = subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJ_ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "desconocido", True
    return commit, bool(estado)


def cargar_viajes(ruta: Path) -> pd.DataFrame:
    """Lee los viajes procesados y aplica el filtro de duración de P1."""
    viajes = pd.read_parquet(ruta, columns=["inicio_viaje", "fin_viaje", "origen_id"])
    duracion = (viajes["fin_viaje"] - viajes["inicio_viaje"]).dt.total_seconds()
    return viajes.loc[duracion.between(P1_MIN_DURATION_SECONDS, P1_MAX_DURATION_SECONDS)].dropna()


def cargar_capacidad(ruta: Path) -> pd.Series:
    """Capacidad instalada (``dpcapacity``) por id de estación."""
    estaciones = pd.read_parquet(ruta, columns=["id", "dpcapacity"])
    return estaciones.set_index("id")["dpcapacity"]


def cortar(df: pd.DataFrame, inicio: str, fin: str) -> pd.DataFrame:
    """Filas con fecha en ``[inicio, fin)``."""
    fecha = df.index.get_level_values("fecha")
    return df.loc[(fecha >= pd.Timestamp(inicio)) & (fecha < pd.Timestamp(fin))]


def construir_lineal() -> Pipeline:
    """Candidato 1: regresión Poisson lineal (enlace log).

    Los conteos (rezagos y medias) entran como ``log1p`` para que el modelo sea
    multiplicativo y estable; el día de la semana entra en one-hot.
    """
    conteos = [c for c in P1_FEATURES if c.startswith(("lag_", "media_"))]
    otras = [c for c in P1_FEATURES if c not in conteos and c != "dia_semana"]
    return Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("dia", OneHotEncoder(handle_unknown="ignore"), ["dia_semana"]),
                        (
                            "conteos",
                            Pipeline([("log", FunctionTransformer(np.log1p)), ("esc", StandardScaler())]),
                            conteos,
                        ),
                        ("num", StandardScaler(), otras),
                    ]
                ),
            ),
            ("modelo", PoissonRegressor(alpha=P1_LINEAL_ALPHA, max_iter=1000)),
        ]
    )


def construir_hgb() -> HistGradientBoostingRegressor:
    """Candidato 2: gradient boosting con pérdida Poisson (conteos no negativos)."""
    return HistGradientBoostingRegressor(**P1_HGB_PARAMS, random_state=SEED)


def evaluar(y: pd.Series, pred: np.ndarray, prefijo: str) -> dict:
    """Métricas de regresión a nivel estación-día y de total diario del sistema."""
    pred = np.clip(pred, 0, None)
    total = pd.DataFrame({"y": y.to_numpy(), "p": pred}, index=y.index).groupby(level="fecha").sum()
    return {
        f"{prefijo}_mae": mean_absolute_error(y, pred),
        f"{prefijo}_rmse": float(np.sqrt(mean_squared_error(y, pred))),
        f"{prefijo}_wape": float(np.abs(y - pred).sum() / y.sum()),
        f"{prefijo}_r2": r2_score(y, pred),
        f"{prefijo}_sesgo": float((pred - y).mean()),
        f"{prefijo}_mape_total_dia": float((np.abs(total["p"] - total["y"]) / total["y"]).mean()),
    }


def guardar_graficas(nombre: str, test: pd.DataFrame, pred: np.ndarray, base: np.ndarray) -> list[Path]:
    """Serie del total diario en TEST y MAE por día de la semana (modelo vs. baseline)."""
    df = test[["viajes", "dia_semana"]].copy()
    df["pred"] = np.clip(pred, 0, None)
    df["base"] = base

    total = df.groupby(level="fecha")[["viajes", "pred", "base"]].sum()
    ruta_serie = P1_OUTPUT_DIR / f"{nombre}_total_diario_test.png"
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(total.index, total["viajes"], color="#1f2937", lw=2, label="Real")
    ax.plot(total.index, total["base"], color="#9ca3af", lw=1.5, ls="--", label="Baseline (t-7 días)")
    if nombre != P1_MLFLOW_BASELINE_RUN_NAME:
        ax.plot(total.index, total["pred"], color="#2563eb", lw=1.8, label=nombre)
    ax.set_title("P1 · Viajes totales por día en TEST (nov-dic 2025)")
    ax.set_ylabel("Viajes iniciados")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.savefig(ruta_serie, dpi=150, bbox_inches="tight")
    plt.close(fig)

    df["err_pred"] = (df["viajes"] - df["pred"]).abs()
    df["err_base"] = (df["viajes"] - df["base"]).abs()
    por_dia = df.groupby("dia_semana")[["err_base", "err_pred"]].mean()
    ruta_dia = P1_OUTPUT_DIR / f"{nombre}_mae_por_dia_test.png"
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(por_dia))
    ax.bar(x - 0.2, por_dia["err_base"], 0.4, color="#9ca3af", label="Baseline (t-7 días)")
    ax.bar(x + 0.2, por_dia["err_pred"], 0.4, color="#2563eb", label=nombre)
    ax.set_xticks(x, [DIAS[int(i)] for i in por_dia.index])
    ax.set_ylabel("MAE (viajes por estación-día)")
    ax.set_title("P1 · Error absoluto medio por día de la semana (TEST)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(ruta_dia, dpi=150, bbox_inches="tight")
    plt.close(fig)

    ruta_pred = P1_OUTPUT_DIR / f"{nombre}_predicciones_test.csv"
    df[["viajes", "pred"]].reset_index().to_csv(ruta_pred, index=False)
    return [ruta_serie, ruta_dia, ruta_pred]


def guardar_importancias(modelo, x_val: pd.DataFrame, y_val: pd.Series, nombre: str) -> list[Path]:
    """Importancia por permutación en VAL (muestra fija de 40 mil filas)."""
    muestra = x_val.sample(min(40_000, len(x_val)), random_state=SEED)
    res = permutation_importance(
        modelo, muestra, y_val.loc[muestra.index], scoring="neg_mean_absolute_error",
        n_repeats=3, random_state=SEED,
    )
    imp = pd.Series(res.importances_mean, index=x_val.columns).sort_values()
    ruta_csv = P1_OUTPUT_DIR / f"{nombre}_importancia_permutacion.csv"
    imp.rename("aumento_mae").to_csv(ruta_csv)
    ruta_png = P1_OUTPUT_DIR / f"{nombre}_importancia_permutacion.png"
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(imp.index, imp.values, color="#2563eb")
    ax.set_xlabel("Aumento del MAE al permutar la variable (VAL)")
    ax.set_title("P1 · Importancia de variables (HGB Poisson)")
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(ruta_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [ruta_png, ruta_csv]


def entrenar_y_registrar(nombre_run, etapa, modelo, datos, contexto, base_test) -> dict:
    """Ajusta en TRAIN, evalúa en TRAIN/VAL/TEST y registra todo en MLflow."""
    (x_train, y_train), (x_val, y_val), (x_test, y_test), test = datos

    modelo.fit(x_train, y_train)
    pred = {n: modelo.predict(x) for n, x in (("train", x_train), ("val", x_val), ("test", x_test))}
    metricas = {
        **evaluar(y_train, pred["train"], "train"),
        **evaluar(y_val, pred["val"], "val"),
        **evaluar(y_test, pred["test"], "test"),
    }
    artefactos = guardar_graficas(nombre_run, test, pred["test"], base_test)
    if isinstance(modelo, HistGradientBoostingRegressor):
        artefactos += guardar_importancias(modelo, x_val, y_val, nombre_run)
    ruta_metricas = P1_OUTPUT_DIR / f"{nombre_run}_metricas.json"
    ruta_metricas.write_text(json.dumps(metricas, indent=2))
    artefactos.append(ruta_metricas)

    if isinstance(modelo, BaselineNaiveSemanal):
        params_modelo = {"modelo": "NaiveEstacional", "lag_dias": P1_BASELINE_LAG_DIAS}
    elif isinstance(modelo, Pipeline):
        params_modelo = {
            "modelo": "PoissonRegressor",
            "alpha": P1_LINEAL_ALPHA,
            "transformacion_conteos": "log1p",
            "escalado": "StandardScaler",
        }
    else:
        params_modelo = {"modelo": "HistGradientBoostingRegressor", **P1_HGB_PARAMS}

    with mlflow.start_run(run_name=nombre_run):
        mlflow.set_tags(
            {
                "equipo": TEAM_ID,
                "mlflow.user": TEAM_ID,
                "problema": "P1",
                "tipo": "regresion",
                "etapa": etapa,
                "pregunta": P1_QUESTION,
                "responsable": "Johanna Camargo",
                "git_commit": contexto["git_commit"],
                "git_sucio": str(contexto["git_sucio"]),
                "data_version": contexto["data_version"],
            }
        )
        mlflow.log_params(
            {
                **params_modelo,
                "objetivo": "viajes iniciados por estacion y dia",
                "horizonte_dias": P1_HORIZONTE_DIAS,
                "lags_dias": str(P1_LAGS_DIAS),
                "ventanas_media_dias": str(P1_VENTANAS_MEDIA_DIAS),
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
        if not isinstance(modelo, BaselineNaiveSemanal):
            mlflow.sklearn.log_model(
                modelo, name="modelo", input_example=x_test.head(5), skops_trusted_types=SKOPS_TRUSTED
            )
        for ruta in artefactos:
            mlflow.log_artifact(str(ruta), artifact_path="evaluacion")

    logger.info(
        f"{nombre_run}: val_mae={metricas['val_mae']:.3f} test_mae={metricas['test_mae']:.3f} "
        f"test_rmse={metricas['test_rmse']:.3f} test_wape={metricas['test_wape']:.3f}"
    )
    return metricas


@app.command()
def main(viajes_path: Path = VIAJES_PARQUET, experiment_name: str = P1_MLFLOW_EXPERIMENT):
    logger.info("Iniciando pipeline P1")
    P1_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    viajes = cargar_viajes(viajes_path)
    capacidad = cargar_capacidad(ESTACIONES_PARQUET)
    panel = p1_construir_panel_diario(viajes, P1_TRAIN_START, P1_TEST_END)
    df = p1_agregar_features(
        panel, capacidad, P1_FERIADOS, P1_LAGS_DIAS, P1_VENTANAS_MEDIA_DIAS, P1_HORIZONTE_DIAS
    )

    train = cortar(df, P1_TRAIN_START, P1_VAL_START)
    val = cortar(df, P1_VAL_START, P1_TEST_START)
    test = cortar(df, P1_TEST_START, P1_TEST_END)

    filas_antes = {"train": len(train), "val": len(val), "test": len(test)}
    train, val, test = (d.dropna(subset=P1_FEATURES) for d in (train, val, test))
    for nombre, d in (("train", train), ("val", val), ("test", test)):
        logger.info(
            f"{nombre}: filas={len(d):,} (sin historia suficiente={filas_antes[nombre] - len(d):,}) "
            f"estaciones={d.index.get_level_values('estacion').nunique()} "
            f"media={d['viajes'].mean():.2f} viajes/estacion-dia"
        )

    datos = (
        (train[P1_FEATURES].astype("float64"), train["viajes"]),
        (val[P1_FEATURES].astype("float64"), val["viajes"]),
        (test[P1_FEATURES].astype("float64"), test["viajes"]),
        test,
    )
    base_test = test[f"lag_{P1_BASELINE_LAG_DIAS}"].to_numpy(dtype="float64")

    git_commit, git_sucio = obtener_git_commit()
    if git_sucio:
        logger.warning("Hay cambios sin commit: el run no será reproducible desde git_commit")
    contexto = {
        "git_commit": git_commit,
        "git_sucio": git_sucio,
        "data_version": json.loads(MANIFEST_PATH.read_text())["data_version"],
        "params_datos": {
            "estaciones_test": test.index.get_level_values("estacion").nunique(),
            "filas_train": len(train),
            "filas_val": len(val),
            "filas_test": len(test),
        },
    }
    mlflow.set_tracking_uri(f"sqlite:///{PROJ_ROOT / 'mlflow.db'}")
    mlflow.set_experiment(experiment_name)

    entrenar_y_registrar(
        P1_MLFLOW_BASELINE_RUN_NAME, "baseline",
        BaselineNaiveSemanal(f"lag_{P1_BASELINE_LAG_DIAS}"), datos, contexto, base_test,
    )
    entrenar_y_registrar(
        P1_MLFLOW_LINEAL_RUN_NAME, "candidato", construir_lineal(), datos, contexto, base_test
    )
    entrenar_y_registrar(
        P1_MLFLOW_CANDIDATE_RUN_NAME, "candidato", construir_hgb(), datos, contexto, base_test
    )
    logger.success("Pipeline P1 terminado")


if __name__ == "__main__":
    app()
