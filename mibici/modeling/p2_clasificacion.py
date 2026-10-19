"""P2: clasificación de estaciones MiBici en riesgo de quedarse sin bicis o sin anclajes."""

import json
from pathlib import Path
import subprocess

from loguru import logger
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import typer

from mibici.config import (
    ESTACIONES_PARQUET,
    MANIFEST_PATH,
    P2_FRACCION_CAPACIDAD,
    P2_HORIZONTE_HORAS,
    P2_MAX_DURATION_SECONDS,
    P2_MIN_DURATION_SECONDS,
    P2_MLFLOW_BASELINE_RUN_NAME,
    P2_MLFLOW_CANDIDATE_RUN_NAME,
    P2_MLFLOW_EXPERIMENT,
    P2_OUTPUT_DIR,
    P2_QUESTION,
    P2_TEST_END,
    P2_TEST_START,
    P2_TRAIN_START,
    P2_VAL_START,
    PROJ_ROOT,
    SEED,
    TEAM_ID,
    VIAJES_PARQUET,
)
from mibici.features import (
    P2_FEATURES,
    p2_agregar_etiqueta,
    p2_agregar_features,
    p2_ajustar_historico,
    p2_aplicar_historico,
    p2_construir_panel,
    p2_historico_pasado,
)

app = typer.Typer()

SKOPS_TRUSTED = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]


def obtener_git_commit() -> tuple[str, bool]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT, text=True
        ).strip()
        estado = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJ_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "desconocido", True
    return commit, bool(estado)


def cargar_viajes(ruta: Path) -> pd.DataFrame:
    viajes = pd.read_parquet(
        ruta, columns=["inicio_viaje", "fin_viaje", "origen_id", "destino_id"]
    )
    duracion = (viajes["fin_viaje"] - viajes["inicio_viaje"]).dt.total_seconds()
    return viajes.loc[duracion.between(P2_MIN_DURATION_SECONDS, P2_MAX_DURATION_SECONDS)].dropna()


def cargar_capacidad(ruta: Path) -> pd.Series:
    estaciones = pd.read_parquet(ruta, columns=["id", "dpcapacity"])
    return estaciones.set_index("id")["dpcapacity"]


def cortar(df: pd.DataFrame, inicio: str, fin: str) -> pd.DataFrame:
    hora = df.index.get_level_values("hora")
    limite = pd.Timestamp(fin) - pd.Timedelta(hours=P2_HORIZONTE_HORAS)
    return df.loc[(hora >= pd.Timestamp(inicio)) & (hora < limite)]


def construir_baseline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("modelo", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )


def construir_candidato() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        class_weight="balanced",
        max_iter=200,
        learning_rate=0.1,
        early_stopping=False,
        random_state=SEED,
    )


def elegir_umbral(y_val: pd.Series, prob_val) -> float:
    precision, recall, umbrales = precision_recall_curve(y_val, prob_val)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(umbrales[f1[:-1].argmax()])


def evaluar(y, prob, umbral: float, prefijo: str, tipo: pd.Series) -> dict:
    pred = (prob >= umbral).astype(int)
    recall_tipo = {
        f"{prefijo}_recall_{t}": float(pred[(tipo == t).to_numpy()].mean())
        for t in ("sin_bicis", "sin_anclajes")
    }
    return {
        f"{prefijo}_f1": f1_score(y, pred),
        f"{prefijo}_recall": recall_score(y, pred),
        f"{prefijo}_precision": precision_score(y, pred, zero_division=0),
        f"{prefijo}_pr_auc": average_precision_score(y, prob),
        f"{prefijo}_accuracy": accuracy_score(y, pred),
        f"{prefijo}_accuracy_trivial": 1 - y.mean(), 
        f"{prefijo}_prevalencia": y.mean(),
        **recall_tipo,
    }


def guardar_graficas(nombre: str, y, prob, umbral: float) -> list[Path]:
    pred = (prob >= umbral).astype(int)

    ruta_cm = P2_OUTPUT_DIR / f"{nombre}_matriz_confusion.png"
    ConfusionMatrixDisplay.from_predictions(y, pred, display_labels=["normal", "riesgo"])
    plt.title(f"{nombre} · TEST")
    plt.savefig(ruta_cm, dpi=150, bbox_inches="tight")
    plt.close()

    ruta_pr = P2_OUTPUT_DIR / f"{nombre}_curva_pr.png"
    PrecisionRecallDisplay.from_predictions(y, prob, name=nombre)
    plt.savefig(ruta_pr, dpi=150, bbox_inches="tight")
    plt.close()

    ruta_reporte = P2_OUTPUT_DIR / f"{nombre}_reporte.txt"
    ruta_reporte.write_text(classification_report(y, pred, target_names=["normal", "riesgo"]))
    return [ruta_cm, ruta_pr, ruta_reporte]


def entrenar_y_registrar(nombre_run, etapa, modelo, datos, contexto) -> dict:
    (x_train, y_train, t_train), (x_val, y_val, t_val), (x_test, y_test, t_test) = datos

    modelo.fit(x_train, y_train)
    umbral = elegir_umbral(y_val, modelo.predict_proba(x_val)[:, 1])
    prob_test = modelo.predict_proba(x_test)[:, 1]

    metricas = {
        **evaluar(y_train, modelo.predict_proba(x_train)[:, 1], umbral, "train", t_train),
        **evaluar(y_val, modelo.predict_proba(x_val)[:, 1], umbral, "val", t_val),
        **evaluar(y_test, prob_test, umbral, "test", t_test),
    }
    artefactos = guardar_graficas(nombre_run, y_test, prob_test, umbral)

    with mlflow.start_run(run_name=nombre_run):
        mlflow.set_tags(
            {
                "equipo": TEAM_ID,
                "mlflow.user": TEAM_ID,
                "problema": "P2",
                "tipo": "clasificacion",
                "etapa": etapa,
                "pregunta": P2_QUESTION,
                "git_commit": contexto["git_commit"],
                "git_sucio": str(contexto["git_sucio"]),
                "data_version": contexto["data_version"],
            }
        )
        mlflow.log_params(
            {
                "modelo": type(modelo).__name__,
                "definicion_riesgo": "abs(neto t+1..t+H) >= fraccion * dpcapacity",
                "horizonte_horas": P2_HORIZONTE_HORAS,
                "fraccion_capacidad": P2_FRACCION_CAPACIDAD,
                "umbral_decision": round(umbral, 4),
                "class_weight": "balanced",
                "features": len(P2_FEATURES),
                "train_inicio": P2_TRAIN_START,
                "val_inicio": P2_VAL_START,
                "test_inicio": P2_TEST_START,
                "test_fin": P2_TEST_END,
                "random_state": SEED,
                "data_version": contexto["data_version"],
                **contexto["params_datos"],
            }
        )
        mlflow.log_metrics(metricas)
        mlflow.sklearn.log_model(
            modelo,
            name="modelo",
            input_example=x_test.head(5),
            skops_trusted_types=SKOPS_TRUSTED,
        )
        for ruta in artefactos:
            mlflow.log_artifact(str(ruta), artifact_path="evaluacion")

    logger.info(
        f"{nombre_run}: test_f1={metricas['test_f1']:.3f} "
        f"test_recall={metricas['test_recall']:.3f} "
        f"test_precision={metricas['test_precision']:.3f}"
    )
    return metricas


@app.command()
def main(viajes_path: Path = VIAJES_PARQUET, experiment_name: str = P2_MLFLOW_EXPERIMENT):
    logger.info("Iniciando pipeline P2")
    P2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    viajes = cargar_viajes(viajes_path)
    capacidad = cargar_capacidad(ESTACIONES_PARQUET)
    panel = p2_construir_panel(viajes, P2_TRAIN_START, P2_TEST_END)
    df = p2_agregar_etiqueta(panel, P2_HORIZONTE_HORAS, capacidad, P2_FRACCION_CAPACIDAD)
    df = p2_agregar_features(df, P2_HORIZONTE_HORAS)

    estaciones_viajes = panel.index.get_level_values("estacion").unique()
    estaciones_modelo = df.index.get_level_values("estacion").unique()
    excluidas = estaciones_viajes.difference(estaciones_modelo)
    logger.info(
        f"Estaciones con viajes={len(estaciones_viajes)} | excluidas por capacidad 0 o sin "
        f"catálogo={len(excluidas)}"
    )

    train = cortar(df, P2_TRAIN_START, P2_VAL_START)
    val = cortar(df, P2_VAL_START, P2_TEST_START)
    test = cortar(df, P2_TEST_START, P2_TEST_END)

    viajes_train = viajes.loc[viajes["inicio_viaje"] < pd.Timestamp(P2_VAL_START)]
    estaciones_train = pd.Index(viajes_train["origen_id"]).union(viajes_train["destino_id"])
    val = val.loc[val.index.get_level_values("estacion").isin(estaciones_train)]
    test = test.loc[test.index.get_level_values("estacion").isin(estaciones_train)]

    historico = p2_ajustar_historico(train)
    train = p2_historico_pasado(train)
    val, test = (p2_aplicar_historico(d, historico) for d in (val, test))

    filas_antes = {"train": len(train), "val": len(val), "test": len(test)}
    train, val, test = (d.dropna(subset=P2_FEATURES) for d in (train, val, test))
    for nombre, d in (("train", train), ("val", val), ("test", test)):
        logger.info(
            f"{nombre}: filas={len(d):,} (descartadas sin lags/histórico="
            f"{filas_antes[nombre] - len(d):,}) estaciones="
            f"{d.index.get_level_values('estacion').nunique()} "
            f"prevalencia={d['riesgo'].mean():.2%}"
        )

    datos = tuple(
        (d[P2_FEATURES].astype("float64"), d["riesgo"], d["tipo_riesgo"])
        for d in (train, val, test)
    )

    git_commit, git_sucio = obtener_git_commit()
    if git_sucio:
        logger.warning("Hay cambios sin commit: el run no será reproducible desde git_commit")
    contexto = {
        "git_commit": git_commit,
        "git_sucio": git_sucio,
        "data_version": json.loads(MANIFEST_PATH.read_text())["data_version"],
        "params_datos": {
            "estaciones_excluidas": len(excluidas),
            "estaciones_train": train.index.get_level_values("estacion").nunique(),
            "filas_train": len(train),
            "filas_val": len(val),
            "filas_test": len(test),
        },
    }
    mlflow.set_tracking_uri(f"sqlite:///{PROJ_ROOT / 'mlflow.db'}")
    mlflow.set_experiment(experiment_name)

    entrenar_y_registrar(
        P2_MLFLOW_BASELINE_RUN_NAME, "baseline", construir_baseline(), datos, contexto
    )
    entrenar_y_registrar(
        P2_MLFLOW_CANDIDATE_RUN_NAME, "candidato", construir_candidato(), datos, contexto
    )
    logger.success("Pipeline P2 terminado")


if __name__ == "__main__":
    app()
