"""P2: clasificación de estaciones MiBici en riesgo de quedarse sin bicis o sin anclajes."""

import json
import os
from pathlib import Path
import subprocess

from loguru import logger
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
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
    P2_ET_PARAMS,
    P2_FRACCION_CAPACIDAD,
    P2_HGB_PARAMS,
    P2_HGB_SEARCH_GRID,
    P2_HORIZONTE_HORAS,
    P2_MAX_DURATION_SECONDS,
    P2_MIN_DURATION_SECONDS,
    P2_MLFLOW_BASELINE_RUN_NAME,
    P2_MLFLOW_CANDIDATE_RUN_NAME,
    P2_MLFLOW_ET_RUN_NAME,
    P2_MLFLOW_EXPERIMENT,
    P2_MLFLOW_RF_RUN_NAME,
    P2_MLFLOW_SEARCH_RUN_NAME,
    P2_OUTPUT_DIR,
    P2_QUESTION,
    P2_RF_PARAMS,
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

# MLflow 3 guarda los modelos de sklearn con skops y pide marcar como confiable este
# tipo interno de HistGradientBoosting (el modelo lo entrenamos nosotros).
SKOPS_TRUSTED = [
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.tree._tree.Tree",
]


def obtener_git_commit() -> tuple[str, bool]:
    """Commit actual y si hay cambios sin commitear (para los tags git_commit y git_sucio)."""
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
    """Lee los viajes y deja fuera los de menos de 1 minuto o más de 24 horas."""
    viajes = pd.read_parquet(
        ruta, columns=["inicio_viaje", "fin_viaje", "origen_id", "destino_id"]
    )
    duracion = (viajes["fin_viaje"] - viajes["inicio_viaje"]).dt.total_seconds()
    return viajes.loc[duracion.between(P2_MIN_DURATION_SECONDS, P2_MAX_DURATION_SECONDS)].dropna()


def cargar_capacidad(ruta: Path) -> pd.Series:
    """Capacidad (dpcapacity) de cada estación, indexada por id."""
    estaciones = pd.read_parquet(ruta, columns=["id", "dpcapacity"])
    return estaciones.set_index("id")["dpcapacity"]


def cortar(df: pd.DataFrame, inicio: str, fin: str) -> pd.DataFrame:
    """Filas de [inicio, fin) cuya ventana futura no se pasa de `fin`.

    Si no se cortan, la etiqueta de las últimas horas de un conjunto se calcularía con
    viajes del siguiente.
    """
    hora = df.index.get_level_values("hora")
    limite = pd.Timestamp(fin) - pd.Timedelta(hours=P2_HORIZONTE_HORAS)
    return df.loc[(hora >= pd.Timestamp(inicio)) & (hora < limite)]


def construir_baseline() -> Pipeline:
    """Baseline: regresión logística con features escaladas y pesos para la clase rara."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("modelo", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )


def construir_candidato(**overrides) -> HistGradientBoostingClassifier:
    """HistGradientBoosting con parámetros centralizados y overrides opcionales."""
    params = {
        **P2_HGB_PARAMS,
        **overrides,
        "class_weight": "balanced",
        "early_stopping": False,
        "random_state": SEED,
    }
    return HistGradientBoostingClassifier(**params)


def construir_random_forest(**overrides) -> RandomForestClassifier:
    """Random Forest como candidato no lineal adicional."""
    params = {
        **P2_RF_PARAMS,
        **overrides,
        "class_weight": "balanced",
        "random_state": SEED,
    }
    return RandomForestClassifier(**params)


def construir_extra_trees(**overrides) -> ExtraTreesClassifier:
    """Extra Trees como segundo candidato basado en ensambles."""
    params = {
        **P2_ET_PARAMS,
        **overrides,
        "class_weight": "balanced",
        "random_state": SEED,
    }
    return ExtraTreesClassifier(**params)


def construir_modelo_p2(model_name: str, params: dict):
    """Construye un modelo P2 a partir del nombre usado por MLflow Project."""
    nombre = model_name.strip().lower()

    if nombre == "logreg":
        if params:
            raise ValueError("logreg no acepta params_json en esta entrega.")
        return construir_baseline()

    if nombre == "hgb":
        return construir_candidato(**params)

    if nombre == "random_forest":
        return construir_random_forest(**params)

    if nombre == "extra_trees":
        return construir_extra_trees(**params)

    raise ValueError(
        f"Modelo P2 inválido: {model_name}. "
        "Usa logreg, hgb, random_forest, extra_trees o hgb_search."
    )


def elegir_umbral(y_val: pd.Series, prob_val) -> float:
    """Umbral que da el mejor F1 en validación.

    Con pesos balanceados el 0.5 no sirve (salen demasiadas alertas). Nunca se elige con test.
    """
    precision, recall, umbrales = precision_recall_curve(y_val, prob_val)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(umbrales[f1[:-1].argmax()])


def evaluar(y, prob, umbral: float, prefijo: str, tipo: pd.Series) -> dict:
    """Métricas de un conjunto con su prefijo (train_, val_ o test_).

    También va el recall por tipo de riesgo y la accuracy de decir siempre "normal", para
    que se vea por qué no usamos accuracy.
    """
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
        f"{prefijo}_accuracy_trivial": 1 - y.mean(),  # como si siempre dijera "normal"
        f"{prefijo}_prevalencia": y.mean(),
        **recall_tipo,
    }


def guardar_graficas(nombre: str, y, prob, umbral: float) -> list[Path]:
    """Matriz de confusión, curva PR y classification_report de test, para subirlos a MLflow."""
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


def preparar_datos_p2(viajes_path: Path = VIAJES_PARQUET):
    """Prepara TRAIN, VAL y TEST de P2 sin entrenar modelos."""
    P2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    viajes = cargar_viajes(viajes_path)
    capacidad = cargar_capacidad(ESTACIONES_PARQUET)

    panel = p2_construir_panel(viajes, P2_TRAIN_START, P2_TEST_END)
    df = p2_agregar_etiqueta(
        panel,
        P2_HORIZONTE_HORAS,
        capacidad,
        P2_FRACCION_CAPACIDAD,
    )
    df = p2_agregar_features(df, P2_HORIZONTE_HORAS)

    estaciones_viajes = panel.index.get_level_values("estacion").unique()
    estaciones_modelo = df.index.get_level_values("estacion").unique()
    excluidas = estaciones_viajes.difference(estaciones_modelo)

    logger.info(
        f"Estaciones con viajes={len(estaciones_viajes)} | "
        f"excluidas por capacidad 0 o sin catálogo={len(excluidas)}"
    )

    train = cortar(df, P2_TRAIN_START, P2_VAL_START)
    val = cortar(df, P2_VAL_START, P2_TEST_START)
    test = cortar(df, P2_TEST_START, P2_TEST_END)

    viajes_train = viajes.loc[
        viajes["inicio_viaje"] < pd.Timestamp(P2_VAL_START)
    ]
    estaciones_train = pd.Index(viajes_train["origen_id"]).union(
        viajes_train["destino_id"]
    )

    val = val.loc[
        val.index.get_level_values("estacion").isin(estaciones_train)
    ]
    test = test.loc[
        test.index.get_level_values("estacion").isin(estaciones_train)
    ]

    historico = p2_ajustar_historico(train)
    train = p2_historico_pasado(train)
    val, test = (
        p2_aplicar_historico(d, historico)
        for d in (val, test)
    )

    filas_antes = {
        "train": len(train),
        "val": len(val),
        "test": len(test),
    }

    train, val, test = (
        d.dropna(subset=P2_FEATURES)
        for d in (train, val, test)
    )

    for nombre, d in (
        ("train", train),
        ("val", val),
        ("test", test),
    ):
        logger.info(
            f"{nombre}: filas={len(d):,} "
            f"(descartadas sin lags/histórico="
            f"{filas_antes[nombre] - len(d):,}) "
            f"estaciones="
            f"{d.index.get_level_values('estacion').nunique()} "
            f"prevalencia={d['riesgo'].mean():.2%}"
        )

    datos = tuple(
        (
            d[P2_FEATURES].astype("float64"),
            d["riesgo"],
            d["tipo_riesgo"],
        )
        for d in (train, val, test)
    )

    git_commit, git_sucio = obtener_git_commit()

    if git_sucio:
        logger.warning(
            "Hay cambios sin commit: el run no será reproducible "
            "desde git_commit"
        )

    contexto = {
        "git_commit": git_commit,
        "git_sucio": git_sucio,
        "data_version": json.loads(
            MANIFEST_PATH.read_text()
        )["data_version"],
        "params_datos": {
            "estaciones_excluidas": len(excluidas),
            "estaciones_train": (
                train.index.get_level_values("estacion").nunique()
            ),
            "filas_train": len(train),
            "filas_val": len(val),
            "filas_test": len(test),
        },
    }

    return datos, contexto


def entrenar_y_registrar(
    nombre_run,
    etapa,
    modelo,
    datos,
    contexto,
    nested: bool = False,
    artefactos_detallados: bool = True,
    registrar_modelo: bool = True,
    evaluar_test: bool = True,
    registrar_contexto_datos: bool = True,
) -> dict:
    """Entrena y registra una corrida P2 reproducible en MLflow."""
    (x_train, y_train, t_train), (x_val, y_val, t_val), (
        x_test,
        y_test,
        t_test,
    ) = datos

    modelo.fit(x_train, y_train)

    prob_train = modelo.predict_proba(x_train)[:, 1]
    prob_val = modelo.predict_proba(x_val)[:, 1]

    # El umbral siempre se selecciona exclusivamente con VALIDATION.
    umbral = elegir_umbral(y_val, prob_val)

    metricas = {
        **evaluar(
            y_train,
            prob_train,
            umbral,
            "train",
            t_train,
        ),
        **evaluar(
            y_val,
            prob_val,
            umbral,
            "val",
            t_val,
        ),
    }

    prob_test = None

    if evaluar_test:
        prob_test = modelo.predict_proba(x_test)[:, 1]
        metricas.update(
            evaluar(
                y_test,
                prob_test,
                umbral,
                "test",
                t_test,
            )
        )

    artefactos = []

    if artefactos_detallados:
        if prob_test is None:
            raise ValueError(
                "Los artefactos detallados requieren evaluación de TEST."
            )

        artefactos = guardar_graficas(
            nombre_run,
            y_test,
            prob_test,
            umbral,
        )

    if isinstance(modelo, Pipeline):
        estimador = modelo.named_steps["modelo"]
    else:
        estimador = modelo

    params_estimador = estimador.get_params(deep=False)

    params_mlflow = {
        "modelo": type(estimador).__name__,
        "definicion_riesgo": (
            "abs(neto t+1..t+H) >= fraccion * dpcapacity"
        ),
        "horizonte_horas": P2_HORIZONTE_HORAS,
        "fraccion_capacidad": P2_FRACCION_CAPACIDAD,
        "umbral_decision": round(umbral, 6),
        "features": len(P2_FEATURES),
        "train_inicio": P2_TRAIN_START,
        "val_inicio": P2_VAL_START,
        "test_inicio": P2_TEST_START,
        "test_fin": P2_TEST_END,
        "random_state": SEED,
        "data_version": contexto["data_version"],
        **contexto["params_datos"],
    }

    # Solo parámetros simples del estimador para evitar objetos complejos.
    for clave, valor in params_estimador.items():
        if isinstance(
            valor,
            (str, int, float, bool, type(None)),
        ):
            params_mlflow[f"model_{clave}"] = valor

    with mlflow.start_run(
        run_name=nombre_run,
        nested=nested,
    ):
        mlflow.set_tags(
            {
                "team": TEAM_ID,
                "equipo": TEAM_ID,
                "question": "p2",
                "pregunta": P2_QUESTION,
                "problem_type": "classification",
                "tipo": "clasificacion",
                "stage_of_work": etapa,
                "etapa": etapa,
                "model_family": type(estimador).__name__,
                "responsable": "Gerardo Ivan Conde Reynoso",
                "git_commit": contexto["git_commit"],
                "git_sucio": str(contexto["git_sucio"]),
                "data_version": contexto["data_version"],
                "threshold_selected_on": "validation",
            }
        )

        mlflow.log_params(params_mlflow)
        mlflow.log_metrics(metricas)

        if registrar_contexto_datos:
            train_dataset = x_train.reset_index(drop=True).copy()
            train_dataset["riesgo"] = y_train.to_numpy()

            dataset_mlflow = mlflow.data.from_pandas(
                train_dataset,
                source=str(MANIFEST_PATH),
                targets="riesgo",
                name="p2_train",
            )

            mlflow.log_input(
                dataset_mlflow,
                context="training",
            )

            mlflow.log_artifact(
                str(MANIFEST_PATH),
                artifact_path="data",
            )

            mlflow.log_dict(
                {
                    "question": "p2",
                    "features": P2_FEATURES,
                    "seed": SEED,
                    "train_inicio": P2_TRAIN_START,
                    "val_inicio": P2_VAL_START,
                    "test_inicio": P2_TEST_START,
                    "test_fin": P2_TEST_END,
                    "horizonte_horas": P2_HORIZONTE_HORAS,
                    "fraccion_capacidad": P2_FRACCION_CAPACIDAD,
                    "data_version": contexto["data_version"],
                    "model_params": {
                        k: v
                        for k, v in params_estimador.items()
                        if isinstance(
                            v,
                            (str, int, float, bool, type(None)),
                        )
                    },
                },
                "config/p2_resolved_config.json",
            )

        if registrar_modelo:
            ejemplo = (
                x_test.head(5)
                if evaluar_test
                else x_val.head(5)
            )

            firma = infer_signature(
                ejemplo,
                modelo.predict_proba(ejemplo)[:, 1],
            )

            mlflow.sklearn.log_model(
                modelo,
                name="modelo",
                signature=firma,
                input_example=ejemplo,
                skops_trusted_types=SKOPS_TRUSTED,
            )

        for ruta in artefactos:
            mlflow.log_artifact(
                str(ruta),
                artifact_path="evaluacion",
            )

    if evaluar_test:
        logger.info(
            f"{nombre_run}: "
            f"val_f1={metricas['val_f1']:.3f} "
            f"test_f1={metricas['test_f1']:.3f} "
            f"test_recall={metricas['test_recall']:.3f} "
            f"test_precision={metricas['test_precision']:.3f}"
        )
    else:
        logger.info(
            f"{nombre_run}: "
            f"val_f1={metricas['val_f1']:.3f} "
            f"val_pr_auc={metricas['val_pr_auc']:.3f}"
        )

    return metricas


def ejecutar_modelo_p2(
    model_name: str,
    params: dict,
    experiment_name: str = P2_MLFLOW_EXPERIMENT,
) -> dict:
    """Ejecuta una sola variante de P2 dentro del experimento común."""
    mlflow.set_experiment(experiment_name)

    modelo = construir_modelo_p2(model_name, params)
    datos, contexto = preparar_datos_p2()

    nombres = {
        "logreg": (
            P2_MLFLOW_BASELINE_RUN_NAME,
            "baseline",
        ),
        "hgb": (
            P2_MLFLOW_CANDIDATE_RUN_NAME,
            "candidate",
        ),
        "random_forest": (
            P2_MLFLOW_RF_RUN_NAME,
            "candidate",
        ),
        "extra_trees": (
            P2_MLFLOW_ET_RUN_NAME,
            "candidate",
        ),
    }

    nombre = model_name.strip().lower()

    if nombre not in nombres:
        raise ValueError(
            f"Modelo P2 inválido: {model_name}. "
            "Usa logreg, hgb, random_forest o extra_trees."
        )

    nombre_run, etapa = nombres[nombre]

    return entrenar_y_registrar(
        nombre_run,
        etapa,
        modelo,
        datos,
        contexto,
    )


def ejecutar_busqueda_hgb_p2(
    experiment_name: str = P2_MLFLOW_EXPERIMENT,
) -> dict:
    """Búsqueda manual HGB con parent run y children evaluados solo en VAL."""
    mlflow.set_experiment(experiment_name)

    datos, contexto = preparar_datos_p2()

    project_run_id = os.environ.get("MLFLOW_RUN_ID")

    if project_run_id:
        parent_context = mlflow.start_run(
            run_id=project_run_id,
        )
    else:
        parent_context = mlflow.start_run(
            run_name=P2_MLFLOW_SEARCH_RUN_NAME,
        )

    resultados = []

    with parent_context as parent_run:
        mlflow.set_tags(
            {
                "team": TEAM_ID,
                "equipo": TEAM_ID,
                "question": "p2",
                "pregunta": P2_QUESTION,
                "problem_type": "classification",
                "tipo": "clasificacion",
                "stage_of_work": "hyperparameter_search",
                "etapa": "hyperparameter_search",
                "model_family": "HistGradientBoostingClassifier",
                "responsable": "Gerardo Ivan Conde Reynoso",
                "git_commit": contexto["git_commit"],
                "git_sucio": str(contexto["git_sucio"]),
                "data_version": contexto["data_version"],
                "selection_split": "validation",
                "threshold_selected_on": "validation",
                "test_used_for_selection": "False",
            }
        )

        mlflow.log_params(
            {
                "search_strategy": "manual_grid",
                "num_trials": len(P2_HGB_SEARCH_GRID),
                "selection_metric": "val_f1",
                "secondary_metric": "val_pr_auc",
                "test_used_for_selection": False,
                "data_version": contexto["data_version"],
            }
        )

        mlflow.log_dict(
            {
                "grid": P2_HGB_SEARCH_GRID,
                "selection_metric": "val_f1",
                "secondary_metric": "val_pr_auc",
            },
            "busqueda/p2_hgb_grid.json",
        )

        # MLflow Projects exporta MLFLOW_RUN_ID. Se retira temporalmente
        # para permitir que start_run(nested=True) cree hijos reales.
        saved_run_id = os.environ.pop(
            "MLFLOW_RUN_ID",
            None,
        )

        try:
            for i, params in enumerate(
                P2_HGB_SEARCH_GRID,
                start=1,
            ):
                nombre_trial = f"p2_hgb_trial_{i:02d}"

                logger.info(
                    f"Ejecutando {nombre_trial}/"
                    f"{len(P2_HGB_SEARCH_GRID)}: {params}"
                )

                modelo = construir_candidato(**params)

                metricas = entrenar_y_registrar(
                    nombre_trial,
                    "hyperparameter_trial",
                    modelo,
                    datos,
                    contexto,
                    nested=True,
                    artefactos_detallados=False,
                    registrar_modelo=False,
                    evaluar_test=False,
                    registrar_contexto_datos=False,
                )

                resultados.append(
                    {
                        "trial": i,
                        "run_name": nombre_trial,
                        "params": params,
                        "val_f1": metricas["val_f1"],
                        "val_pr_auc": metricas["val_pr_auc"],
                        "val_recall": metricas["val_recall"],
                        "val_precision": metricas["val_precision"],
                    }
                )
        finally:
            if saved_run_id is not None:
                os.environ["MLFLOW_RUN_ID"] = saved_run_id

        mejor = max(
            resultados,
            key=lambda r: (
                r["val_f1"],
                r["val_pr_auc"],
            ),
        )

        mlflow.log_metrics(
            {
                "best_val_f1": mejor["val_f1"],
                "best_val_pr_auc": mejor["val_pr_auc"],
                "best_val_recall": mejor["val_recall"],
                "best_val_precision": mejor["val_precision"],
            }
        )

        mlflow.set_tags(
            {
                "best_trial": mejor["run_name"],
                "best_params": json.dumps(
                    mejor["params"],
                    sort_keys=True,
                ),
            }
        )

        resumen = {
            "parent_run_id": parent_run.info.run_id,
            "selection_metric": "val_f1",
            "secondary_metric": "val_pr_auc",
            "best_trial": mejor,
            "results": resultados,
        }

        mlflow.log_dict(
            resumen,
            "busqueda/p2_hgb_resultados.json",
        )

        logger.success(
            f"Mejor trial: {mejor['run_name']} "
            f"val_f1={mejor['val_f1']:.4f} "
            f"val_pr_auc={mejor['val_pr_auc']:.4f}"
        )

        return resumen


@app.command()
def main(viajes_path: Path = VIAJES_PARQUET, experiment_name: str = P2_MLFLOW_EXPERIMENT):
    """Pipeline completo de P2: datos, partición, baseline, candidato y registro en MLflow."""
    logger.info("Iniciando pipeline P2")
    P2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Panel estación x hora, etiqueta y features
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

    # 2. Partición temporal, sin revolver: TRAIN ene-sep, VAL oct, TEST nov-dic
    train = cortar(df, P2_TRAIN_START, P2_VAL_START)
    val = cortar(df, P2_VAL_START, P2_TEST_START)
    test = cortar(df, P2_TEST_START, P2_TEST_END)

    # en val/test solo entran estaciones que ya tenían viajes en TRAIN
    viajes_train = viajes.loc[viajes["inicio_viaje"] < pd.Timestamp(P2_VAL_START)]
    estaciones_train = pd.Index(viajes_train["origen_id"]).union(viajes_train["destino_id"])
    val = val.loc[val.index.get_level_values("estacion").isin(estaciones_train)]
    test = test.loc[test.index.get_level_values("estacion").isin(estaciones_train)]

    # 3. Perfil histórico, siempre sacado de TRAIN
    historico = p2_ajustar_historico(train)
    train = p2_historico_pasado(train)
    val, test = (p2_aplicar_historico(d, historico) for d in (val, test))

    # fuera filas sin lags completos (la primera semana de cada estación)
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

    # 4. MLflow
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
