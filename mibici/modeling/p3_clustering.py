"""Pipeline reproducible de clustering de estaciones MiBici."""

import json
from pathlib import Path
import subprocess

import folium
from folium.plugins import Fullscreen
import joblib
from loguru import logger
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
import typer

from mibici.config import (
    ESTACIONES_PARQUET,
    MANIFEST_PATH,
    P3_AGGLO_LINKAGE,
    P3_BASELINE_K,
    P3_GEO_PATH,
    P3_MAP_PATH,
    P3_MAX_DURATION_SECONDS,
    P3_MAX_INACTIVITY_DAYS,
    P3_MIN_DURATION_SECONDS,
    P3_MIN_HISTORY_DAYS,
    P3_MLFLOW_BASELINE_RUN_NAME,
    P3_MLFLOW_CANDIDATE_RUN_NAME,
    P3_MLFLOW_EXPERIMENT,
    P3_MLFLOW_RUN_NAME,
    P3_MODEL_PATH,
    P3_N_CLUSTERS,
    P3_N_INIT,
    P3_OUTPUT_DIR,
    P3_PCA_PATH,
    P3_QUESTION,
    P3_REDUCED_START,
    P3_TABLE_PATH,
    P3_TEST_END,
    P3_TEST_START,
    P3_TRAIN_END,
    P3_TRAIN_START,
    PROJ_ROOT,
    SEED,
    TEAM_ID,
    VIAJES_PARQUET,
)
from mibici.features import (
    P3_MODEL_FEATURES,
    p3_agregar_entropias,
    p3_construir_features,
    p3_filtrar_viajes,
)

app = typer.Typer()

COLORES_PERFIL = {
    "Fin de semana y mediodia": "#e74c3c",
    "Llegadas matutinas / salidas vespertinas": "#3498db",
    "Salidas matutinas / llegadas vespertinas": "#2ecc71",
    "Alta intensidad balanceada": "#8e44ad",
}


def preparar_features(
    features: pd.DataFrame,
    scaler: StandardScaler | None = None,
):
    x = features[P3_MODEL_FEATURES].copy()

    x["salidas_dia"] = np.log1p(x["salidas_dia"])
    x["llegadas_dia"] = np.log1p(x["llegadas_dia"])

    if scaler is None:
        scaler = StandardScaler()
        valores = scaler.fit_transform(x)
    else:
        valores = scaler.transform(x)

    x_scaled = pd.DataFrame(
        valores,
        index=x.index,
        columns=x.columns,
    )

    return x_scaled, scaler


def asignar_perfiles(
    features: pd.DataFrame,
    labels: pd.Series,
) -> dict[int, str]:
    perfiles = (
        features[P3_MODEL_FEATURES]
        .assign(cluster=labels)
        .groupby("cluster")
        .mean()
    )

    disponibles = {int(x) for x in perfiles.index}

    intensidad = (
        perfiles["salidas_dia"]
        + perfiles["llegadas_dia"]
    )

    cluster_alta = int(intensidad.idxmax())
    disponibles.remove(cluster_alta)

    cluster_fin_semana = int(
        perfiles.loc[
            sorted(disponibles),
            "share_fin_semana",
        ].idxmax()
    )
    disponibles.remove(cluster_fin_semana)

    score_salidas = (
        perfiles.loc[
            sorted(disponibles),
            "salida_manana_share",
        ]
        + perfiles.loc[
            sorted(disponibles),
            "llegada_tarde_share",
        ]
    )

    cluster_salidas = int(score_salidas.idxmax())
    disponibles.remove(cluster_salidas)

    cluster_llegadas = disponibles.pop()

    return {
        cluster_alta: "Alta intensidad balanceada",
        cluster_fin_semana: "Fin de semana y mediodia",
        cluster_salidas: "Salidas matutinas / llegadas vespertinas",
        cluster_llegadas: "Llegadas matutinas / salidas vespertinas",
    }


def generar_pca(
    x_train: pd.DataFrame,
    labels: pd.Series,
    perfiles: dict[int, str],
    output_path: Path,
):
    pca = PCA(n_components=2)
    componentes = pca.fit_transform(x_train)

    df = pd.DataFrame(
        componentes,
        index=x_train.index,
        columns=["PC1", "PC2"],
    )

    df["cluster"] = labels.astype(int)
    df["perfil"] = df["cluster"].map(perfiles)

    fig, ax = plt.subplots(figsize=(11, 7))

    for perfil, color in COLORES_PERFIL.items():
        datos = df.loc[df["perfil"] == perfil]

        ax.scatter(
            datos["PC1"],
            datos["PC2"],
            s=55,
            alpha=0.72,
            label=f"{perfil} ({len(datos)})",
            color=color,
            edgecolors="white",
            linewidths=0.4,
        )

    centroides = (
        df.groupby("perfil")[["PC1", "PC2"]]
        .mean()
    )

    for perfil, fila in centroides.iterrows():
        ax.scatter(
            fila["PC1"],
            fila["PC2"],
            s=240,
            marker="X",
            color=COLORES_PERFIL[perfil],
            edgecolors="black",
            linewidths=1.2,
        )

    ax.axhline(0, linewidth=0.6, alpha=0.4)
    ax.axvline(0, linewidth=0.6, alpha=0.4)

    ax.set_xlabel(
        f"PC1 — patrón temporal "
        f"({pca.explained_variance_ratio_[0]:.1%})"
    )

    ax.set_ylabel(
        f"PC2 — intensidad de uso "
        f"({pca.explained_variance_ratio_[1]:.1%})"
    )

    ax.set_title(
        "Perfiles de uso de estaciones MiBici\n"
        "Proyección PCA del modelo K-Means final"
    )

    ax.legend(
        title="Perfil",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return pca


def generar_mapa(
    resultado_geo: pd.DataFrame,
    output_path: Path,
):
    mapa = folium.Map(
        location=[
            resultado_geo["latitude"].mean(),
            resultado_geo["longitude"].mean(),
        ],
        tiles=None,
        zoom_start=13,
        control_scale=True,
        prefer_canvas=True,
    )

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Street_Map/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Tiles © Esri",
        name="Esri WorldStreetMap",
        overlay=False,
        control=False,
    ).add_to(mapa)

    Fullscreen(
        position="topleft",
        title="Pantalla completa",
        title_cancel="Salir",
    ).add_to(mapa)

    capas = {}

    for perfil in COLORES_PERFIL:
        n = int(
            (resultado_geo["perfil"] == perfil).sum()
        )

        capas[perfil] = folium.FeatureGroup(
            name=f"{perfil} ({n})",
            show=True,
        )

        capas[perfil].add_to(mapa)

    estados = {
        "estable": "Estable",
        "cambio": "Cambió de perfil",
        "no_comparable": "No comparable",
    }

    for _, fila in resultado_geo.iterrows():
        perfil = fila["perfil"]
        color = COLORES_PERFIL[perfil]
        estado = estados[fila["estado_test"]]

        popup = f"""
        <div style="width:320px">
        <b>{fila['name']}</b><br><br>
        <b>Estación:</b> {int(fila['estacion'])}<br>
        <b>Perfil:</b> {perfil}<br>
        <b>Cluster:</b> {int(fila['cluster'])}<br>
        <b>Estado en TEST:</b> {estado}<br><br>
        <b>Salidas/día:</b> {fila['salidas_dia']:.1f}<br>
        <b>Llegadas/día:</b> {fila['llegadas_dia']:.1f}<br>
        <b>Balance:</b> {fila['balance_relativo']:.3f}<br>
        <b>Fin de semana:</b> {fila['share_fin_semana']:.1%}<br><br>
        <b>Entropía destinos:</b> {fila['entropia_destinos']:.3f}<br>
        <b>Entropía orígenes:</b> {fila['entropia_origenes']:.3f}
        </div>
        """

        folium.CircleMarker(
            location=[
                fila["latitude"],
                fila["longitude"],
            ],
            radius=5,
            color=color,
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.80,
            popup=folium.Popup(
                popup,
                max_width=350,
            ),
            tooltip=(
                f"{fila['name']} | "
                f"{perfil} | {estado}"
            ),
        ).add_to(capas[perfil])

    folium.LayerControl(
        collapsed=False,
        position="topright",
    ).add_to(mapa)

    mapa.fit_bounds(
        [
            [
                resultado_geo["latitude"].min(),
                resultado_geo["longitude"].min(),
            ],
            [
                resultado_geo["latitude"].max(),
                resultado_geo["longitude"].max(),
            ],
        ]
    )

    mapa.save(output_path)


@app.command()
def main(
    viajes_path: Path = VIAJES_PARQUET,
    estaciones_path: Path = ESTACIONES_PARQUET,
    experiment_name: str = P3_MLFLOW_EXPERIMENT,
):
    logger.info("Iniciando pipeline P3")

    P3_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    P3_PCA_PATH.parent.mkdir(parents=True, exist_ok=True)
    P3_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    viajes = pd.read_parquet(viajes_path)
    estaciones = pd.read_parquet(estaciones_path)

    viajes["inicio_viaje"] = pd.to_datetime(
        viajes["inicio_viaje"]
    )
    viajes["fin_viaje"] = pd.to_datetime(
        viajes["fin_viaje"]
    )

    viajes = p3_filtrar_viajes(
        viajes,
        min_duration_seconds=P3_MIN_DURATION_SECONDS,
        max_duration_seconds=P3_MAX_DURATION_SECONDS,
    )

    train = viajes.loc[
        (viajes["inicio_viaje"] >= pd.Timestamp(P3_TRAIN_START))
        & (viajes["inicio_viaje"] <= pd.Timestamp(P3_TRAIN_END))
        & (viajes["fin_viaje"] < pd.Timestamp(P3_TEST_START))
    ].copy()

    test = viajes.loc[
        (viajes["inicio_viaje"] >= pd.Timestamp(P3_TEST_START))
        & (viajes["fin_viaje"] < pd.Timestamp(P3_TEST_END))
    ].copy()

    features_train, _ = (
        p3_construir_features(
            train,
            fin_periodo=pd.Timestamp(P3_TRAIN_END),
            max_inactivity_days=P3_MAX_INACTIVITY_DAYS,
            min_history_days=P3_MIN_HISTORY_DAYS,
        )
    )

    features_test, _ = (
        p3_construir_features(
            test,
            fin_periodo=pd.Timestamp(P3_TEST_END) - pd.Timedelta(seconds=1),
            max_inactivity_days=P3_MAX_INACTIVITY_DAYS,
            min_history_days=P3_MIN_HISTORY_DAYS,
        )
    )

    train_reducido = train.loc[
        train["inicio_viaje"] >= pd.Timestamp(P3_REDUCED_START)
    ].copy()

    features_reducido, _ = p3_construir_features(
        train_reducido,
        fin_periodo=pd.Timestamp(P3_TRAIN_END),
        max_inactivity_days=P3_MAX_INACTIVITY_DAYS,
        min_history_days=P3_MIN_HISTORY_DAYS,
    )

    x_train, scaler = preparar_features(
        features_train
    )

    x_test, _ = preparar_features(
        features_test,
        scaler=scaler,
    )

    x_reducido, _ = preparar_features(
        features_reducido
    )

    baseline = KMeans(
        n_clusters=P3_BASELINE_K,
        random_state=SEED,
        n_init=P3_N_INIT,
    )

    labels_baseline = pd.Series(
        baseline.fit_predict(x_train),
        index=x_train.index,
        name="cluster_baseline",
    )

    candidato = AgglomerativeClustering(
        n_clusters=P3_N_CLUSTERS,
        linkage=P3_AGGLO_LINKAGE,
    )

    labels_candidato = pd.Series(
        candidato.fit_predict(x_train),
        index=x_train.index,
        name="cluster_candidato",
    )

    modelo = KMeans(
        n_clusters=P3_N_CLUSTERS,
        random_state=SEED,
        n_init=P3_N_INIT,
    )

    labels_train = pd.Series(
        modelo.fit_predict(x_train),
        index=x_train.index,
        name="cluster_train",
    )

    labels_test = pd.Series(
        modelo.predict(x_test),
        index=x_test.index,
        name="cluster_test",
    )

    modelo_reducido = KMeans(
        n_clusters=P3_N_CLUSTERS,
        random_state=SEED,
        n_init=P3_N_INIT,
    )

    labels_reducido = pd.Series(
        modelo_reducido.fit_predict(x_reducido),
        index=x_reducido.index,
    )

    perfiles = asignar_perfiles(
        features_train,
        labels_train,
    )

    comunes_test = (
        x_train.index
        .intersection(x_test.index)
    )

    comunes_historico = (
        x_train.index
        .intersection(x_reducido.index)
    )

    comparacion_test = pd.DataFrame(
        {
            "cluster_train": labels_train.loc[
                comunes_test
            ],
            "cluster_test": labels_test.loc[
                comunes_test
            ],
        }
    )

    comparacion_test["cambio"] = (
        comparacion_test["cluster_train"]
        != comparacion_test["cluster_test"]
    )

    entropias = p3_agregar_entropias(
        features_train,
        train,
    )

    resultado = features_train.copy()

    resultado["cluster"] = (
        labels_train.astype(int)
    )

    resultado["perfil"] = (
        resultado["cluster"].map(perfiles)
    )

    resultado["entropia_destinos"] = (
        entropias["entropia_destinos"]
    )

    resultado["entropia_origenes"] = (
        entropias["entropia_origenes"]
    )

    estado_test = pd.Series(
        "no_comparable",
        index=resultado.index,
        dtype="object",
    )

    estado_test.loc[comparacion_test.index] = np.where(
        comparacion_test["cambio"],
        "cambio",
        "estable",
    )

    resultado["estado_test"] = estado_test

    resultado_geo = (
        resultado
        .reset_index()
        .merge(
            estaciones[
                [
                    "id",
                    "name",
                    "latitude",
                    "longitude",
                ]
            ],
            left_on="estacion",
            right_on="id",
            how="left",
            validate="one_to_one",
        )
        .drop(columns="id")
    )

    metricas_cluster = (
        resultado
        .reset_index()
        .groupby(
            ["cluster", "perfil"],
            as_index=False,
        )
        .agg(
            estaciones_train=(
                "estacion",
                "size",
            ),
            salidas_dia=(
                "salidas_dia",
                "mean",
            ),
            llegadas_dia=(
                "llegadas_dia",
                "mean",
            ),
            balance_relativo=(
                "balance_relativo",
                "mean",
            ),
            share_fin_semana=(
                "share_fin_semana",
                "mean",
            ),
            entropia_destinos=(
                "entropia_destinos",
                "mean",
            ),
            entropia_origenes=(
                "entropia_origenes",
                "mean",
            ),
        )
    )

    estabilidad_cluster = (
        comparacion_test
        .groupby("cluster_train")
        .agg(
            comparables_test=(
                "cambio",
                "size",
            ),
            cambios_test=(
                "cambio",
                "sum",
            ),
        )
    )

    estabilidad_cluster["estables_test"] = (
        estabilidad_cluster["comparables_test"]
        - estabilidad_cluster["cambios_test"]
    )

    estabilidad_cluster["estabilidad_pct"] = (
        estabilidad_cluster["estables_test"]
        / estabilidad_cluster["comparables_test"]
        * 100
    )

    tabla_maestra = (
        metricas_cluster
        .merge(
            estabilidad_cluster,
            left_on="cluster",
            right_index=True,
            how="left",
        )
        .sort_values("cluster")
    )

    tabla_path = (
        P3_TABLE_PATH
    )

    geo_path = (
        P3_GEO_PATH
    )

    mapa_path = (
        P3_MAP_PATH
    )

    pca_path = (
        P3_PCA_PATH
    )

    model_path = (
        P3_MODEL_PATH
    )

    tabla_maestra.to_csv(
        tabla_path,
        index=False,
    )

    resultado_geo.to_parquet(
        geo_path,
        index=False,
    )

    pca = generar_pca(
        x_train,
        labels_train,
        perfiles,
        pca_path,
    )

    generar_mapa(
        resultado_geo,
        mapa_path,
    )

    bundle = {
        "scaler": scaler,
        "kmeans": modelo,
        "features": P3_MODEL_FEATURES,
        "perfiles": perfiles,
    }

    joblib.dump(
        bundle,
        model_path,
    )

    metricas_baseline = {
        "train_silhouette": silhouette_score(
            x_train,
            labels_baseline,
        ),
        "train_davies_bouldin": davies_bouldin_score(
            x_train,
            labels_baseline,
        ),
        "train_calinski_harabasz": calinski_harabasz_score(
            x_train,
            labels_baseline,
        ),
    }

    metricas_candidato = {
        "train_silhouette": silhouette_score(
            x_train,
            labels_candidato,
        ),
        "train_davies_bouldin": davies_bouldin_score(
            x_train,
            labels_candidato,
        ),
        "train_calinski_harabasz": calinski_harabasz_score(
            x_train,
            labels_candidato,
        ),
    }

    metricas = {
        "train_silhouette": silhouette_score(
            x_train,
            labels_train,
        ),
        "test_silhouette": silhouette_score(
            x_test,
            labels_test,
        ),
        "train_davies_bouldin": davies_bouldin_score(
            x_train,
            labels_train,
        ),
        "test_davies_bouldin": davies_bouldin_score(
            x_test,
            labels_test,
        ),
        "train_calinski_harabasz": calinski_harabasz_score(
            x_train,
            labels_train,
        ),
        "test_calinski_harabasz": calinski_harabasz_score(
            x_test,
            labels_test,
        ),
        "ari_train_test": adjusted_rand_score(
            labels_train.loc[comunes_test],
            labels_test.loc[comunes_test],
        ),
        "nmi_train_test": normalized_mutual_info_score(
            labels_train.loc[comunes_test],
            labels_test.loc[comunes_test],
        ),
        "estabilidad_exacta_test": (
            comparacion_test["cluster_train"]
            == comparacion_test["cluster_test"]
        ).mean(),
        "ari_historico_reducido": adjusted_rand_score(
            labels_train.loc[comunes_historico],
            labels_reducido.loc[comunes_historico],
        ),
        "nmi_historico_reducido": normalized_mutual_info_score(
            labels_train.loc[comunes_historico],
            labels_reducido.loc[comunes_historico],
        ),
        "estaciones_train": len(x_train),
        "estaciones_test": len(x_test),
        "estaciones_comunes": len(comunes_test),
        "pca_varianza_acumulada": (
            pca.explained_variance_ratio_.sum()
        ),
    }

    manifest = json.loads(
        MANIFEST_PATH.read_text()
    )
    data_version = manifest["data_version"]

    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJ_ROOT,
        text=True,
    ).strip()

    mlflow.set_tracking_uri(
        f"sqlite:///{PROJ_ROOT / 'mlflow.db'}"
    )

    mlflow.set_experiment(
        experiment_name
    )

    with mlflow.start_run(
        run_name=P3_MLFLOW_BASELINE_RUN_NAME
    ):
        mlflow.log_params(
            {
                "modelo": "KMeans",
                "k": P3_BASELINE_K,
                "random_state": SEED,
                "n_init": P3_N_INIT,
                "features": len(P3_MODEL_FEATURES),
                "data_version": data_version,
                "max_inactivity_days": P3_MAX_INACTIVITY_DAYS,
                "min_history_days": P3_MIN_HISTORY_DAYS,
                "min_duration_seconds": P3_MIN_DURATION_SECONDS,
                "max_duration_seconds": P3_MAX_DURATION_SECONDS,
                "train_inicio": P3_TRAIN_START[:10],
                "train_fin": P3_TRAIN_END[:10],
                "transformacion_intensidad": "log1p",
                "scaler": "StandardScaler",
            }
        )

        mlflow.log_metrics(metricas_baseline)

        mlflow.set_tags(
            {
                "equipo": TEAM_ID,
                "mlflow.user": TEAM_ID,
                "problema": "P3",
                "tipo": "clustering",
                "etapa": "baseline",
                "pregunta": P3_QUESTION,
                "git_commit": git_commit,
                "data_version": data_version,
            }
        )

    with mlflow.start_run(
        run_name=P3_MLFLOW_CANDIDATE_RUN_NAME
    ):
        mlflow.log_params(
            {
                "modelo": "AgglomerativeClustering",
                "k": P3_N_CLUSTERS,
                "linkage": P3_AGGLO_LINKAGE,
                "features": len(P3_MODEL_FEATURES),
                "data_version": data_version,
                "max_inactivity_days": P3_MAX_INACTIVITY_DAYS,
                "min_history_days": P3_MIN_HISTORY_DAYS,
                "min_duration_seconds": P3_MIN_DURATION_SECONDS,
                "max_duration_seconds": P3_MAX_DURATION_SECONDS,
                "train_inicio": P3_TRAIN_START[:10],
                "train_fin": P3_TRAIN_END[:10],
                "transformacion_intensidad": "log1p",
                "scaler": "StandardScaler",
            }
        )

        mlflow.log_metrics(metricas_candidato)

        mlflow.set_tags(
            {
                "equipo": TEAM_ID,
                "mlflow.user": TEAM_ID,
                "problema": "P3",
                "tipo": "clustering",
                "etapa": "candidato",
                "pregunta": P3_QUESTION,
                "git_commit": git_commit,
                "data_version": data_version,
            }
        )

    with mlflow.start_run(
        run_name=P3_MLFLOW_RUN_NAME
    ) as run:
        mlflow.log_params(
            {
                "modelo": "KMeans",
                "k": P3_N_CLUSTERS,
                "random_state": SEED,
                "n_init": P3_N_INIT,
                "features": len(P3_MODEL_FEATURES),
                "data_version": data_version,
                "max_inactivity_days": P3_MAX_INACTIVITY_DAYS,
                "min_history_days": P3_MIN_HISTORY_DAYS,
                "min_duration_seconds": P3_MIN_DURATION_SECONDS,
                "max_duration_seconds": P3_MAX_DURATION_SECONDS,
                "split": "temporal_80_20",
                "train_inicio": P3_TRAIN_START[:10],
                "train_fin": P3_TRAIN_END[:10],
                "test_inicio": P3_TEST_START[:10],
                "test_fin": str(
                    (
                        pd.Timestamp(P3_TEST_END)
                        - pd.Timedelta(seconds=1)
                    ).date()
                ),
                "transformacion_intensidad": "log1p",
                "scaler": "StandardScaler",
            }
        )

        mlflow.log_metrics(metricas)

        mlflow.set_tags(
            {
                "equipo": TEAM_ID,
                "mlflow.user": TEAM_ID,
                "problema": "P3",
                "tipo": "clustering",
                "estado": "modelo_final",
                "pregunta": P3_QUESTION,
                "git_commit": git_commit,
                "data_version": data_version,
            }
        )

        for artifact in [
            model_path,
            mapa_path,
            pca_path,
            geo_path,
            tabla_path,
        ]:
            mlflow.log_artifact(
                artifact,
                artifact_path="p3",
            )

        run_id = run.info.run_id

    logger.info(
        f"Viajes limpios: {len(viajes):,}"
    )

    logger.info(
        f"Train: {len(train):,} | "
        f"Test: {len(test):,}"
    )

    logger.info(
        f"Estaciones TRAIN: {len(x_train)} | "
        f"TEST: {len(x_test)}"
    )

    logger.info(
        f"Perfiles: {perfiles}"
    )

    logger.info(
        f"Silhouette TRAIN: "
        f"{metricas['train_silhouette']:.4f}"
    )

    logger.info(
        f"Silhouette TEST: "
        f"{metricas['test_silhouette']:.4f}"
    )

    logger.info(
        f"ARI TRAIN-TEST: "
        f"{metricas['ari_train_test']:.4f}"
    )

    logger.info(
        f"ARI histórico reducido: "
        f"{metricas['ari_historico_reducido']:.4f}"
    )

    logger.success(
        f"Pipeline P3 terminado | "
        f"MLflow run: {run_id}"
    )


if __name__ == "__main__":
    app()
