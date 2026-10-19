"""Feature engineering reutilizable del proyecto MiBici (Equipo 32).

Este módulo concentra TODA la construcción de variables; los pipelines de
``mibici/modeling`` solo importan estas funciones, de modo que el mismo código
se usa en entrenamiento y en predicción.

Organización por pregunta y autoría:

* ``p1_*`` – P1, regresión de viajes diarios por estación (Johanna Camargo).
* ``p2_*`` – P2, clasificación de riesgo de vaciado/llenado (Manuel Cardona).
* ``p3_*`` – P3, clustering de estaciones por patrón de uso (Gerardo Conde).

Regla común (prevención de fuga): ninguna variable de entrada usa información
posterior al momento en que se emitiría la predicción. Las únicas funciones que
miran al futuro son las que construyen la *etiqueta* (p. ej.
``p2_agregar_etiqueta``), nunca las que construyen variables.
"""

import numpy as np
import pandas as pd

P3_MODEL_FEATURES = [
    "salidas_dia",
    "llegadas_dia",
    "balance_relativo",
    "share_fin_semana",
    "salida_manana_share",
    "llegada_manana_share",
    "salida_mediodia_share",
    "llegada_mediodia_share",
    "salida_tarde_share",
    "llegada_tarde_share",
    "salida_noche_share",
    "llegada_noche_share",
]


def p3_filtrar_viajes(
    viajes: pd.DataFrame,
    min_duration_seconds: int = 60,
    max_duration_seconds: int = 86400,
) -> pd.DataFrame:
    """Descarta viajes con duración fuera de ``[min, max]`` segundos.

    Viajes de menos de 60 s suelen ser desanclajes fallidos y los de más de
    24 h, bicicletas no ancladas o registros anómalos; ninguno representa uso
    real de la estación. Devuelve una copia filtrada de ``viajes``.
    (P3 – Gerardo Conde.)
    """
    df = viajes.copy()

    duracion = (
        df["fin_viaje"] - df["inicio_viaje"]
    ).dt.total_seconds()

    return df.loc[
        (duracion >= min_duration_seconds)
        & (duracion <= max_duration_seconds)
    ].copy()


def p3_construir_features(
    viajes: pd.DataFrame,
    fin_periodo: pd.Timestamp,
    max_inactivity_days: int = 60,
    min_history_days: int = 30,
):
    """Construye el perfil de uso de cada estación elegible para un periodo.

    1. Calcula la cobertura de cada estación (primera/última actividad, días
       activos y movimientos) a partir de salidas y llegadas.
    2. Marca como elegibles las estaciones con al menos ``min_history_days``
       de historial y menos de ``max_inactivity_days`` sin actividad al cierre
       ``fin_periodo`` (criterio dinámico, sin lista fija de estaciones).
    3. Para las elegibles calcula las 12 variables de ``P3_MODEL_FEATURES``:
       intensidad (``salidas_dia``, ``llegadas_dia``, normalizadas por días de
       exposición), ``balance_relativo``, ``share_fin_semana`` y la proporción
       de salidas y llegadas en los bloques mañana (05-10 h), mediodía
       (11-15 h), tarde (16-20 h) y noche (21-04 h).

    Solo usa los viajes recibidos, por lo que se llama por separado para TRAIN
    y TEST. Devuelve ``(features, cobertura)``. (P3 – Gerardo Conde.)
    """
    salidas_ev = viajes[
        ["inicio_viaje", "origen_id"]
    ].rename(
        columns={
            "inicio_viaje": "fecha",
            "origen_id": "estacion",
        }
    )

    llegadas_ev = viajes[
        ["fin_viaje", "destino_id"]
    ].rename(
        columns={
            "fin_viaje": "fecha",
            "destino_id": "estacion",
        }
    )

    eventos = pd.concat(
        [salidas_ev, llegadas_ev],
        ignore_index=True,
    )

    eventos["dia"] = eventos["fecha"].dt.normalize()

    cobertura = (
        eventos
        .groupby("estacion")
        .agg(
            primera_actividad=("fecha", "min"),
            ultima_actividad=("fecha", "max"),
            dias_activos=("dia", "nunique"),
            movimientos=("fecha", "size"),
        )
        .reset_index()
    )

    cobertura["dias_historial"] = (
        cobertura["ultima_actividad"]
        - cobertura["primera_actividad"]
    ).dt.days + 1

    cobertura["dias_sin_actividad"] = (
        fin_periodo
        - cobertura["ultima_actividad"]
    ).dt.days

    cobertura["eligible_p3"] = (
        (
            cobertura["dias_sin_actividad"]
            < max_inactivity_days
        )
        & (
            cobertura["dias_historial"]
            >= min_history_days
        )
    )

    elegibles = cobertura.loc[
        cobertura["eligible_p3"],
        ["estacion", "primera_actividad"],
    ].copy()

    if elegibles.empty:
        raise ValueError(
            "No hay estaciones elegibles para P3."
        )

    elegibles["dias_exposicion"] = (
        fin_periodo.normalize()
        - elegibles["primera_actividad"].dt.normalize()
    ).dt.days + 1

    ids = set(elegibles["estacion"])

    salidas = viajes.loc[
        viajes["origen_id"].isin(ids),
        ["origen_id", "inicio_viaje"],
    ].copy()

    llegadas = viajes.loc[
        viajes["destino_id"].isin(ids),
        ["destino_id", "fin_viaje"],
    ].copy()

    salidas.columns = ["estacion", "fecha"]
    llegadas.columns = ["estacion", "fecha"]

    for datos in (salidas, llegadas):
        hora = datos["fecha"].dt.hour

        datos["bloque"] = np.select(
            [
                (hora >= 5) & (hora <= 10),
                (hora >= 11) & (hora <= 15),
                (hora >= 16) & (hora <= 20),
            ],
            [
                "manana",
                "mediodia",
                "tarde",
            ],
            default="noche",
        )

        datos["fin_semana"] = (
            datos["fecha"].dt.dayofweek >= 5
        )

    base = elegibles.set_index("estacion")

    base["salidas"] = (
        salidas.groupby("estacion").size()
    )

    base["llegadas"] = (
        llegadas.groupby("estacion").size()
    )

    base["salidas_fin_semana"] = (
        salidas.loc[salidas["fin_semana"]]
        .groupby("estacion")
        .size()
    )

    base["llegadas_fin_semana"] = (
        llegadas.loc[llegadas["fin_semana"]]
        .groupby("estacion")
        .size()
    )

    bloques_salida = (
        salidas
        .groupby(["estacion", "bloque"])
        .size()
        .unstack(fill_value=0)
        .reindex(
            columns=[
                "manana",
                "mediodia",
                "tarde",
                "noche",
            ],
            fill_value=0,
        )
    )

    bloques_llegada = (
        llegadas
        .groupby(["estacion", "bloque"])
        .size()
        .unstack(fill_value=0)
        .reindex(
            columns=[
                "manana",
                "mediodia",
                "tarde",
                "noche",
            ],
            fill_value=0,
        )
    )

    base = base.join(
        bloques_salida.add_prefix("salida_")
    )

    base = base.join(
        bloques_llegada.add_prefix("llegada_")
    )

    base = base.fillna(0)

    features = pd.DataFrame(index=base.index)

    features["salidas_dia"] = (
        base["salidas"]
        / base["dias_exposicion"]
    )

    features["llegadas_dia"] = (
        base["llegadas"]
        / base["dias_exposicion"]
    )

    features["balance_relativo"] = (
        (base["llegadas"] - base["salidas"])
        / (base["llegadas"] + base["salidas"])
    )

    features["share_fin_semana"] = (
        (
            base["salidas_fin_semana"]
            + base["llegadas_fin_semana"]
        )
        / (
            base["salidas"]
            + base["llegadas"]
        )
    )

    for bloque in [
        "manana",
        "mediodia",
        "tarde",
        "noche",
    ]:
        features[f"salida_{bloque}_share"] = (
            base[f"salida_{bloque}"]
            / base["salidas"]
        )

        features[f"llegada_{bloque}_share"] = (
            base[f"llegada_{bloque}"]
            / base["llegadas"]
        )

    return (
        features[P3_MODEL_FEATURES].fillna(0),
        cobertura,
    )


def p3_entropia_normalizada(
    serie: pd.Series,
) -> float:
    """Entropía de Shannon de una serie categórica, normalizada a ``[0, 1]``.

    Divide la entropía entre ``log(n_categorias)``; devuelve 0 si hay una sola
    categoría. Se usa para medir qué tan diversos son los destinos (u orígenes)
    de una estación. (P3 – Gerardo Conde.)
    """
    p = serie.value_counts(normalize=True)

    if len(p) <= 1:
        return 0.0

    h = -(p * np.log(p)).sum()

    return float(h / np.log(len(p)))


def p3_agregar_entropias(
    features: pd.DataFrame,
    viajes: pd.DataFrame,
) -> pd.DataFrame:
    """Agrega ``entropia_destinos`` y ``entropia_origenes`` a cada estación.

    Son variables descriptivas de conectividad: se usan para interpretar los
    perfiles, no como entrada del modelo final. (P3 – Gerardo Conde.)
    """
    resultado = features.copy()
    ids = set(resultado.index)

    ent_destinos = (
        viajes.loc[
            viajes["origen_id"].isin(ids)
        ]
        .groupby("origen_id")["destino_id"]
        .apply(p3_entropia_normalizada)
    )

    ent_origenes = (
        viajes.loc[
            viajes["destino_id"].isin(ids)
        ]
        .groupby("destino_id")["origen_id"]
        .apply(p3_entropia_normalizada)
    )

    resultado["entropia_destinos"] = (
        resultado.index.to_series()
        .map(ent_destinos)
        .fillna(0)
    )

    resultado["entropia_origenes"] = (
        resultado.index.to_series()
        .map(ent_origenes)
        .fillna(0)
    )

    return resultado


# P2

P2_FEATURES = [
    "hora_dia",
    "dia_semana",
    "fin_semana",
    "salidas_h0",
    "llegadas_h0",
    "neto_h0",
    "neto_h1",
    "neto_h2",
    "neto_3h",
    "capacidad",
    "neto_3h_rel",
    "neto_futuro_ayer",
    "neto_futuro_semana",
    "tasa_riesgo_hist",
    "neto_futuro_medio_hist",
]

P2_LLAVES_HISTORICO = ["estacion", "hora_dia", "fin_semana"]


def p2_construir_panel(viajes: pd.DataFrame, inicio: str, fin: str) -> pd.DataFrame:
    """Panel horario estación × hora con ``salidas``, ``llegadas`` y ``neto``.

    Las salidas se asignan a la hora de inicio del viaje y las llegadas a la
    hora de fin. Rellena con 0 las horas sin movimientos y conserva cada
    estación solo a partir de su primera actividad, para no inventar historia
    a estaciones nuevas. (P2 – Manuel Cardona.)
    """
    salidas = (
        viajes.assign(hora=viajes["inicio_viaje"].dt.floor("h"))
        .groupby(["origen_id", "hora"])
        .size()
        .rename("salidas")
    )
    salidas.index.names = ["estacion", "hora"]

    llegadas = (
        viajes.assign(hora=viajes["fin_viaje"].dt.floor("h"))
        .groupby(["destino_id", "hora"])
        .size()
        .rename("llegadas")
    )
    llegadas.index.names = ["estacion", "hora"]

    estaciones = sorted(set(viajes["origen_id"]) | set(viajes["destino_id"]))
    horas = pd.date_range(inicio, fin, freq="h", inclusive="left")
    indice = pd.MultiIndex.from_product([estaciones, horas], names=["estacion", "hora"])

    eventos = pd.concat([salidas, llegadas], axis=1)
    panel = eventos.reindex(indice).fillna(0).astype("int32")
    panel["neto"] = panel["llegadas"] - panel["salidas"]

    primera_hora = eventos.reset_index().groupby("estacion")["hora"].min()
    estacion = panel.index.get_level_values("estacion")
    hora = panel.index.get_level_values("hora")
    return panel.loc[hora >= estacion.map(primera_hora)]


def p2_agregar_etiqueta(
    panel: pd.DataFrame, horizonte: int, capacidad: pd.Series, fraccion: float
) -> pd.DataFrame:
    """Construye la etiqueta de riesgo de P2 (mira al futuro a propósito).

    ``neto_futuro`` es la suma del flujo neto en ``t+1 … t+H``. Hay riesgo
    (``riesgo = 1``) si ``|neto_futuro| >= fraccion × capacidad``; el signo
    indica el tipo (``sin_bicis`` si salen más de las que llegan,
    ``sin_anclajes`` en el caso contrario). Se descartan filas sin ventana
    futura completa o con capacidad 0. (P2 – Manuel Cardona.)
    """
    df = panel.copy()
    neto = df.groupby(level="estacion")["neto"]
    df["neto_futuro"] = sum(neto.shift(-k) for k in range(1, horizonte + 1))
    df["capacidad"] = df.index.get_level_values("estacion").map(capacidad)

    umbral = fraccion * df["capacidad"]
    df["riesgo"] = (df["neto_futuro"].abs() >= umbral).astype("int8")
    df["tipo_riesgo"] = np.select(
        [df["neto_futuro"] <= -umbral, df["neto_futuro"] >= umbral],
        ["sin_bicis", "sin_anclajes"],
        default="normal",
    )
    return df.loc[df["neto_futuro"].notna() & (df["capacidad"] > 0)].copy()


def p2_agregar_features(df: pd.DataFrame, horizonte: int) -> pd.DataFrame:
    """Variables de entrada de P2 disponibles en la hora de decisión ``t``.

    Calendario (hora, día de la semana, fin de semana), flujos de la hora
    actual y de las dos anteriores, neto acumulado de 3 h (absoluto y relativo
    a la capacidad) y el neto que hubo en la misma ventana futura ayer
    (``shift(24-k)``) y hace una semana (``shift(168-k)``). Como ``H < 24``,
    todos los rezagos son anteriores a ``t``; por eso se valida el horizonte.
    (P2 – Manuel Cardona.)
    """
    if not 1 <= horizonte < 24:
        raise ValueError("El horizonte de P2 debe estar entre 1 y 23 horas.")
    df = df.copy()
    neto = df.groupby(level="estacion")["neto"]
    hora = df.index.get_level_values("hora")

    df["hora_dia"] = hora.hour
    df["dia_semana"] = hora.dayofweek
    df["fin_semana"] = (hora.dayofweek >= 5).astype("int8")

    df["salidas_h0"] = df["salidas"]
    df["llegadas_h0"] = df["llegadas"]
    df["neto_h0"] = df["neto"]
    df["neto_h1"] = neto.shift(1)
    df["neto_h2"] = neto.shift(2)
    df["neto_3h"] = df["neto_h0"] + df["neto_h1"] + df["neto_h2"]
    df["neto_3h_rel"] = df["neto_3h"] / df["capacidad"]

    df["neto_futuro_ayer"] = sum(neto.shift(24 - k) for k in range(1, horizonte + 1))
    df["neto_futuro_semana"] = sum(neto.shift(168 - k) for k in range(1, horizonte + 1))
    return df


def p2_ajustar_historico(train: pd.DataFrame) -> pd.DataFrame:
    """Tasa histórica de riesgo y neto futuro medio por estación × hora × fin de semana.

    Se ajusta SOLO con TRAIN y luego se aplica a validación y prueba con
    ``p2_aplicar_historico``. (P2 – Manuel Cardona.)
    """
    return (
        train.groupby(P2_LLAVES_HISTORICO)
        .agg(
            tasa_riesgo_hist=("riesgo", "mean"),
            neto_futuro_medio_hist=("neto_futuro", "mean"),
        )
        .reset_index()
    )


def p2_aplicar_historico(df: pd.DataFrame, historico: pd.DataFrame) -> pd.DataFrame:
    """Une a ``df`` el histórico ajustado en TRAIN (sin recalcularlo). (P2 – Manuel Cardona.)"""
    return (
        df.reset_index()
        .merge(historico, on=P2_LLAVES_HISTORICO, how="left")
        .set_index(["estacion", "hora"])
    )


def p2_historico_pasado(train: pd.DataFrame) -> pd.DataFrame:
    """Versión sin fuga del histórico dentro de TRAIN.

    Para cada fila usa solo las observaciones anteriores del mismo grupo
    (estación, hora del día, fin de semana), excluyendo la fila actual. Como
    las filas previas del grupo están al menos 24 h antes y ``H < 24``, sus
    etiquetas ya se conocían en ``t``. (P2 – Manuel Cardona.)
    """
    df = train.copy()
    grupos = df.groupby(
        [df.index.get_level_values("estacion"), df["hora_dia"], df["fin_semana"]]
    )
    previas = grupos.cumcount()
    for origen, destino in (
        ("riesgo", "tasa_riesgo_hist"),
        ("neto_futuro", "neto_futuro_medio_hist"),
    ):
        suma_previa = grupos[origen].cumsum() - df[origen]
        df[destino] = (suma_previa / previas).where(previas > 0)
    return df


# P1 (Johanna Camargo): viajes iniciados por estación y día, horizonte 1 día

P1_FEATURES = [
    "dia_semana",
    "fin_semana",
    "feriado",
    "lag_1",
    "lag_2",
    "lag_7",
    "lag_14",
    "media_7d",
    "media_28d",
    "media_mismo_dia_4sem",
    "dias_activa",
    "capacidad",
]


class LeakageError(ValueError):
    """Se intentó construir una variable con información posterior a la predicción."""


def p1_construir_panel_diario(viajes: pd.DataFrame, inicio: str, fin: str) -> pd.DataFrame:
    """Panel diario estación × fecha con la variable objetivo ``viajes``.

    ``viajes`` cuenta los viajes iniciados (salidas) en la estación ese día
    calendario. Los días sin salidas se rellenan con 0 y cada estación entra al
    panel a partir de su primera salida, de modo que el panel es contiguo y un
    ``shift(k)`` por estación equivale exactamente a "hace k días".
    (P1 – Johanna Camargo.)
    """
    conteo = (
        viajes.assign(fecha=viajes["inicio_viaje"].dt.normalize())
        .groupby(["origen_id", "fecha"])
        .size()
        .rename("viajes")
    )
    conteo.index.names = ["estacion", "fecha"]

    estaciones = sorted(conteo.index.get_level_values("estacion").unique())
    dias = pd.date_range(inicio, fin, freq="D", inclusive="left")
    indice = pd.MultiIndex.from_product([estaciones, dias], names=["estacion", "fecha"])
    panel = conteo.reindex(indice, fill_value=0).astype("int32").to_frame()

    primera = conteo.reset_index().groupby("estacion")["fecha"].min()
    estacion = panel.index.get_level_values("estacion")
    fecha = panel.index.get_level_values("fecha")
    return panel.loc[fecha >= estacion.map(primera)]


def p1_agregar_features(
    panel: pd.DataFrame,
    capacidad: pd.Series,
    feriados: list[str],
    lags: list[int],
    ventanas: list[int],
    horizonte: int = 1,
) -> pd.DataFrame:
    """Variables de P1 conocidas al cierre del día anterior al que se predice.

    * Calendario del día objetivo (conocido de antemano): ``dia_semana``,
      ``fin_semana`` y ``feriado``.
    * Rezagos de la demanda de la estación (``lag_k`` = viajes de hace k días)
      y medias móviles de 7 y 28 días que terminan en D-1 (``shift(1)``).
    * ``media_mismo_dia_4sem``: promedio de las 4 semanas previas en el mismo
      día de la semana (estacionalidad semanal).
    * ``dias_activa`` (antigüedad de la estación) y ``capacidad`` instalada.

    Lanza ``LeakageError`` si algún rezago es menor que el horizonte, porque
    ese dato aún no existiría cuando se emite la predicción.
    (P1 – Johanna Camargo.)
    """
    if min(lags) < horizonte:
        raise LeakageError(f"Rezago {min(lags)} < horizonte {horizonte}: fuga de información.")

    df = panel.copy()
    fecha = df.index.get_level_values("fecha")
    viajes = df.groupby(level="estacion")["viajes"]

    df["dia_semana"] = fecha.dayofweek.astype("int8")
    df["fin_semana"] = (fecha.dayofweek >= 5).astype("int8")
    df["feriado"] = fecha.isin(pd.to_datetime(feriados)).astype("int8")

    for k in lags:
        df[f"lag_{k}"] = viajes.shift(k)
    for w in ventanas:
        df[f"media_{w}d"] = viajes.transform(
            lambda s, w=w: s.shift(horizonte).rolling(w, min_periods=w).mean()
        )
    df["media_mismo_dia_4sem"] = sum(viajes.shift(7 * k) for k in range(1, 5)) / 4
    df["dias_activa"] = df.groupby(level="estacion").cumcount()
    df["capacidad"] = df.index.get_level_values("estacion").map(capacidad).fillna(0)
    return df
