"""Diagnóstico de unicidad de viajes sin modificar datos originales."""

import hashlib
import json

import pandas as pd

from mibici.config import (
    VIAJES_PARQUET,
    P1_MIN_DURATION_SECONDS,
    P1_MAX_DURATION_SECONDS,
)

from mibici.dataset import leer_csv, normalizar_viajes

from monitoring.config_2026 import (
    RAW,
    SALIDA,
    MANIFEST_DESCARGA,
    MESES,
)


CAMPOS = [
    "viaje_id",
    "inicio_viaje",
    "fin_viaje",
    "origen_id",
    "destino_id",
]

CAMPOS_EVENTO = CAMPOS[1:]


def sha256(ruta):

    with ruta.open("rb") as archivo:
        return hashlib.file_digest(
            archivo, "sha256"
        ).hexdigest()


def cargar_viajes_2026():

    manifest = json.loads(
        MANIFEST_DESCARGA.read_text()
    )

    assert manifest["descarga_completa"]
    assert set(manifest["archivos"]) == set(MESES)

    partes = []

    for mes in MESES:

        metadata = manifest["archivos"][mes]

        ruta = RAW / metadata["archivo"]

        assert sha256(ruta) == metadata["sha256"]

        datos = normalizar_viajes(
            leer_csv(ruta),
            origen=mes.replace("-", "_"),
        )

        assert set(CAMPOS) <= set(datos.columns)

        datos = datos[CAMPOS].copy()

        assert datos["viaje_id"].notna().all()

        datos["mes_archivo"] = mes

        duracion = (
            datos["fin_viaje"]
            - datos["inicio_viaje"]
        ).dt.total_seconds()

        origen = pd.to_numeric(
            datos["origen_id"],
            errors="coerce",
        )

        datos["valido_p1"] = (
            datos["inicio_viaje"].between(
                "2026-01-01",
                "2026-11-01",
                inclusive="left",
            )
            & duracion.between(
                P1_MIN_DURATION_SECONDS,
                P1_MAX_DURATION_SECONDS,
            )
            & origen.notna()
            & origen.gt(0)
            & origen.mod(1).eq(0)
        )

        partes.append(datos)

        print(
            f"{mes}: "
            f"{len(datos):,} registros | "
            f"{datos['viaje_id'].nunique():,} IDs únicos",
            flush=True,
        )

    return pd.concat(
        partes,
        ignore_index=True,
    )


def clasificar_repeticiones(viajes):

    repetidos = viajes.loc[
        viajes["viaje_id"].duplicated(keep=False)
    ].copy()

    grupos = []

    for _, grupo in repetidos.groupby(
        "viaje_id",
        sort=False,
    ):

        eventos = grupo.drop_duplicates(
            subset=CAMPOS_EVENTO
        )

        validos = grupo.loc[
            grupo["valido_p1"]
        ]

        eventos_validos = validos.drop_duplicates(
            subset=CAMPOS_EVENTO
        )

        meses = sorted(
            grupo["mes_archivo"].unique()
        )

        if len(eventos) == 1:

            tipo = "duplicado_operacional"

        else:

            tipo = "identificador_conflictivo"

        fila = {
            "tipo": tipo,
            "registros": len(grupo),
            "repeticiones_id": len(grupo) - 1,
            "eventos_distintos": len(eventos),
            "copias_operacionales_redundantes": (
                len(grupo) - len(eventos)
            ),
            "registros_validos_p1": len(validos),
            "copias_redundantes_validas_p1": (
                len(validos) - len(eventos_validos)
            ),
            "eventos_distintos_validos_p1": len(
                eventos_validos
            ),
            "cantidad_meses": len(meses),
            "meses": " / ".join(meses),
            "inicio_distinto": (
                grupo["inicio_viaje"]
                .nunique(dropna=False) > 1
            ),
            "fin_distinto": (
                grupo["fin_viaje"]
                .nunique(dropna=False) > 1
            ),
            "origen_distinto": (
                grupo["origen_id"]
                .nunique(dropna=False) > 1
            ),
            "destino_distinto": (
                grupo["destino_id"]
                .nunique(dropna=False) > 1
            ),
        }

        grupos.append(fila)

    return repetidos, pd.DataFrame(grupos)


def main():

    SALIDA.mkdir(
        parents=True,
        exist_ok=True,
    )

    viajes = cargar_viajes_2026()

    repetidos, grupos = clasificar_repeticiones(
        viajes
    )

    assert not grupos.empty

    repeticiones = int(
        viajes["viaje_id"].duplicated().sum()
    )

    assert (
        grupos["repeticiones_id"].sum()
        == repeticiones
    )

    print("\n========== RESUMEN GLOBAL ==========")

    print("Registros totales:", len(viajes))
    print("IDs únicos:", viajes["viaje_id"].nunique())
    print("Repeticiones excedentes:", repeticiones)
    print("Filas involucradas:", len(repetidos))
    print("IDs afectados:", len(grupos))

    print("\n========== CLASIFICACION ==========")

    columnas = [
        "registros",
        "repeticiones_id",
        "eventos_distintos",
        "copias_operacionales_redundantes",
        "registros_validos_p1",
        "copias_redundantes_validas_p1",
        "eventos_distintos_validos_p1",
    ]

    resumen = grupos.groupby("tipo").agg(
        ids_afectados=("tipo", "size"),
        **{
            columna: (columna, "sum")
            for columna in columnas
        },
    ).reset_index()

    print(resumen.to_string(index=False))

    resumen.to_csv(
        SALIDA / "clasificacion_duplicados_2026.csv",
        index=False,
    )

    print("\n========== DUPLICADOS POR MES ==========")

    mensual = []

    for mes, datos in viajes.groupby(
        "mes_archivo",
        sort=True,
    ):

        afectados = repetidos.loc[
            repetidos["mes_archivo"] == mes
        ]

        mensual.append({
            "mes": mes,
            "registros": len(datos),
            "ids_unicos": datos["viaje_id"].nunique(),
            "repeticiones_dentro_mes": int(
                datos["viaje_id"].duplicated().sum()
            ),
            "filas_afectadas_globalmente": len(
                afectados
            ),
            "filas_afectadas_validas_p1": int(
                afectados["valido_p1"].sum()
            ),
        })

    mensual = pd.DataFrame(mensual)

    print(mensual.to_string(index=False))

    mensual.to_csv(
        SALIDA / "duplicados_por_mes_2026.csv",
        index=False,
    )

    print("\n========== DISTRIBUCION ENTRE MESES ==========")

    por_periodo = (
        grupos.groupby(
            ["tipo", "meses"],
            as_index=False,
        )
        .agg(
            ids_afectados=("tipo", "size"),
            registros=("registros", "sum"),
            repeticiones=("repeticiones_id", "sum"),
        )
    )

    print(por_periodo.to_string(index=False))

    por_periodo.to_csv(
        SALIDA / "duplicados_entre_meses_2026.csv",
        index=False,
    )

    conflictos = grupos.loc[
        grupos["tipo"] == "identificador_conflictivo"
    ]

    print("\n========== NATURALEZA DE LOS CONFLICTOS ==========")

    diferencias = {
        campo: int(conflictos[campo].sum())
        for campo in [
            "inicio_distinto",
            "fin_distinto",
            "origen_distinto",
            "destino_distinto",
        ]
    }

    print(json.dumps(diferencias, indent=2))

    print("\n========== IMPACTO POTENCIAL EN P1 ==========")

    validos = viajes.loc[
        viajes["valido_p1"]
    ]

    repeticiones_validas = int(
        validos["viaje_id"].duplicated().sum()
    )

    redundancias_validas = int(
        grupos["copias_redundantes_validas_p1"].sum()
    )

    print("Viajes utilizables antes de tratamiento:", len(validos))

    print(
        "Repeticiones de ID entre viajes válidos:",
        repeticiones_validas,
    )

    print(
        "Copias operacionales redundantes válidas:",
        redundancias_validas,
    )

    print(
        "IDs conflictivos con más de un evento válido:",
        int(
            conflictos["eventos_distintos_validos_p1"]
            .gt(1)
            .sum()
        ),
    )

    print("\n========== CONTRASTE CON HISTORICO 2025 ==========")

    historico = pd.read_parquet(
        VIAJES_PARQUET,
        columns=CAMPOS,
    )

    historico_repetido = historico.loc[
        historico["viaje_id"].duplicated(keep=False)
        & historico["viaje_id"].notna()
    ]

    ids_2025 = pd.Index(
        historico["viaje_id"].dropna().unique()
    )

    ids_2026 = pd.Index(
        viajes["viaje_id"].unique()
    )

    comunes = ids_2025.intersection(ids_2026)

    historico_info = {
        "registros_2025": len(historico),
        "ids_unicos_2025": len(ids_2025),
        "ids_nulos_2025": int(
            historico["viaje_id"].isna().sum()
        ),
        "ids_repetidos_2025": int(
            historico_repetido["viaje_id"].nunique()
        ),
        "repeticiones_excedentes_2025": (
            len(historico_repetido)
            - int(
                historico_repetido["viaje_id"].nunique()
            )
        ),
        "copias_operacionales_redundantes_2025": int(
            historico_repetido.duplicated(
                subset=CAMPOS
            ).sum()
        ),
        "ids_compartidos_entre_2025_y_2026": len(
            comunes
        ),
        "historico_modificado": False,
    }

    print(json.dumps(historico_info, indent=2))

    resultado = {
        "periodo": "2026-01/2026-10",
        "registros_totales": len(viajes),
        "ids_unicos": int(
            viajes["viaje_id"].nunique()
        ),
        "repeticiones_excedentes": repeticiones,
        "filas_involucradas": len(repetidos),
        "ids_afectados": len(grupos),
        "ids_duplicados_operacionales": int(
            grupos["tipo"]
            .eq("duplicado_operacional")
            .sum()
        ),
        "ids_conflictivos": len(conflictos),
        "ids_repetidos_entre_meses": int(
            grupos["cantidad_meses"].gt(1).sum()
        ),
        "copias_operacionales_redundantes": int(
            grupos[
                "copias_operacionales_redundantes"
            ].sum()
        ),
        "copias_redundantes_validas_p1": (
            redundancias_validas
        ),
        "repeticiones_id_validas_p1": (
            repeticiones_validas
        ),
        "ids_conflictivos_con_varios_eventos_validos": int(
            conflictos["eventos_distintos_validos_p1"]
            .gt(1)
            .sum()
        ),
        "diferencias_en_conflictos": diferencias,
        "contraste_historico": historico_info,
        "registros_eliminados": 0,
        "tratamiento_aplicado": False,
        "nota": (
            "Identidad operacional definida por ID, "
            "inicio, fin, origen y destino. "
            "No implica identidad de todos los campos "
            "del CSV original. "
            "No se exportan identificadores individuales "
            "ni atributos personales."
        ),
    }

    (
        SALIDA / "diagnostico_duplicados_2026.json"
    ).write_text(
        json.dumps(
            resultado,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )

    print("\n========== DIAGNOSTICO DEFINITIVO ==========")

    print(
        json.dumps(
            resultado,
            indent=2,
            ensure_ascii=False,
        )
    )

    if len(conflictos) == 0:

        print(
            "\nRECOMENDACION: deduplicacion operacional "
            "documentada; conservar una copia por evento."
        )

    else:

        print(
            "\nRECOMENDACION: investigar los IDs "
            "conflictivos antes de aplicar tratamiento."
        )

    print("\nNo se eliminaron registros.")
    print("Evidencias agregadas guardadas en:", SALIDA)


if __name__ == "__main__":
    main()
