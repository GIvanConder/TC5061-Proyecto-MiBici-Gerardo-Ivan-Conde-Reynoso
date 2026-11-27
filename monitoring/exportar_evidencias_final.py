"""Regenera evidencias agregadas sin entrenamiento ni datos crudos."""

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from monitoring.config_2026 import SALIDA


def cargar(nombre):
    return json.loads(
        (SALIDA / nombre).read_text(encoding="utf-8")
    )


def exportar_tratamiento():

    manifest = cargar("manifest_2026.json")

    diagnostico = cargar(
        "diagnostico_duplicados_2026.json"
    )

    calidad = pd.read_csv(
        SALIDA / "calidad_ingesta_2026.csv"
    )

    originales = manifest["total_registros"]

    unicos = manifest["total_registros_unicos"]

    retiradas = manifest[
        "copias_operacionales_redundantes_retiradas"
    ]

    validos_retirados = diagnostico[
        "copias_redundantes_validas_p1"
    ]

    validos = manifest["viajes_utilizables_p1"]

    assert originales - retiradas == unicos

    assert retiradas == diagnostico[
        "copias_operacionales_redundantes"
    ]

    assert diagnostico["ids_conflictivos"] == 0
    assert manifest["ids_duplicados"] == 0

    assert manifest["csv_originales_modificados"] is False

    assert calidad["filas_originales"].sum() == originales
    assert calidad["filas_unicas"].sum() == unicos
    assert calidad["filas_utilizables_p1"].sum() == validos

    afectados = calidad.loc[
        calidad["copias_operacionales_redundantes"] > 0,
        "mes_archivo",
    ].tolist()

    resultado = {
        "registros_originales": originales,
        "copias_redundantes_retiradas": retiradas,
        "registros_unicos": unicos,
        "viajes_validos_antes_del_tratamiento": (
            validos + validos_retirados
        ),
        "viajes_validos_despues_del_tratamiento": validos,
        "copias_redundantes_validas_p1_retiradas": (
            validos_retirados
        ),
        "copias_redundantes_no_validas_p1_retiradas": (
            retiradas - validos_retirados
        ),
        "ids_conflictivos": diagnostico["ids_conflictivos"],
        "ids_duplicados_despues_del_tratamiento": (
            manifest["ids_duplicados"]
        ),
        "mes_afectado": " / ".join(afectados),
        "csv_originales_modificados": False,
        "tratamiento_aplicado": True,
        "resultado": "APROBADO",
    }

    ruta = SALIDA / "tratamiento_duplicados_2026.json"

    if ruta.exists():
        assert cargar(ruta.name) == resultado

    ruta.write_text(
        json.dumps(
            resultado,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )

    print("Tratamiento reproducido correctamente.")


def grafica_drift():

    tabla = pd.read_csv(
        SALIDA / "resumen_drift_variables_2026.csv"
    )

    tabla["multiplo_umbral"] = (
        tabla["valor_drift"] / tabla["umbral"]
    )

    tabla = tabla.sort_values("multiplo_umbral")

    colores = [
        "#d97706" if alerta else "#4263eb"
        for alerta in tabla["drift_detectado"]
    ]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.barh(
        tabla["variable"],
        tabla["multiplo_umbral"],
        color=colores,
    )

    ax.axvline(
        1,
        color="black",
        linestyle="--",
        label="Umbral Evidently",
    )

    ax.set_title("P1: drift de características en 2026")
    ax.set_xlabel("Distancia / umbral correspondiente")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        SALIDA / "04_drift_variables_2026.png",
        dpi=180,
    )

    plt.close(fig)


def grafica_politica():

    tabla = pd.read_csv(
        SALIDA / "politica_ct_mensual_2026.csv"
    )

    politica = cargar(
        "resultado_politica_ct_2026.json"
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    colores = [
        "#d97706" if alerta else "#4263eb"
        for alerta in tabla["alerta_desempeno"]
    ]

    ax.bar(tabla["periodo"], tabla["mae"], color=colores)

    ax.axhline(
        politica["umbral_mae"],
        linestyle="--",
        color="#b91c1c",
        label="Umbral inicial: +15% MAE",
    )

    ax.axhline(
        tabla["mae_referencia"].iloc[0],
        linestyle=":",
        color="#374151",
        label="MAE TEST 2025",
    )

    for i, fila in tabla.iterrows():

        if fila["nueva_solicitud_evaluacion"]:

            ax.annotate(
                "Evaluar candidato",
                xy=(i, fila["mae"]),
                xytext=(i - 2.2, fila["mae"] + 0.6),
                arrowprops={"arrowstyle": "->"},
                fontsize=9,
            )

    ax.set_title("Política inicial de Continuous Training P1")
    ax.set_ylabel("MAE (viajes por estación-día)")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    ax.set_ylim(0, max(tabla["mae"]) * 1.22)

    fig.tight_layout()

    fig.savefig(
        SALIDA / "05_politica_continuous_training_2026.png",
        dpi=180,
    )

    plt.close(fig)


def main():

    exportar_tratamiento()

    grafica_drift()
    grafica_politica()

    print("Evidencias complementarias regeneradas.")
    print("Sin entrenamiento ni acceso al Registry.")


if __name__ == "__main__":
    main()
