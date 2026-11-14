"""Verifica y exporta evidencias reproducibles del A/B P1."""

import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from monitoring.comparar_ab_p1 import bootstrap_pareado

ORIGEN = Path("reports/p1/drift_abtest")
DESTINO = Path("docs/evidencias/ab_p1")


def verificar_consistencia(resumen, diario, mensual):
    """Comprueba que JSON y CSV representan la misma evaluación."""
    if resumen["tipo"] != "comparacion_ab_retrospectiva_pareada":
        raise ValueError("Tipo de experimento incorrecto")

    if resumen["aleatorizacion_en_produccion"] is not False:
        raise ValueError("No debe presentarse como A/B aleatorizado")

    if str(resumen["champion"]["version"]) != "1":
        raise ValueError("Champion inesperado")

    if str(resumen["challenger"]["version"]) != "2":
        raise ValueError("Challenger inesperado")

    diario["fecha"] = pd.to_datetime(diario["fecha"])

    fechas_esperadas = pd.date_range(
        "2025-11-01", "2025-12-31", freq="D"
    )
    if not diario["fecha"].equals(
        pd.Series(fechas_esperadas, name="fecha")
    ):
        raise ValueError("Fechas diarias incompletas o desordenadas")

    if len(diario) != resumen["dias"]:
        raise ValueError("Número de días inconsistente")

    if int(diario["n"].sum()) != resumen["observaciones"]:
        raise ValueError("Número de observaciones inconsistente")

    if diario.isna().any().any():
        raise ValueError("Resumen diario con valores faltantes")

    if (diario[["n", "abs_a", "abs_b"]] < 0).any().any():
        raise ValueError("Resumen diario con valores negativos")

    n = diario["n"].sum()
    mae_a = diario["abs_a"].sum() / n
    mae_b = diario["abs_b"].sum() / n

    for calculado, etiqueta in (
        (mae_a, "champion"),
        (mae_b, "challenger"),
    ):
        registrado = resumen[etiqueta]["metricas"]["mae"]
        if not math.isclose(
            calculado, registrado, rel_tol=0, abs_tol=1e-9
        ):
            raise ValueError(f"MAE inconsistente: {etiqueta}")

    delta = mae_a - mae_b
    if not math.isclose(
        delta,
        resumen["diferencia_mae_champion_menos_challenger"],
        rel_tol=0,
        abs_tol=1e-9,
    ):
        raise ValueError("Diferencia global inconsistente")

    if list(mensual["mes"]) != ["2025-11", "2025-12"]:
        raise ValueError("Meses de evaluación inesperados")

    for _, mes in mensual.iterrows():
        periodo = diario[
            diario["fecha"].dt.strftime("%Y-%m") == mes["mes"]
        ]
        if int(periodo["n"].sum()) != int(mes["filas"]):
            raise ValueError(f"Filas incompatibles: {mes['mes']}")

        for columna, salida in (
            ("abs_a", "mae_champion"),
            ("abs_b", "mae_challenger"),
        ):
            calculado = periodo[columna].sum() / periodo["n"].sum()
            if not np.isclose(
                calculado, mes[salida], atol=1e-9, rtol=0
            ):
                raise ValueError(
                    f"MAE mensual inconsistente: {mes['mes']}"
                )

    return float(delta)


def guardar_graficas(resumen, diario, mensual, sensibilidad):
    plt.rcParams.update({
        "font.size": 10,
        "figure.dpi": 150,
        "savefig.dpi": 180,
    })

    # Figura 1: comparación mensual.
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    posiciones = np.arange(len(mensual))
    ancho = 0.35

    ax.bar(
        posiciones - ancho / 2,
        mensual["mae_champion"],
        ancho,
        label="Champion v1",
    )
    ax.bar(
        posiciones + ancho / 2,
        mensual["mae_challenger"],
        ancho,
        label="Challenger v2",
    )

    ax.set_xticks(posiciones, ["Noviembre", "Diciembre"])
    ax.set_ylabel("MAE (viajes por estación-día)")
    ax.set_title("Desempeño mensual en TEST — MiBici P1")
    ax.set_ylim(
        0,
        max(
            mensual["mae_champion"].max(),
            mensual["mae_challenger"].max(),
        ) * 1.18,
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    for grupo in ax.containers:
        ax.bar_label(grupo, fmt="%.3f", padding=3)

    fig.tight_layout()
    fig.savefig(DESTINO / "ab_mae_mensual.png")
    plt.close(fig)

    # Figura 2: diferencia diaria y tendencia de 7 días.
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(
        diario["fecha"],
        diario["mejora_mae"],
        alpha=0.45,
        label="Diferencia diaria",
    )
    ax.plot(
        diario["fecha"],
        diario["mejora_mae"].rolling(
            7, min_periods=1
        ).mean(),
        linewidth=2,
        label="Media móvil de 7 días",
    )
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_ylabel("MAE champion − MAE challenger")
    ax.set_title("Diferencia diaria pareada — MiBici P1")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(DESTINO / "ab_diferencia_diaria.png")
    plt.close(fig)

    # Figura 3: sensibilidad al tamaño de bloque.
    fig, ax = plt.subplots(figsize=(8.5, 4.1))
    delta = resumen[
        "diferencia_mae_champion_menos_challenger"
    ]

    for i, resultado in enumerate(sensibilidad):
        inferior = resultado["ic95_inferior"]
        superior = resultado["ic95_superior"]
        ax.plot(
            [inferior, superior], [i, i],
            linewidth=3, solid_capstyle="round",
        )
        ax.scatter([delta], [i], zorder=3)

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(
        range(len(sensibilidad)),
        [
            f"{r['bloque_dias']} días"
            for r in sensibilidad
        ],
    )
    ax.invert_yaxis()
    ax.set_xlabel("MAE champion − MAE challenger")
    ax.set_title("Sensibilidad del IC95% bootstrap")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(DESTINO / "ab_sensibilidad_bootstrap.png")
    plt.close(fig)


def main():
    resumen = json.loads(
        (ORIGEN / "comparacion_ab_p1.json").read_text(
            encoding="utf-8"
        )
    )
    diario = pd.read_csv(
        ORIGEN / "comparacion_ab_diaria.csv"
    )
    mensual = pd.read_csv(
        ORIGEN / "comparacion_ab_mensual.csv"
    )

    delta = verificar_consistencia(
        resumen, diario, mensual
    )

    sensibilidad = []
    for bloque in (3, 7, 14):
        intervalo = bootstrap_pareado(
            diario,
            repeticiones=3000,
            bloque=bloque,
            semilla=42,
        )
        sensibilidad.append(intervalo)

    intervalo_7 = next(
        item for item in sensibilidad
        if item["bloque_dias"] == 7
    )
    original = resumen["bootstrap"]

    for clave in ("ic95_inferior", "ic95_superior"):
        if not math.isclose(
            intervalo_7[clave], original[clave],
            abs_tol=1e-9, rel_tol=0,
        ):
            raise ValueError(
                f"No se reproduce bootstrap: {clave}"
            )

    DESTINO.mkdir(parents=True, exist_ok=True)

    archivos = (
        "comparacion_ab_p1.json",
        "comparacion_ab_diaria.csv",
        "comparacion_ab_mensual.csv",
    )
    for archivo in archivos:
        shutil.copy2(
            ORIGEN / archivo,
            DESTINO / archivo,
        )

    (DESTINO / "sensibilidad_bootstrap.json").write_text(
        json.dumps(
            sensibilidad,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    guardar_graficas(
        resumen, diario, mensual, sensibilidad
    )

    a = resumen["champion"]["metricas"]
    b = resumen["challenger"]["metricas"]
    mejor = (
        "Conservar champion v1"
        if not resumen["regla_promocion_cumplida"]
        else "Revisión humana requerida antes de promover"
    )

    tabla_sensibilidad = "\n".join(
        f"| {r['bloque_dias']} | "
        f"{r['ic95_inferior']:.6f} | "
        f"{r['ic95_superior']:.6f} |"
        for r in sensibilidad
    )

    contenido = f"""# Evaluación A/B retrospectiva — MiBici P1

## Objetivo

Comparar las versiones registradas de P1 sobre exactamente
las mismas observaciones de noviembre–diciembre de 2025.

Esta prueba es retrospectiva y pareada. No es un experimento
online aleatorizado ni mide impacto causal en producción.

## Versiones

- **Champion v1:** run `{resumen['champion']['run_id']}`.
- **Challenger v2:** run `{resumen['challenger']['run_id']}`.
- **Versión de datos:** `{resumen['data_version']}`.
- **Observaciones:** {resumen['observaciones']:,}.
- **Días:** {resumen['dias']}.

Ambos modelos utilizan el Pipeline original Poisson.
El challenger incorpora octubre al entrenamiento y mantiene
los hiperparámetros originales.

## Resultados globales

| Métrica | Champion v1 | Challenger v2 |
|---|---:|---:|
| MAE | {a['mae']:.6f} | {b['mae']:.6f} |
| RMSE | {a['rmse']:.6f} | {b['rmse']:.6f} |
| WAPE | {a['wape']:.6f} | {b['wape']:.6f} |
| R² | {a['r2']:.6f} | {b['r2']:.6f} |
| Sesgo | {a['sesgo']:.6f} | {b['sesgo']:.6f} |

Diferencia MAE (champion menos challenger): **{delta:.6f}**.

Mejora relativa del challenger:
**{resumen['mejora_relativa_mae_pct']:.3f}%**.

Días con mejor MAE del challenger:
**{resumen['dias_con_mejora_challenger']} de {resumen['dias']}**.

## Resultados mensuales

![Comparación MAE por mes](ab_mae_mensual.png)

![Diferencia diaria pareada](ab_diferencia_diaria.png)

## Incertidumbre y sensibilidad

Se utiliza bootstrap pareado circular con bloques
temporales, 3,000 réplicas y semilla 42.

La especificación principal utiliza bloques de siete días.
Las otras longitudes constituyen un análisis exploratorio
de sensibilidad y no sustituyen una validación futura.

| Bloque (días) | IC95% inferior | IC95% superior |
|---|---:|---:|
{tabla_sensibilidad}

![Sensibilidad bootstrap](ab_sensibilidad_bootstrap.png)

El intervalo representa la diferencia
`MAE champion - MAE challenger`; valores negativos
favorecen al champion.

El bootstrap es aproximado y no elimina todas las posibles
dependencias entre estaciones y periodos.

## Decisión

**{mejor}.**

El criterio predefinido exige mejora global del MAE,
límite inferior del IC95% superior a cero y ausencia
de regresiones mensuales, además de revisión humana.

No se ha cambiado ningún alias en MLflow.

## Reproducibilidad

Desde la raíz del repositorio, con MLflow accesible
en `http://127.0.0.1:5000`:

```bash
./.venv/bin/python -m monitoring.preparar_p1
./.venv/bin/python -m monitoring.comparar_ab_p1
./.venv/bin/python -m monitoring.exportar_evidencias_ab_p1
```

La evaluación consulta el Model Registry y utiliza
el conjunto TEST únicamente para comparación final.
No se deben ajustar hiperparámetros retrospectivamente
en función de estos resultados.

Los CSV incluidos son agregados diarios y mensuales;
no contienen las observaciones individuales originales.
"""

    (DESTINO / "README.md").write_text(
        contenido,
        encoding="utf-8",
    )

    print("=== CONSISTENCIA VERIFICADA ===")
    print("Observaciones:", resumen["observaciones"])
    print("Días:", resumen["dias"])
    print("Diferencia MAE:", round(delta, 6))

    print("\n=== SENSIBILIDAD BOOTSTRAP ===")
    for item in sensibilidad:
        print(
            f"Bloque {item['bloque_dias']:2d} días:",
            f"[{item['ic95_inferior']:.6f},",
            f"{item['ic95_superior']:.6f}]",
        )

    print("\n=== DECISIÓN ===")
    print(mejor)
    print("\n=== ARCHIVOS EXPORTADOS ===")
    for archivo in sorted(DESTINO.iterdir()):
        print(archivo.name, archivo.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
