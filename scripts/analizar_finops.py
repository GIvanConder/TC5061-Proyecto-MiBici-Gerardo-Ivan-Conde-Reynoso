"""Estimacion FinOps basada en mediciones experimentales reales."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/evidencias/operacion_finops"
LOCUST = BASE / "locust"
SALIDA = BASE / "finops"

DIAS_MES = 30
SEGUNDOS_MES = DIAS_MES * 24 * 3600

# SLO exploratorio, propuesto despues de los experimentos.
P95_OBJETIVO_MS = 200
ERROR_MAX_PCT = 0.1

UTILIZACIONES = [0.05, 0.10, 0.25, 0.50, 0.75, 1.0]


def costo_unitario(costo_mes, rps, utilizacion):
    if costo_mes < 0 or rps <= 0:
        raise ValueError("Costo o rendimiento invalido")

    if not 0 < utilizacion <= 1:
        raise ValueError("Utilizacion fuera de rango")

    inferencias = (
        SEGUNDOS_MES * rps * utilizacion
    )

    return costo_mes / inferencias


def cargar_experimento(archivo, workers):
    run = (LOCUST / archivo).read_text().strip()
    ruta = LOCUST / run

    rendimiento = pd.read_csv(
        ruta / "resumen_rendimiento.csv"
    )

    recursos = pd.read_csv(
        ruta / "resumen_recursos.csv"
    )

    assert set(rendimiento["usuarios"]) == {5, 20, 50}

    assert (
        rendimiento["solicitudes"]
        == rendimiento["exitos"] + rendimiento["fallos"]
    ).all()

    filas = []

    for _, dato in rendimiento.iterrows():
        escenario = dato["escenario"]

        api = recursos[
            (recursos["escenario"] == escenario)
            & (recursos["servicio"] == "API")
        ]

        mlflow = recursos[
            (recursos["escenario"] == escenario)
            & (recursos["servicio"] == "MLflow")
        ]

        assert len(api) == 1
        assert len(mlflow) == 1

        api = api.iloc[0]
        mlflow = mlflow.iloc[0]

        filas.append({
            "run": run,
            "workers": workers,
            "escenario": escenario,
            "usuarios": int(dato["usuarios"]),
            "solicitudes": int(dato["solicitudes"]),
            "exitos": int(dato["exitos"]),
            "fallos": int(dato["fallos"]),
            "tasa_error_pct": float(dato["tasa_error_pct"]),
            "rps_exitosas": float(dato["rps_exitosas"]),
            "p95_ms": float(dato["p95_ms"]),
            "p99_ms": float(dato["p99_ms"]),
            "cumple_slo_propuesto": bool(
                dato["p95_ms"] <= P95_OBJETIVO_MS
                and dato["tasa_error_pct"] <= ERROR_MAX_PCT
            ),
            "api_cpu_media_pct": float(api["cpu_media_pct"]),
            "api_cpu_max_pct": float(api["cpu_max_pct"]),
            "mlflow_cpu_media_pct": float(mlflow["cpu_media_pct"]),
            "mlflow_cpu_max_pct": float(mlflow["cpu_max_pct"]),
            "api_ram_media_mb": float(api["memoria_media_mb"]),
            "mlflow_ram_media_mb": float(mlflow["memoria_media_mb"]),
            "ram_conjunta_media_mb": float(
                api["memoria_media_mb"]
                + mlflow["memoria_media_mb"]
            ),
            "ram_conjunta_limite_superior_mb": float(
                api["memoria_max_mb"]
                + mlflow["memoria_max_mb"]
            ),
        })

    return filas


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)

    tarifas = json.loads(
        (SALIDA / "precios_referencia.json").read_text()
    )

    paquetes = {
        p["vcpu"]: p
        for p in tarifas["paquetes"]
    }

    registros = (
        cargar_experimento("ejecucion_original.txt", 1)
        + cargar_experimento("ejecucion_optimizada.txt", 2)
    )

    mediciones = pd.DataFrame(registros)

    assert len(mediciones) == 6
    assert (mediciones["rps_exitosas"] > 0).all()

    mediciones.to_csv(
        SALIDA / "mediciones_consolidadas.csv",
        index=False,
        float_format="%.6f",
    )

    # Configuraciones candidatas, no benchmarks de AWS.
    planes = [
        ("1w_media_2vcpu", 1, 20, 2),
        ("2w_media_2vcpu", 2, 20, 2),
        ("1w_media_4vcpu", 1, 20, 4),
        ("2w_alta_4vcpu", 2, 50, 4),
    ]

    filas = []

    for nombre, workers, usuarios, vcpu in planes:
        seleccion = mediciones[
            (mediciones["workers"] == workers)
            & (mediciones["usuarios"] == usuarios)
        ]

        assert len(seleccion) == 1

        m = seleccion.iloc[0]
        p = paquetes[vcpu]

        cpu_conjunta = (
            m["api_cpu_media_pct"]
            + m["mlflow_cpu_media_pct"]
        )

        cpu_picos = (
            m["api_cpu_max_pct"]
            + m["mlflow_cpu_max_pct"]
        )

        for u in UTILIZACIONES:
            costo = costo_unitario(
                p["precio_usd_mes"],
                m["rps_exitosas"],
                u,
            )

            filas.append({
                "plan": nombre,
                "workers": workers,
                "usuarios": usuarios,
                "vcpu": vcpu,
                "ram_instancia_gb": p["ram_gb"],
                "precio_usd_mes": p["precio_usd_mes"],
                "precio_equivalente_usd_hora": (
                    p["precio_usd_mes"] / (DIAS_MES * 24)
                ),
                "utilizacion_efectiva": u,
                "rps_referencia": m["rps_exitosas"],
                "p95_ms": m["p95_ms"],
                "cumple_slo_propuesto": m["cumple_slo_propuesto"],
                "cpu_conjunta_media_pct_local": cpu_conjunta,
                "cpu_conjunta_max_cota_pct_local": cpu_picos,
                "cpu_media_pct_capacidad_candidata": (
                    cpu_conjunta / vcpu
                ),
                "ram_conjunta_media_mb": m["ram_conjunta_media_mb"],
                "ram_conjunta_max_cota_mb": (
                    m["ram_conjunta_limite_superior_mb"]
                ),
                "inferencias_mes_estimadas": (
                    SEGUNDOS_MES * m["rps_exitosas"] * u
                ),
                "costo_usd_inferencia": costo,
                "costo_usd_millon": costo * 1_000_000,
            })

    escenarios = pd.DataFrame(filas)

    assert len(escenarios) == 24

    escenarios.to_csv(
        SALIDA / "escenarios_finops.csv",
        index=False,
        float_format="%.12f",
    )

    # Comparacion a la misma demanda mensual.
    demanda = []

    for volumen in [100_000, 1_000_000, 10_000_000]:
        for p in paquetes.values():
            demanda.append({
                "inferencias_mes": volumen,
                "vcpu": p["vcpu"],
                "precio_usd_mes": p["precio_usd_mes"],
                "costo_usd_inferencia": (
                    p["precio_usd_mes"] / volumen
                ),
                "costo_usd_millon": (
                    p["precio_usd_mes"] / volumen * 1_000_000
                ),
            })

    pd.DataFrame(demanda).to_csv(
        SALIDA / "costo_demanda_fija.csv",
        index=False,
        float_format="%.12f",
    )

    resumen = escenarios[
        escenarios["utilizacion_efectiva"] == 0.50
    ].copy()

    resultado = {
        "tipo": "Estimacion, no facturacion real",
        "dias_mes": DIAS_MES,
        "slo": {
            "caracter": "Exploratorio y posterior al experimento",
            "p95_max_ms": P95_OBJETIVO_MS,
            "tasa_error_max_pct": ERROR_MAX_PCT,
        },
        "solicitudes_evaluadas": int(
            mediciones["solicitudes"].sum()
        ),
        "inferencias_exitosas": int(
            mediciones["exitos"].sum()
        ),
        "fallos": int(mediciones["fallos"].sum()),
        "escenarios_utilizacion_50_pct": (
            resumen.to_dict("records")
        ),
        "exclusiones": [
            "Impuestos",
            "Trabajo humano",
            "Alta disponibilidad",
            "Respaldos adicionales",
            "Transferencias fuera del paquete",
            "Costos de entrenamiento y CI/CD",
        ],
        "limitacion": (
            "Benchmark local WSL2; rendimiento de AWS "
            "no validado experimentalmente."
        ),
    }

    (SALIDA / "resumen_finops.json").write_text(
        json.dumps(
            resultado,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for nombre, datos in escenarios.groupby("plan"):
        ax.plot(
            datos["utilizacion_efectiva"] * 100,
            datos["costo_usd_millon"],
            marker="o",
            label=nombre,
        )

    ax.set_title("Sensibilidad del costo por inferencia")
    ax.set_xlabel("Utilizacion efectiva (%)")
    ax.set_ylabel("USD por millon de inferencias")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(
        SALIDA / "10_sensibilidad_costos.png",
        dpi=220,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.bar(
        resumen["plan"],
        resumen["costo_usd_millon"],
    )

    ax.set_title("Costo unitario al 50% de utilizacion")
    ax.set_ylabel("USD por millon de inferencias")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        SALIDA / "11_costo_unitario.png",
        dpi=220,
    )
    plt.close(fig)

    print("========== MEDICIONES CONSOLIDADAS ==========")

    print(mediciones[[
        "workers",
        "usuarios",
        "exitos",
        "fallos",
        "rps_exitosas",
        "p95_ms",
        "cumple_slo_propuesto",
    ]].to_string(index=False))

    print("\n========== FINOPS AL 50% ==========")

    print(resumen[[
        "plan",
        "precio_usd_mes",
        "rps_referencia",
        "costo_usd_millon",
        "cpu_media_pct_capacidad_candidata",
        "ram_conjunta_media_mb",
    ]].round(6).to_string(index=False))

    print("\nInferencias exitosas:", resultado["inferencias_exitosas"])
    print("Fallos:", resultado["fallos"])

    print(f"\nEvidencias guardadas en {SALIDA}")


if __name__ == "__main__":
    main()
