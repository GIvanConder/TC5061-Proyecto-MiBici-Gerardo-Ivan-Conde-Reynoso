"""Comparación A/B retrospectiva y pareada de P1.

Evalúa champion y challenger sobre el MISMO TEST.
No entrena, no modifica el Registry y no cambia alias.
"""
import argparse
import json
import math
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mibici.features import P1_FEATURES

BASE = Path("reports/p1/drift_abtest")
URI = "http://127.0.0.1:5000"


def validar_test(datos, info):
    requeridas = {"estacion", "fecha", "viajes_real", *P1_FEATURES}
    faltantes = requeridas - set(datos.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas: {faltantes}")

    df = datos.copy()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="raise")

    if df.empty:
        raise ValueError("TEST vacío")

    if not df["fecha"].between(
        "2025-11-01", "2026-01-01", inclusive="left"
    ).all():
        raise ValueError("TEST contiene fechas fuera del periodo")

    if df.duplicated(["estacion", "fecha"]).any():
        raise ValueError("TEST contiene estación-fecha duplicada")

    if len(df) != info["filas"]["test"]:
        raise ValueError("El tamaño de TEST no coincide con el manifiesto")

    x = df[P1_FEATURES].astype("float64")
    y = df["viajes_real"].astype("float64")

    if not np.isfinite(x.to_numpy()).all():
        raise ValueError("TEST contiene variables no finitas")

    if not np.isfinite(y.to_numpy()).all() or (y < 0).any():
        raise ValueError("TEST contiene objetivos inválidos")

    return df.sort_values(["fecha", "estacion"]).reset_index(drop=True)


def metricas(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)

    if y.shape != pred.shape or not y.size:
        raise ValueError("Predicciones y objetivos incompatibles")

    if not np.isfinite(pred).all() or (pred < 0).any():
        raise ValueError("Predicciones inválidas")

    if np.sum(y) <= 0:
        raise ValueError("No se puede calcular WAPE")

    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "wape": float(np.abs(y - pred).sum() / y.sum()),
        "r2": float(r2_score(y, pred)),
        "sesgo": float(np.mean(pred - y)),
    }


def bootstrap_pareado(diario, repeticiones=3000, bloque=7, semilla=42):
    """IC percentil para MAE_A - MAE_B, remuestreando bloques de días."""
    if repeticiones < 100 or bloque < 1:
        raise ValueError("Configuración bootstrap insuficiente")

    diario = diario.sort_values("fecha").reset_index(drop=True)
    n = len(diario)

    if n < bloque:
        raise ValueError("No hay días suficientes para el bloque")

    columnas = ["n", "abs_a", "abs_b"]
    valores = diario[columnas].to_numpy(dtype=float)
    if not np.isfinite(valores).all() or (valores[:, 0] <= 0).any():
        raise ValueError("Resumen diario inválido")

    rng = np.random.default_rng(semilla)
    diferencias = np.empty(repeticiones)

    for i in range(repeticiones):
        inicios = rng.integers(
            0, n, size=math.ceil(n / bloque)
        )
        indices = (
            inicios[:, None] + np.arange(bloque)
        ) % n
        muestra = valores[indices.ravel()[:n]]

        diferencias[i] = (
            (muestra[:, 1].sum() - muestra[:, 2].sum())
            / muestra[:, 0].sum()
        )

    inferior, superior = np.quantile(
        diferencias, [0.025, 0.975]
    )
    return {
        "ic95_inferior": float(inferior),
        "ic95_superior": float(superior),
        "repeticiones": repeticiones,
        "bloque_dias": bloque,
        "semilla": semilla,
    }


def ejecutar(repeticiones=3000):
    info = json.loads(
        (BASE / "particiones.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        Path("data/manifest.json").read_text(encoding="utf-8")
    )
    if info["data_version"] != manifest["data_version"]:
        raise ValueError("Versión de datos incompatible")
    if info["variables"] != P1_FEATURES:
        raise ValueError("Orden de variables incompatible")

    test = validar_test(
        pd.read_parquet(BASE / "test.parquet"), info
    )

    mlflow.set_tracking_uri(URI)
    mlflow.set_registry_uri(URI)
    client = MlflowClient()

    nombre = info["modelo"]
    a = client.get_model_version_by_alias(nombre, "champion")
    b = client.get_model_version_by_alias(nombre, "challenger")

    if int(a.version) != int(info["version_champion"]):
        raise ValueError("Cambió la versión de referencia")
    if a.version == b.version:
        raise ValueError("Los alias apuntan a la misma versión")
    if b.tags.get("base_run_id") != a.run_id:
        raise ValueError("Linaje del challenger incorrecto")

    run_a = client.get_run(a.run_id)
    run_b = client.get_run(b.run_id)

    if run_b.data.tags.get("test_used_for_training") != "false":
        raise ValueError("No se acreditó la separación temporal")
    if run_b.data.tags.get("data_version") != info["data_version"]:
        raise ValueError("Versión de datos del challenger incompatible")

    x = test[P1_FEATURES].astype("float64")
    y = test["viajes_real"].to_numpy(dtype=float)

    print("=== RECUPERAR VERSIONES DESDE MLFLOW ===")
    modelos = {}
    for etiqueta, version in (("champion", a), ("challenger", b)):
        modelo = mlflow.sklearn.load_model(
            f"models:/{nombre}/{version.version}"
        )
        if list(modelo.feature_names_in_) != P1_FEATURES:
            raise ValueError(f"Esquema incompatible: {etiqueta}")
        modelos[etiqueta] = modelo
        print(
            etiqueta,
            "version=", version.version,
            "run=", version.run_id
        )

    pred_a = np.asarray(modelos["champion"].predict(x))
    pred_b = np.asarray(modelos["challenger"].predict(x))

    resultado_a = metricas(y, pred_a)
    resultado_b = metricas(y, pred_b)

    mae_original = float(run_a.data.metrics["test_mae"])
    if abs(resultado_a["mae"] - mae_original) > 1e-4:
        raise ValueError("No se reproduce el MAE original del champion")

    observaciones = pd.DataFrame({
        "fecha": test["fecha"],
        "real": y,
        "pred_a": pred_a,
        "pred_b": pred_b,
        "abs_a": np.abs(y - pred_a),
        "abs_b": np.abs(y - pred_b),
    })

    diario = observaciones.groupby(
        "fecha", as_index=False
    ).agg(
        n=("real", "size"),
        abs_a=("abs_a", "sum"),
        abs_b=("abs_b", "sum"),
    )
    diario["mae_a"] = diario["abs_a"] / diario["n"]
    diario["mae_b"] = diario["abs_b"] / diario["n"]
    diario["mejora_mae"] = diario["mae_a"] - diario["mae_b"]

    ic = bootstrap_pareado(
        diario, repeticiones=repeticiones
    )

    meses = []
    for mes, grupo in observaciones.groupby(
        observaciones["fecha"].dt.to_period("M"),
        sort=True,
    ):
        ma = metricas(grupo["real"], grupo["pred_a"])
        mb = metricas(grupo["real"], grupo["pred_b"])
        meses.append({
            "mes": str(mes),
            "filas": int(len(grupo)),
            "mae_champion": ma["mae"],
            "mae_challenger": mb["mae"],
            "diferencia_mae": ma["mae"] - mb["mae"],
            "wape_champion": ma["wape"],
            "wape_challenger": mb["wape"],
            "sesgo_champion": ma["sesgo"],
            "sesgo_challenger": mb["sesgo"],
        })

    delta = resultado_a["mae"] - resultado_b["mae"]
    mejora_relativa = 100 * delta / resultado_a["mae"]
    meses_sin_regresion = all(
        m["diferencia_mae"] >= 0 for m in meses
    )

    # Regla declarada antes de inspeccionar resultados.
    # Es orientativa: la promoción requiere revisión humana.
    evidencia_favorable = (
        delta > 0
        and ic["ic95_inferior"] > 0
        and meses_sin_regresion
    )

    resumen = {
        "tipo": "comparacion_ab_retrospectiva_pareada",
        "aleatorizacion_en_produccion": False,
        "data_version": info["data_version"],
        "modelo": nombre,
        "periodo": "2025-11-01/2025-12-31",
        "observaciones": int(len(test)),
        "dias": int(len(diario)),
        "champion": {
            "version": a.version,
            "run_id": a.run_id,
            "metricas": resultado_a,
        },
        "challenger": {
            "version": b.version,
            "run_id": b.run_id,
            "metricas": resultado_b,
        },
        "diferencia_mae_champion_menos_challenger": delta,
        "mejora_relativa_mae_pct": mejora_relativa,
        "bootstrap": ic,
        "comparacion_mensual": meses,
        "dias_con_mejora_challenger": int(
            (diario["mejora_mae"] > 0).sum()
        ),
        "regla_promocion_cumplida": bool(evidencia_favorable),
        "nota": (
            "Bootstrap de bloques de 7 días, IC aproximado. "
            "No equivale a un A/B online aleatorizado. "
            "TEST no se utilizó para ajustar hiperparámetros."
        ),
    }

    BASE.mkdir(parents=True, exist_ok=True)
    (BASE / "comparacion_ab_p1.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    diario.to_csv(BASE / "comparacion_ab_diaria.csv", index=False)
    pd.DataFrame(meses).to_csv(
        BASE / "comparacion_ab_mensual.csv", index=False
    )

    print("\n=== RESULTADOS GLOBALES ===")
    for etiqueta, m in (
        ("CHAMPION", resultado_a),
        ("CHALLENGER", resultado_b),
    ):
        print(etiqueta)
        for k, v in m.items():
            print(f"  {k}: {v:.6f}")

    print("\n=== DIFERENCIA PAREADA ===")
    print("MAE champion - challenger:", round(delta, 6))
    print("Mejora relativa (%):", round(mejora_relativa, 3))
    print(
        "IC95% bootstrap por bloques:",
        round(ic["ic95_inferior"], 6),
        "a",
        round(ic["ic95_superior"], 6),
    )

    print("\n=== COMPARACION MENSUAL ===")
    print(pd.DataFrame(meses).to_string(index=False))

    print("\nDías con menor MAE del challenger:",
          resumen["dias_con_mejora_challenger"],
          "de", len(diario))
    print("Regla de promoción cumplida:", evidencia_favorable)
    print("Alias MLflow: SIN MODIFICACIONES")
    print("OK: comparación guardada")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeticiones", type=int, default=3000)
    args = parser.parse_args()
    ejecutar(repeticiones=args.repeticiones)
