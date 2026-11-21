"""Smoke test de servicios y equivalencia del Pipeline P3."""
import json
import os
import time
from pathlib import Path
from urllib.request import Request, urlopen

import mlflow.pyfunc
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
API = f"http://127.0.0.1:{os.getenv('MIBICI_API_PORT', '8000')}"
MLFLOW = f"http://127.0.0.1:{os.getenv('MIBICI_MLFLOW_PORT', '5000')}"

entrada = {
    "salidas_dia": 20.0,
    "llegadas_dia": 19.0,
    "balance_relativo": 1.0 / 39.0,
    "share_fin_semana": 0.25,
    "salida_manana_share": 0.25,
    "llegada_manana_share": 0.25,
    "salida_mediodia_share": 0.25,
    "llegada_mediodia_share": 0.25,
    "salida_tarde_share": 0.25,
    "llegada_tarde_share": 0.25,
    "salida_noche_share": 0.25,
    "llegada_noche_share": 0.25,
}

def consultar(url, payload=None):
    datos = json.dumps(payload).encode() if payload is not None else None
    req = Request(
        url,
        data=datos,
        headers={"Content-Type": "application/json"} if datos else {},
        method="POST" if datos else "GET",
    )
    with urlopen(req, timeout=8) as respuesta:
        return respuesta.status, respuesta.read().decode()

def esperar(nombre, url):
    for intento in range(90):
        try:
            estado, contenido = consultar(url)
            if estado == 200:
                print(f"{nombre}: HTTP 200")
                return contenido
        except OSError:
            pass
        if intento % 15 == 0:
            print(f"Esperando {nombre}...")
        time.sleep(2)
    raise RuntimeError(f"{nombre} no responde correctamente")

def main():
    health = json.loads(esperar("API", API + "/health"))
    esperar("MLflow", MLFLOW + "/health")

    assert health["status"] == "ok"
    assert health["model_name"] == "mibici-equipo32-p3-grupos"
    assert health["model_version"] == 2

    status, cuerpo = consultar(API + "/predict", entrada)
    pred = json.loads(cuerpo)

    assert status == 200
    assert pred["model_name"] == health["model_name"]
    assert pred["model_version"] == 2

    modelo_dir = ROOT / "serving/models/p3"
    metadata = json.loads(
        (modelo_dir / "deployment_metadata.json").read_text()
    )
    columnas = metadata["feature_names"]
    df = pd.DataFrame([entrada], columns=columnas).astype("float64")
    modelo = mlflow.pyfunc.load_model(str(modelo_dir))
    esperado = int(np.asarray(modelo.predict(df)).reshape(-1)[0])

    assert pred["cluster"] == esperado
    print(f"Equivalencia verificada: cluster {esperado}")

    evidencia = {
        "modelo": health["model_name"],
        "version": health["model_version"],
        "api_health_http": 200,
        "mlflow_health_http": 200,
        "predict_http": status,
        "respuesta": pred,
        "cluster_local": esperado,
        "equivalencia": True,
        "resultado": "APROBADO",
    }
    salida = ROOT / "docs/evidencias/operacion_finops/validacion_compose.json"
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(json.dumps(evidencia, indent=2) + "\n")
    print(json.dumps(evidencia, indent=2))

if __name__ == "__main__":
    main()
