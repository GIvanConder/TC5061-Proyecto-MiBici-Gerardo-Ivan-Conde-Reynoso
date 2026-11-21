"""Carga concurrente sobre FastAPI P3 de MiBici."""

import json
import os
from itertools import cycle
from pathlib import Path

from locust import HttpUser, between, task, events

CONFIG = json.loads(
    Path(__file__).with_name("perfiles.json").read_text()
)

MODELO = CONFIG["modelo"]
VERSION = CONFIG["version"]


class UsuarioMiBici(HttpUser):
    wait_time = between(0.1, 0.3)

    def on_start(self):
        self.perfiles = cycle(CONFIG["perfiles"])

    @task
    def predecir_estacion(self):
        perfil = next(self.perfiles)

        with self.client.post(
            "/predict",
            json=perfil["entrada"],
            name="/predict",
            catch_response=True,
            timeout=10,
        ) as respuesta:

            if respuesta.status_code != 200:
                respuesta.failure(
                    f"HTTP {respuesta.status_code}"
                )
                return

            try:
                datos = respuesta.json()
            except ValueError:
                respuesta.failure("JSON invalido")
                return

            if datos.get("model_name") != MODELO:
                respuesta.failure("Modelo inesperado")
                return

            if datos.get("model_version") != VERSION:
                respuesta.failure("Version inesperada")
                return

            cluster = datos.get("cluster")

            if (
                type(cluster) is not int
                or cluster != perfil["cluster_esperado"]
            ):
                respuesta.failure("Cluster inesperado")
                return

            respuesta.success()


# Exportacion final de estadisticas
@events.test_stop.add_listener
def exportar_resultado_final(environment, **kwargs):
    destino = os.getenv("MIBICI_LOCUST_FINAL_JSON")

    if not destino:
        return

    estadistica = environment.stats.entries[
        ("/predict", "POST")
    ]

    datos = {
        "solicitudes": estadistica.num_requests,
        "fallos": estadistica.num_failures,
        "rps": estadistica.total_rps,
        "latencia_media_ms": estadistica.avg_response_time,
        "p50_ms": estadistica.get_response_time_percentile(0.50),
        "p95_ms": estadistica.get_response_time_percentile(0.95),
        "p99_ms": estadistica.get_response_time_percentile(0.99),
    }

    Path(destino).write_text(
        json.dumps(datos, indent=2) + "\n"
    )
