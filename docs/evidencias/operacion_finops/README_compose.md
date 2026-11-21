# Construye | Microservicios, CI/CD y FinOps: orquestacion MiBici

## Servicios

- API: FastAPI con el Pipeline P3 v2.
- MLflow: Tracking Server con SQLite y artefactos persistentes.

La API usa el artefacto P3 local verificado y no consulta
el Registry en cada inferencia.

La instancia MLflow de Compose utiliza un volumen separado.
No reemplaza ni contiene automaticamente el historial del
Registry original del proyecto.

## Iniciar

Definir PODMAN_COMPOSE_PROVIDER con la ruta del proveedor
podman-compose instalado en el sistema.

Ejecutar desde la raiz del repositorio:

    podman compose -f docker-compose.yml up -d --build

## Verificar

    podman compose -f docker-compose.yml ps
    curl -f http://127.0.0.1:8000/health
    curl -f http://127.0.0.1:5000/health
    .venv/bin/python scripts/validar_compose.py

## Detener

    podman compose -f docker-compose.yml down

No ejecutar down -v si se desea preservar el volumen
de MLflow creado por Compose.

Los puertos se limitan a 127.0.0.1.
La imagen utiliza el usuario interno 10001.
Los modelos historicos no se reentrenan ni promocionan.
