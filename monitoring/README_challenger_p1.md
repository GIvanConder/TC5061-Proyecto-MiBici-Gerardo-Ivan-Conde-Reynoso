# Entrenamiento reproducible del challenger P1

## Objetivo

Generar una versión alternativa del modelo de regresión de demanda
de MiBici, incorporando octubre de 2025 al entrenamiento original,
sin utilizar noviembre ni diciembre para ajustar el modelo.

## Modelo de referencia

- Modelo registrado: `mibici-equipo32-p1-demanda`.
- Versión original: 1, alias `champion`.
- Algoritmo: `PoissonRegressor`.
- Se conserva el Pipeline completo de preprocesamiento.
- Hiperparámetros: `alpha=0.001`, `max_iter=1000`.

## Particiones temporales

| Conjunto | Periodo | Observaciones | Uso |
|---|---|---:|---|
| TRAIN | Enero-septiembre 2025 | 90,230 | Entrenamiento |
| VALIDATION | Octubre 2025 | 11,439 | Se incorpora al reentrenamiento |
| TEST | Noviembre-diciembre 2025 | 22,565 | Evaluación posterior exclusivamente |

Los primeros registros utilizables de TRAIN corresponden al
29 de enero de 2025, después de generar los rezagos históricos.

El nuevo entrenamiento comprende 101,669 observaciones,
correspondientes a 12 características por estación-día.

Versión de datos: `7aa5e436d887`.

## Requisitos

- Repositorio MiBici y datos procesados disponibles.
- Entorno `.venv` con las dependencias del proyecto.
- MLflow Tracking Server accesible en `127.0.0.1:5000`.
- Registry y artefactos históricos disponibles.
- Rama `main` limpia para registrar el modelo definitivo.

## Iniciar MLflow

Desde la raíz del repositorio:

```bash
./.venv/bin/mlflow server \
  --backend-store-uri "sqlite:///$PWD/mlflow.db" \
  --artifacts-destination "$PWD/mlartifacts" \
  --serve-artifacts \
  --host 127.0.0.1 \
  --port 5000 \
  --workers 1
```

Si el servidor ya está funcionando con ese mismo almacenamiento,
no es necesario iniciar otro.

## Preparar datos y verificar

```bash
./.venv/bin/python -m monitoring.preparar_p1
./.venv/bin/python -m pytest tests -q
./.venv/bin/python -m monitoring.registrar_challenger_p1
```

El último comando ejecuta las comprobaciones previas sin
entrenar ni crear versiones nuevas.

## Registrar la alternativa

Una vez integrado el código y sincronizada la rama `main`:

```bash
./.venv/bin/python -m monitoring.registrar_challenger_p1 --register
```

El procedimiento:

1. Recupera el `champion` desde MLflow.
2. Clona su Pipeline, conservando las transformaciones.
3. Valida periodos, esquema, duplicados y valores no finitos.
4. Comprueba la versión de datos y calcula una huella SHA-256.
5. Reentrena usando exclusivamente TRAIN y octubre.
6. Registra parámetros, métricas de entrenamiento, firma,
   ejemplo de inferencia y metadatos de trazabilidad.
7. Recupera la nueva versión y verifica sus predicciones.
8. Conserva `champion` y asigna `challenger` a la nueva versión.

El script comprueba versiones equivalentes para evitar registros
duplicados bajo la misma huella y modelo de origen.

## Evaluación

La métrica de entrenamiento no determina la versión ganadora.

Ambos modelos deberán evaluarse sobre las mismas observaciones
de noviembre-diciembre de 2025, contrastando MAE, WAPE,
sesgo y errores pareados.

La asignación final del alias `champion` dependerá de los
resultados de esa comparación y de su interpretación.
