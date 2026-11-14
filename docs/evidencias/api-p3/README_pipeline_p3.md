# Corrección del preprocesamiento del modelo P3

## Problema identificado

La versión 1 del modelo P3 registrada en MLflow conserva el
K-Means ajustado, pero su artefacto de inferencia no incorpora
las transformaciones utilizadas durante el entrenamiento.

Por ello, el consumidor debía proporcionar 12 características
ya transformadas y estandarizadas.

## Solución implementada

Se reconstruye el preprocesamiento con la metodología
histórica del proyecto:

1. Recuperar las características originales de P3.
2. Aplicar log1p a salidas_dia y llegadas_dia.
3. Reconstruir StandardScaler exclusivamente con TRAIN.
4. Recuperar K-Means v1 desde MLflow Model Registry.
5. Combinar los componentes en un Pipeline completo.
6. Verificar transformaciones y predicciones históricas.
7. Comprobar serialización y recuperación con MLflow.

No se reajustan los centroides del K-Means.

## Procedencia

- Modelo: mibici-equipo32-p3-grupos.
- Versión histórica: 1.
- Run ID: c30b8d94d8f445e3a8abd1699676bfd9.
- Data version: 7aa5e436d887.
- Estaciones TRAIN: 367.
- Estaciones TEST: 367.

## Resultados del preflight

- Diferencia máxima de transformación TRAIN: 0.0.
- Diferencia máxima de transformación TEST: 0.0.
- Coincidencia de clusters TRAIN: completa.
- Coincidencia de clusters TEST: completa.
- Coincidencia con ejemplo histórico: verificada.
- Serialización y recuperación MLflow/skops: verificadas.

La comparación sobre TEST valida la equivalencia de
predicciones; TEST no interviene en el ajuste del scaler
ni en el entrenamiento de K-Means.

El diagnóstico estructurado se encuentra en
validacion_pipeline_p3.json.

## Reproducción

Desde la raíz del repositorio, con los datos originales,
el entorno .venv y MLflow Tracking Server disponibles:

```bash
./.venv/bin/python -m pytest tests -q
./.venv/bin/python -m serving.registrar_pipeline_p3
```

El segundo comando ejecuta el preflight sin registrar
versiones nuevas.

## Registro de la versión corregida

Una vez integrado el código en main:

```bash
./.venv/bin/python -m serving.registrar_pipeline_p3 --register
```

El registro se realiza desde un commit limpio y recupera
la versión generada para comprobar sus predicciones.

La nueva versión recibe el alias serving. El alias
champion permanece en la versión histórica 1.

## Limitaciones

El procedimiento requiere acceso a los datos procesados
originales y al Registry local. El repositorio público
no contiene mlflow.db ni todos los artefactos históricos.

La integración de esta nueva versión con FastAPI y
Podman se realizará en una corrección independiente.
