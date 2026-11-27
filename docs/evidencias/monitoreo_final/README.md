# Monitoreo temporal externo P1 — Proyecto final

## Propósito

Extender el monitoreo histórico del proyecto con datos reales
de enero a octubre de 2026, evaluar el efecto predictivo de
dias_activa y establecer una política inicial de Continuous
Training.

El champion P1 v1 permanece congelado. No se modificaron
los modelos, sus hiperparámetros ni los aliases de MLflow.

## Datos y trazabilidad

- Versión histórica: `7aa5e436d887`.
- Versión externa: `c915fc2638a4`.
- Diez meses de datos oficiales de MiBici.
- 304 días con cobertura.
- 3,997,597 registros originales.
- 758 copias operacionales redundantes en septiembre.
- 3,996,839 viajes únicos después de deduplicación.
- 3,840,981 viajes utilizables por P1.
- 122,365 observaciones estación-día evaluadas.
- 472 estaciones evaluadas.

Se conservó una copia por ID, inicio, fin, origen y destino.
No se identificaron IDs asociados a eventos contradictorios.
Los CSV originales permanecen intactos.

## Impacto predictivo del drift histórico

En TRAIN, dias_activa presenta un rango de 28 a 272 días.
El 99.48% del TEST histórico supera ese intervalo.

El coeficiente efectivo aprendido por el modelo es
aproximadamente -0.00006033 por día.

Un incremento de 365 días reduce la predicción aproximadamente
2.18%, manteniendo las demás características constantes.

La extrapolación es generalizada, pero su efecto marginal
directo es relativamente pequeño.

## Evaluación externa

| Métrica | TEST 2025 | Enero-octubre 2026 |
|---|---:|---:|
| MAE | 5.409786 | 6.068723 |
| RMSE | 9.139358 | 11.396509 |
| WAPE | 0.175258 | 0.194508 |
| Sesgo | 1.357662 | -1.632986 |
| MAE baseline semanal | 7.219455 | 7.758607 |

El MAE aumenta aproximadamente 12.18%, pero el champion
supera al baseline semanal en los diez meses evaluados.

En 2026, el 91.76% de las observaciones se encuentra fuera
del intervalo TRAIN de dias_activa.

El recorte experimental de la variable produce un MAE
de 6.005901 en 2026, frente a 6.068723 del modelo original.
No se adopta este cambio: el análisis es exploratorio y
requeriría validación independiente.

## Evidently

- Referencia: 90,230 observaciones TRAIN.
- Actual: 122,365 observaciones de 2026.
- Variables analizadas: 12.
- Variables con drift: 4.
- Proporción: 33.33%.
- Umbral global: 50%.
- Pruebas automáticas: 76 satisfactorias y 33 fallidas.

Las variables detectadas son dias_activa, lag_14, media_28d
y media_mismo_dia_4sem.

Las alertas automáticas no equivalen al número de variables
con drift y no demuestran concept drift.

## Gobernanza

La política inicial establece un incremento de MAE de 15%
como alerta de desempeño. Dos meses consecutivos sobre
el umbral solicitan la evaluación de un candidato.

El análisis retrospectivo genera alertas en junio,
septiembre y octubre. La persistencia de septiembre y
octubre justifica investigar un posible reentrenamiento.

No se ejecutó entrenamiento ni promoción de modelos.

Consultar `monitoring/README_politica_ct_p1.md`.

## Limitaciones

Se utiliza el catálogo histórico de septiembre de 2026.
No se reconstruyen capacidades vigentes para cada fecha.
El panel representa como cero los días sin salidas y no
distingue suspensión operativa de ausencia de demanda.

La evaluación usa rezagos observados hasta el día anterior.
No constituye un pronóstico emitido al inicio de 2026.

Las diferencias entre segmentos y periodos no permiten
atribuir causalmente el deterioro a una sola característica.

## Reproducción principal

Con los datos históricos, artefactos originales de MLflow
y los entornos del proyecto disponibles:

```bash
./.venv/bin/python -m monitoring.impacto_drift_p1
./.venv/bin/python -m monitoring.descargar_2026
./.venv/bin/python -m monitoring.diagnosticar_duplicados_2026
./.venv/bin/python -m monitoring.evaluar_temporal_2026
$HOME/.venvs/mibici-evidently/bin/python -m monitoring.reporte_drift_2026
./.venv/bin/python -m monitoring.politica_ct_p1
./.venv/bin/python -m monitoring.exportar_evidencias_final
./.venv/bin/python -m pytest -q
```

Los datos originales y los Parquet intermedios permanecen
en almacenamiento local y no forman parte de Git.

Los resultados agregados, reportes HTML, JSON y gráficas
permiten inspeccionar la evidencia sin repetir el entrenamiento.


## Diagnóstico por segmentos

| Segmento | Observaciones | MAE | WAPE | MAE baseline |
|---|---:|---:|---:|---:|
| Estaciones históricas | 112783 | 6.302195 | 0.192416 | 8.078017 |
| Estaciones nuevas | 9582 | 3.320694 | 0.256886 | 3.999061 |
| Estaciones sin catálogo | 233 | 2.558065 | 0.889596 | 2.506438 |

Las estaciones nuevas presentan menor error absoluto,
pero mayor error relativo. El segmento sin catálogo
presenta un WAPE elevado y no supera al baseline;
requiere revisión de calidad antes de cualquier uso
operacional que dependa de estas predicciones.

## Interpretación de filas repetidas en Evidently

Se detectaron 166 copias excedentes de vectores de
características, correspondientes a 274 filas,
108 grupos y 14 estaciones.

No existen claves estación-fecha duplicadas. Se conservaron
todas las observaciones, ya que la igualdad de características
no demuestra identidad de la unidad de análisis.

## Resultados de la política operativa

- 42 semanas completas y 2 parciales.
- 42 semanas completas con alerta de extrapolación.
- 17 semanas con alerta de catálogo, incluidas las parciales.
- Alertas de desempeño en junio, septiembre y octubre.
- Una recomendación retrospectiva de evaluar reentrenamiento
  en octubre de 2026.
- Ningún entrenamiento ni promoción ejecutados.

## Relación con el A/B testing histórico

La comparación pareada de P1 v1 y P1 v2 corresponde
al periodo histórico de evaluación de 2025.

Consultar [evidencias A/B](../ab_p1/README.md).

La evaluación de 2026 supervisa exclusivamente al champion
P1 v1. No constituye una nueva comparación entre versiones
ni demuestra superioridad del champion sobre el challenger
durante 2026.

La política de Continuous Training recomienda investigar
un posible reentrenamiento. No forma parte de su alcance
entrenar, registrar ni promover nuevas versiones.

## Reproducción de evidencias complementarias

Con las tablas agregadas ya disponibles, ejecutar:

```bash
./.venv/bin/python -m monitoring.exportar_evidencias_final
```

Este módulo regenera el resumen del tratamiento de duplicados
y las gráficas 04 y 05 sin utilizar datos crudos ni MLflow.

Los CSV de este directorio se versionan explícitamente
como evidencias agregadas. Los CSV originales de viajes,
los Parquet intermedios y la base histórica de MLflow
permanecen excluidos de Git.
