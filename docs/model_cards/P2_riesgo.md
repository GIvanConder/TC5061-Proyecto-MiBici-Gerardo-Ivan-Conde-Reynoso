# Model Card — P2: Clasificación de riesgo

**Proyecto:** MiBici — Equipo 32

**Fecha de revisión:** 27 de noviembre de 2026

**Estado:** Champion registrado y evaluado históricamente.

## 1. Identificación y linaje

| Campo | Valor |
|---|---|
| Modelo registrado | `mibici-equipo32-p2-riesgo` |
| Versión vigente | 1 |
| Alias | `champion` |
| Algoritmo | HistGradientBoostingClassifier |
| Experimento | `mibici-equipo32` |
| Run ID | `9e6d7264d7a74a68a9f8a02f4be3b03f` |
| Logged Model ID | `m-8bddd02b531d4d61936d74a3f9cadc93` |
| URI | `models:/mibici-equipo32-p2-riesgo/1` |
| Versión de datos | `7aa5e436d887` |
| Commit de entrenamiento | `99cfd3e9473bf3614caefc9d91d2546f5ba20606` |

El modelo fue seleccionado mediante VALIDATION y registrado
en MLflow. No forma parte de la API P3 actualmente desplegada.

## 2. Propósito

Identificar estaciones con riesgo de desequilibrio
operacional durante las próximas dos horas.

La unidad de análisis es estación-hora. El problema se
formula como clasificación binaria utilizando una etiqueta
indirecta de riesgo basada en los flujos de viajes.

## 3. Definición del objetivo

Se calcula el flujo neto acumulado de llegadas menos salidas
durante las horas t+1 y t+2.

La etiqueta positiva se asigna cuando:

`abs(neto_futuro) >= 0.40 * capacidad`

El signo del flujo permite caracterizar el riesgo asociado:

- Flujo neto negativo: riesgo de vaciado.
- Flujo neto positivo: riesgo de llenado.

La clase predicha es binaria. La dirección del desequilibrio
se utiliza para analizar resultados, no como una segunda
salida independiente del clasificador.

Este indicador no equivale a observar directamente
el número de bicicletas o anclajes disponibles.

## 4. Usos previstos

- Identificación preliminar de estaciones que requieren revisión.
- Apoyo a la planificación de redistribución.
- Priorización de alertas operativas supervisadas.
- Análisis temporal de patrones de desequilibrio.

### Usos no recomendados

- Afirmar que una estación está físicamente vacía o llena.
- Garantizar disponibilidad de bicicletas a usuarios.
- Ordenar redistribuciones sin revisar inventario real.
- Interpretar la salida como probabilidad calibrada
  sin una validación específica.
- Utilizarlo para decisiones autónomas de seguridad.

## 5. Datos y particiones

Fuente: [Datos abiertos de MiBici](https://mibici.net/es/datos-abiertos/).

Se utilizaron viajes de 2025, el catálogo de capacidades
y el panel horario construido mediante feature engineering.

Se consideran viajes con duración entre 60 y 86,400 segundos.
Se excluyen observaciones sin ventana futura completa
o con capacidad no positiva.

| Partición | Periodo | Observaciones |
|---|---|---:|
| TRAIN | Enero-septiembre 2025 | 2,348,476 |
| VALIDATION | Octubre 2025 | 273,798 |
| TEST | Noviembre-diciembre 2025 | 539,478 |

Estaciones en TRAIN: 369.

La selección del candidato y el umbral de decisión se
realizaron mediante VALIDATION. TEST se reservó para
evaluación final.

## 6. Características de entrada

| Grupo | Variables |
|---|---|
| Calendario | `hora_dia`, `dia_semana`, `fin_semana` |
| Flujos actuales | `salidas_h0`, `llegadas_h0`, `neto_h0` |
| Flujos recientes | `neto_h1`, `neto_h2`, `neto_3h` |
| Capacidad | `capacidad`, `neto_3h_rel` |
| Comparación histórica | `neto_futuro_ayer`, `neto_futuro_semana` |
| Estadísticas históricas | `tasa_riesgo_hist`, `neto_futuro_medio_hist` |

Total: 15 características.

Las variables con nombre `neto_futuro_ayer` y
`neto_futuro_semana` representan ventanas históricas
equivalentes, no el flujo futuro del instante actual.

El diseño incorpora estadísticas históricas por estación,
hora del día y condición de fin de semana. Las tablas
de referencia aplicadas a VALIDATION y TEST se ajustan
con TRAIN.

Para un despliegue real debe verificarse también la
disponibilidad temporal efectiva de las etiquetas
históricas y de los flujos de la hora de decisión.

## 7. Configuración del modelo

| Parámetro | Valor |
|---|---|
| Algoritmo | HistGradientBoostingClassifier |
| learning_rate | 0.05 |
| max_iter | 350 |
| max_leaf_nodes | 63 |
| min_samples_leaf | 30 |
| l2_regularization | 2.0 |
| class_weight | balanced |
| loss | log_loss |
| early_stopping | False |
| random_state | 42 |
| Umbral de decisión | 0.877356 |

El tratamiento del desbalance incorpora ponderación
de clases y selección del umbral mediante VALIDATION.

El umbral de decisión no debe confundirse con una
probabilidad de riesgo físicamente calibrada.

## 8. Evaluación predictiva

| Métrica | TRAIN | VALIDATION | TEST |
|---|---:|---:|---:|
| F1 | 0.480446 | 0.469138 | 0.400710 |
| PR-AUC | 0.496913 | 0.486865 | 0.399357 |
| Precision | 0.439518 | 0.438200 | 0.341416 |
| Recall | 0.529779 | 0.504777 | 0.484927 |
| Accuracy | 0.955977 | 0.955018 | 0.958705 |
| Prevalencia positiva | 3.8421% | 3.9376% | 2.8470% |

### Comparación contra baseline

El baseline corresponde a Logistic Regression.

| Métrica | Baseline VAL | Champion VAL | Baseline TEST | Champion TEST |
|---|---:|---:|---:|---:|
| F1 | 0.415096 | 0.469138 | 0.364526 | 0.400710 |
| PR-AUC | 0.414629 | 0.486865 | 0.351402 | 0.399357 |
| Precision | 0.381259 | 0.438200 | 0.304629 | 0.341416 |
| Recall | 0.455524 | 0.504777 | 0.453740 | 0.484927 |

El modelo tuneado mejora el baseline en las métricas
relevantes para detectar la clase minoritaria.

La accuracy trivial de TEST, obtenida al predecir siempre
la clase mayoritaria, es 97.15%, frente a 95.87% del champion.

Por ello, accuracy no constituye el criterio principal
de selección en este problema desbalanceado.

### Evaluación por tipo de riesgo

| Tipo | Recall TEST |
|---|---:|
| Riesgo de vaciado | 0.456948 |
| Riesgo de llenado | 0.509547 |

El desempeño no es uniforme entre tipos de riesgo.

## 9. Limitaciones y riesgos

- La etiqueta representa un indicador indirecto de desequilibrio.
- No existe confirmación mediante inventario físico en tiempo real.
- La prevalencia de positivos cambia entre periodos.
- El recall de TEST implica que aproximadamente la mitad
  de los eventos positivos no son detectados.
- La precision de TEST indica que una proporción importante
  de alertas positivas corresponde a falsos positivos.
- El modelo depende de la disponibilidad y calidad del
  catálogo de capacidad.
- No se ha demostrado calibración de probabilidades.
- No se ha evaluado el desempeño con datos externos de 2026.
- No se ha realizado una validación operacional prospectiva.
- Deben considerarse costos asimétricos de falsos positivos
  y falsos negativos antes de automatizar decisiones.
- No se ha realizado una evaluación de equidad demográfica.

## 10. Gobernanza y mantenimiento

Se conserva la versión 1 bajo el alias `champion`.

Para una implementación futura se recomienda supervisar:

- Prevalencia de la clase positiva.
- F1, PR-AUC, precision y recall.
- Recall por tipo de riesgo.
- Calidad del catálogo de capacidad.
- Disponibilidad temporal de variables.
- Volumen de alertas y consecuencias operacionales.

Cualquier ajuste de umbral deberá realizarse sobre datos
de validación independientes, sin utilizar TEST para
la selección.

No existe un pipeline de monitoreo Evidently específico
para P2 dentro del alcance de la entrega final.

No se realizaron nuevos entrenamientos ni modificaciones
del modelo registrado.

## 11. Reproducibilidad y evidencias

- [Configuración](../../mibici/config.py).
- [Feature engineering](../../mibici/features.py).
- [Entrenamiento P2](../../mibici/modeling/p2_clasificacion.py).
- [MLflow Project](../../MLproject).
- [Model Registry](../../mibici/model_registry.py).
- [Documentación general](../../README.md).

Los parámetros y métricas anteriores se recuperaron
del Registry y Tracking históricos de MLflow en modo
de solo lectura.
