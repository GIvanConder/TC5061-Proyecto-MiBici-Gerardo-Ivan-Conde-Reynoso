# Model Card — P3: Segmentación de estaciones

**Proyecto:** MiBici — Equipo 32

**Fecha de revisión:** 27 de noviembre de 2026

**Estado:** Champion registrado y pipeline serving desplegable.

## 1. Identificación y linaje

| Campo | Champion | Serving |
|---|---|---|
| Modelo registrado | `mibici-equipo32-p3-grupos` | `mibici-equipo32-p3-grupos` |
| Versión | 1 | 2 |
| Alias | `champion` | `serving` |
| Algoritmo | KMeans | Pipeline con KMeans original |
| Run ID | `c30b8d94d8f445e3a8abd1699676bfd9` | `72a306c654714c35b9e1b73a0b4a784d` |
| Logged Model ID | `m-7918112016fa487f87ac4e87eb31d359` | `m-0a39cbba14e4464b95225e1e3d3392ff` |
| Commit | `ff1bdba3a500d296a0507451c1674d0ba16f4f8a` | `0b034b78e75dbd733f5f1c14a748286b0ee2ddcf` |

**Experimento:** `mibici-equipo32`.

**Versión de datos:** `7aa5e436d887`.

**URI de serving:** `models:/mibici-equipo32-p3-grupos/2`.

La versión 2 conserva el KMeans entrenado de la versión 1.
Añade el preprocesamiento ajustado durante el entrenamiento
para permitir inferencia a partir de características sin
transformar.

No se ejecutó un nuevo entrenamiento para crear la versión
serving. Las métricas de calidad predictiva y estabilidad
corresponden al modelo base v1.

## 2. Propósito

Agrupar estaciones de MiBici según sus patrones de uso
y evaluar la estabilidad de esa segmentación a través
de distintos periodos históricos.

La unidad de análisis es una estación caracterizada
mediante indicadores agregados de actividad.

El resultado es una etiqueta de clúster entre 0 y 3.
Las etiquetas son identificadores nominales y no representan
un orden de importancia, calidad o riesgo.

## 3. Usos previstos

- Segmentación de estaciones con patrones de uso similares.
- Apoyo a la planificación de recursos y redistribución.
- Identificación de perfiles de actividad.
- Comparaciones entre estaciones y periodos.
- Inferencia de perfiles mediante una API REST.

### Usos no recomendados

- Predecir directamente la demanda futura.
- Determinar la disponibilidad instantánea de bicicletas.
- Clasificar personas o inferir atributos demográficos.
- Utilizar el identificador del clúster como nivel de riesgo.
- Tomar decisiones irreversibles de infraestructura basadas
  exclusivamente en la segmentación.

## 4. Datos y criterios de elegibilidad

Fuente: [Datos abiertos de MiBici](https://mibici.net/es/datos-abiertos/).

Se utilizan viajes históricos de 2025, con duración entre
60 y 86,400 segundos.

| Partición | Periodo |
|---|---|
| TRAIN | 1 de enero al 19 de octubre de 2025 |
| TEST | 20 de octubre al 31 de diciembre de 2025 |
| Histórico reducido | 1 de abril al 19 de octubre de 2025 |

Se construyen perfiles de estaciones elegibles mediante
criterios dinámicos:

- Al menos 30 días de historial.
- Menos de 60 días sin actividad al cierre del periodo.
- Indicadores de salidas y llegadas normalizados por
  días de exposición.

No se utiliza una lista fija de estaciones para determinar
la elegibilidad.

## 5. Características de entrada

| Grupo | Variables |
|---|---|
| Intensidad | `salidas_dia`, `llegadas_dia` |
| Balance | `balance_relativo` |
| Fin de semana | `share_fin_semana` |
| Mañana | `salida_manana_share`, `llegada_manana_share` |
| Mediodía | `salida_mediodia_share`, `llegada_mediodia_share` |
| Tarde | `salida_tarde_share`, `llegada_tarde_share` |
| Noche | `salida_noche_share`, `llegada_noche_share` |

Total: 12 características.

Los bloques horarios se definen como mañana de 05 a 10 h,
mediodía de 11 a 15 h, tarde de 16 a 20 h y noche de
21 a 04 h.

El balance relativo se calcula como:

`(llegadas - salidas) / (llegadas + salidas)`

Las variables de intensidad utilizan transformación log1p.
Posteriormente se aplica StandardScaler ajustado en TRAIN.

La versión serving recibe las características agregadas
sin transformar. No recibe viajes crudos ni identificadores
de usuarios.

## 6. Configuración del modelo

| Parámetro | Valor |
|---|---|
| Algoritmo | KMeans |
| n_clusters | 4 |
| n_init | 20 |
| max_iter | 300 |
| init | k-means++ |
| algorithm | lloyd |
| tol | 0.0001 |
| random_state | 42 |

El baseline corresponde a KMeans con k=2.

La trazabilidad de MLflow indica selección mediante
métricas internas de TRAIN y que TEST no se utilizó
para seleccionar el modelo.

La elección de cuatro grupos busca un compromiso entre
calidad interna e interpretabilidad de la segmentación.
Las métricas posteriores caracterizan su estabilidad.

## 7. Calidad interna de los clústeres

| Métrica | TRAIN k=2 | TRAIN k=4 | TEST k=2 | TEST k=4 |
|---|---:|---:|---:|---:|
| Silhouette | 0.197666 | 0.186395 | 0.186141 | 0.166636 |
| Davies-Bouldin | 1.771526 | 1.584114 | 1.847495 | 1.744842 |
| Calinski-Harabasz | 100.827701 | 80.721639 | 92.242514 | 70.298461 |

Una menor puntuación Davies-Bouldin indica mejor
separación relativa según esa métrica.

El modelo seleccionado no supera al baseline en todas
las métricas internas. Silhouette indica una separación
modesta y posible solapamiento entre grupos.

Por tanto, no se afirma que cuatro clústeres constituyan
una partición universalmente óptima.

## 8. Estabilidad temporal e histórica

| Indicador | Baseline k=2 | Champion k=4 |
|---|---:|---:|
| ARI TRAIN-TEST | 0.725865 | 0.792685 |
| NMI TRAIN-TEST | 0.619183 | 0.736070 |
| Estabilidad exacta TEST | 0.926230 | 0.909836 |
| ARI histórico reducido | 0.883356 | 0.941493 |
| NMI histórico reducido | 0.808057 | 0.916712 |
| Estaciones comunes | 366 | 366 |

El modelo k=4 presenta mayor ARI y NMI tanto en la
comparación temporal como en la evaluación con histórico
reducido.

La estabilidad exacta es ligeramente inferior a k=2.
La selección representa un compromiso metodológico,
no una mejora simultánea en todas las métricas.

## 9. Pipeline de serving y despliegue

La versión serving v2 incorpora:

1. Recepción de las 12 características agregadas.
2. Transformación log1p de las intensidades.
3. Escalado mediante el StandardScaler original.
4. Inferencia con el KMeans seleccionado.

El Run ID base es:

`c30b8d94d8f445e3a8abd1699676bfd9`

La huella del preprocesamiento documentada es:

`e4cc9cb71612990a225a587227aa1538543122f2f3c11dd7dce5425545cc2086`

La API está implementada mediante FastAPI y empaquetada
en un contenedor compatible con Podman y Docker.

Se conservan los artefactos del modelo, la información
de linaje y las huellas SHA-256 de los ocho archivos
verificados.

### Interfaz de inferencia

- `GET /health`: estado e identificación del modelo.
- `POST /predict`: asignación de clúster a partir de
  las características agregadas.

La aplicación devuelve el clúster y la identificación
de la versión del modelo utilizado.

## 10. Operación y FinOps

La arquitectura Compose incluye un servicio de API
y un servicio MLflow con almacenamiento independiente.

La API realiza inferencia mediante el artefacto local
empaquetado; no necesita consultar el Registry en cada
petición.

El proyecto conserva evidencias de:

- Pruebas de integridad y reproducibilidad del pipeline.
- Pruebas de inferencia de la API.
- Construcción y ejecución del contenedor.
- Validación de los microservicios mediante Compose.
- Integración continua mediante GitHub Actions.
- Pruebas de carga con Locust.
- Comparación de configuraciones de workers.
- Estimaciones de costos de inferencia.

Los resultados de carga corresponden a experimentos
locales con perfiles sintéticos.

Las estimaciones FinOps son escenarios de referencia;
no constituyen facturación real de un despliegue en nube.

## 11. Limitaciones y riesgos

- Los grupos pueden solaparse, según las métricas internas.
- KMeans es sensible al escalado y a valores extremos.
- Las etiquetas no poseen significado ordinal.
- Los perfiles dependen del periodo de agregación.
- La elegibilidad excluye estaciones con historial insuficiente.
- La expansión del sistema puede modificar la segmentación.
- Una asignación de clúster no garantiza que el perfil
  pertenezca a la distribución de entrenamiento.
- No se realizó una evaluación específica de drift
  de P3 con los datos externos de 2026.
- Las pruebas de carga no representan tráfico real
  ni garantizan desempeño en otros entornos.
- No se realizó una evaluación de equidad demográfica.

## 12. Gobernanza y mantenimiento

Se mantienen dos roles diferenciados:

- `champion`: modelo seleccionado durante la experimentación.
- `serving`: pipeline completo autorizado para inferencia.

Se recomienda revisar periódicamente la distribución
de clústeres, estabilidad temporal, características de
entrada y elegibilidad de estaciones.

En operación también deben supervisarse latencia,
errores de inferencia, consumo de recursos y costos.

Cualquier versión futura debe mantener trazabilidad
de datos, entrenamiento, preprocesamiento y artefactos.

La promoción o reversión de una versión requiere
validaciones y aprobación supervisada.

No se ha implementado reentrenamiento automático.

## 13. Reproducibilidad y evidencias

- [Configuración](../../mibici/config.py).
- [Feature engineering](../../mibici/features.py).
- [Entrenamiento P3](../../mibici/modeling/p3_clustering.py).
- [MLflow Project](../../MLproject).
- [Pipeline completo](../evidencias/api-p3/README_pipeline_p3.md).
- [Validación del pipeline](../evidencias/api-p3/validacion_pipeline_p3.json).
- [Validación API y Registry](../evidencias/api-p3/validacion_api_registry_v2.json).
- [Metadatos de despliegue](../../serving/models/p3/deployment_metadata.json).
- [Microservicios Compose](../../docker-compose.yml).
- [Integración continua](../../.github/workflows/ci-cd.yml).
- [Documentación CI/CD](../evidencias/operacion_finops/README_ci_cd.md).
- [Locust y FinOps](../evidencias/operacion_finops/README_locust_finops.md).
- [Resumen FinOps](../evidencias/operacion_finops/finops/resumen_finops.json).
- [Auditoría de reproducibilidad](../evidencias/operacion_finops/auditoria_reproducibilidad.json).
