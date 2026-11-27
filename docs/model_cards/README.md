# Model Cards — Proyecto MiBici

Documentación de los modelos registrados y de sus usos,
resultados, restricciones y condiciones de mantenimiento.

**Fecha de revisión:** 27 de noviembre de 2026.

## Modelos del proyecto

| Problema | Modelo | Versión vigente | Ficha técnica |
|---|---|---|---|
| P1: Demanda | PoissonRegressor | Champion v1 | [Model Card P1](P1_demanda.md) |
| P2: Riesgo | HistGradientBoostingClassifier | Champion v1 | [Model Card P2](P2_riesgo.md) |
| P3: Segmentación | KMeans | Champion v1 / Serving v2 | [Model Card P3](P3_segmentacion.md) |

## Inventario del Model Registry

| Modelo registrado | Versión | Alias | Estado |
|---|---:|---|---|
| mibici-equipo32-p1-demanda | 1 | champion | READY |
| mibici-equipo32-p1-demanda | 2 | challenger | READY |
| mibici-equipo32-p2-riesgo | 1 | champion | READY |
| mibici-equipo32-p3-grupos | 1 | champion | READY |
| mibici-equipo32-p3-grupos | 2 | serving | READY |

El estado READY indica disponibilidad de la versión en el
Registry; no implica que todas las versiones estén desplegadas.

## Distribución de responsabilidades MLOps

| Componente | Alcance demostrado |
|---|---|
| Experimentación | P1, P2 y P3 |
| Model Registry | P1, P2 y P3 |
| Monitoreo Evidently | P1 |
| Comparación A/B retrospectiva | P1 |
| Política de Continuous Training | P1 |
| API REST y contenedores | P3 serving v2 |
| Microservicios y pruebas de carga | API P3 y arquitectura Compose |
| CI/CD, seguridad y reproducibilidad | Proyecto |

## Consideraciones comunes

Los modelos fueron desarrollados para apoyar decisiones
operacionales relacionadas con el sistema MiBici.

Los datos de viajes se procesan para obtener características
agregadas de estaciones. No se pretende inferir atributos
personales ni realizar perfiles individuales de usuarios.

Las métricas corresponden a los periodos y protocolos
descritos en cada ficha. No constituyen garantías de
desempeño futuro.

La evaluación externa de 2026 corresponde exclusivamente
al champion P1.

Las Model Cards deberán actualizarse cuando cambien
las versiones, los datos, los criterios de uso, las métricas
de desempeño o las condiciones operacionales.

La revisión actual no implica reentrenamiento, registro
ni promoción de nuevas versiones.

## Fuentes de trazabilidad

- [Documentación general](../../README.md).
- [MLflow Project](../../MLproject).
- [Model Registry](../../mibici/model_registry.py).
- [Monitoreo y gobernanza](../evidencias/monitoreo_final/README.md).
- [Evidencias operacionales](../evidencias/operacion_finops/README_locust_finops.md).
- [Metadatos del modelo desplegado](../../serving/models/p3/deployment_metadata.json).

Los identificadores, hiperparámetros y métricas registrados
se recuperaron de la base histórica de MLflow en modo
de solo lectura. Los resultados complementarios proceden
de las evidencias experimentales versionadas.
