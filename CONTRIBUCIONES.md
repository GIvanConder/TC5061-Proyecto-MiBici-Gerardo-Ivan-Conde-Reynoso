# Evidencia de trabajo — Etapa 2

## Modalidad de entrega

Para la segunda etapa del Proyecto Integrador de TC5061, el personal docente autorizó que cada integrante desarrollara y entregara una versión independiente del proyecto mediante una rama o repositorio propio.

La presente versión corresponde a:

- **Gerardo Iván Conde Reynoso**
- Matrícula: **A01840602**
- GitHub: `GIvanConder`
- Repositorio: `TC5061-Proyecto-MiBici-Gerardo-Ivan-Conde-Reynoso`

El repositorio conserva como base parte del historial y de los materiales generados durante la primera etapa colaborativa. Sin embargo, los cambios correspondientes a la Etapa 2 fueron desarrollados e integrados en esta versión individual.

## Integración de la Etapa 2

| PR | Commit en `main` | Alcance |
|---|---|---|
| #1 | `251e57e` | Infraestructura reproducible con MLflow Projects |
| #2 | `9c136ee` | Experimentación avanzada y tuning para P1 |
| #3 | `17308c3` | Experimentación avanzada y tuning para P2 |
| #4 | `b7a7a24` | Experimentación reproducible y tuning para P3 |
| #5 | `b41cbde` | Artefactos visuales y tercer candidato completo de P3 |
| #6 | `85fe07c` | Registro reproducible de modelos en MLflow Model Registry |
| #7 | `a30792d` | Actualización del README para la segunda entrega |

## Alcance implementado

La Etapa 2 incluye:

- MLflow Tracking con backend SQLite.
- MLflow Project con entrypoints `ingest`, `train` y `register`.
- Experimentación reproducible para P1, P2 y P3.
- Baselines y candidatos adicionales para cada pregunta.
- Búsquedas de hiperparámetros mediante nested runs.
- Registro de métricas, parámetros, artefactos y trazabilidad.
- Model Registry para los tres modelos seleccionados.
- Esquema de ciclo de vida mediante alias `champion`.
- Documentación reproducible desde un clon limpio.

No se simulan revisiones cruzadas ni contribuciones de otros integrantes para esta segunda etapa. La trazabilidad de esta entrega se encuentra en el historial de Git, Pull Requests, MLflow y Model Registry.
