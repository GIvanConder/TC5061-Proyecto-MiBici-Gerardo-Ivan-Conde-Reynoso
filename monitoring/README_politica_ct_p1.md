# Política operativa de Continuous Training P1

## Alcance

La política supervisa el modelo de regresión de demanda
`mibici-equipo32-p1-demanda`, versión 1, alias `champion`.

El procedimiento es un motor de decisión basado en evidencia.
No ejecuta entrenamiento ni promoción automática.

Los umbrales se establecieron el 27 de noviembre de 2026
después de analizar los resultados retrospectivos disponibles.
Son valores operativos iniciales, no umbrales prospectivamente
validados.

## Referencia

- Modelo: PoissonRegressor con preprocesamiento ajustado en TRAIN.
- Versión histórica de datos: `7aa5e436d887`.
- Referencia de desempeño: TEST noviembre-diciembre de 2025.
- MAE de referencia: 5.409785669092729.
- Umbral inicial de alerta: incremento relativo de 15%.
- Umbral absoluto aproximado: 6.221254 viajes por estación-día.

## Frecuencias y acciones

| Indicador | Umbral inicial | Frecuencia | Acción |
|---|---|---|---|
| Duplicados estación-fecha | Mayor que cero | Cada ejecución | Bloquear |
| Nulos o valores no finitos | Mayor que cero | Cada ejecución | Bloquear |
| Capacidad no positiva | Al menos una fila | Semanal | Revisar catálogo |
| Extrapolación de dias_activa | Al menos 10% | Semanal | Diagnosticar |
| Drift individual | Distancia de al menos 0.10 | Semanal propuesta | Investigar |
| Drift global | Al menos 50% de variables | Semanal propuesta | Revisión ampliada |
| MAE | Incremento de al menos 15% | Mensual | Alertar |
| Persistencia | Dos meses consecutivos | Mensual | Evaluar reentrenamiento |

El diagnóstico semanal implementado comprueba calidad,
cobertura y extrapolación. Las semanas parciales no se
utilizan para emitir alertas semanales de extrapolación.

El reporte Evidently disponible compara TRAIN 2025 con
enero-octubre de 2026. Su ejecución semanal corresponde
a la frecuencia operativa propuesta y no a una tarea
programada actualmente en producción.

## Procedimiento de actuación

1. Verificar la disponibilidad e integridad de los datos.
2. Generar características respetando el horizonte temporal.
3. Analizar calidad, extrapolación y drift estadístico.
4. Revisar mensualmente el desempeño con etiquetas disponibles.
5. Investigar las alertas y posibles cambios estacionales.
6. Ante deterioro persistente, solicitar la evaluación de un candidato.
7. Definir nuevas particiones temporales antes de entrenar.
8. Comparar las versiones mediante un protocolo pareado.
9. Exigir aprobación humana antes de cualquier promoción.
10. Conservar el modelo anterior y verificar el procedimiento de reversión.

## Regla de promoción

Se conserva íntegramente la regla de promoción preestablecida
en `monitoring/comparar_ab_p1.py`, documentada en las
[evidencias A/B históricas](../docs/evidencias/ab_p1/README.md).

La política de monitoreo no redefine esa regla, no ejecuta
una comparación adicional ni autoriza promociones.

Un candidato futuro deberá evaluarse en datos independientes
de los empleados para ajustarlo y seleccionar sus parámetros.

La detección de drift o el incumplimiento de un umbral
no autoriza por sí mismo la reasignación del alias champion.

## Reversión

Si una versión promovida presenta una regresión operacional,
el responsable deberá revisar las métricas y restaurar,
cuando proceda, la versión anterior identificada por su
número y Run ID. La reversión debe registrarse con su causa,
evidencia y fecha.

No se han ejecutado promociones ni reversiones como parte
de este diagnóstico retrospectivo.

## Limitaciones

- La referencia histórica no controla completamente la estacionalidad.
- El catálogo de capacidad corresponde a septiembre de 2026.
- Los días sin salidas se representan como cero.
- El crecimiento de la red modifica la población evaluada.
- El monitoreo retrospectivo no equivale a tráfico real en producción.
- La sensibilidad por recorte de dias_activa no constituye una
  corrección validada del modelo.
- Se requiere recalibración prospectiva de los umbrales.

## Reproducción

Desde la raíz del repositorio y después de generar
los datos de evaluación temporal:

```bash
./.venv/bin/python -m monitoring.politica_ct_p1
./.venv/bin/python -m pytest tests/test_politica_ct_p1.py -v
```

La configuración se encuentra en `monitoring/politica_p1.json`.
Las evidencias se guardan en
`docs/evidencias/monitoreo_final/`.
