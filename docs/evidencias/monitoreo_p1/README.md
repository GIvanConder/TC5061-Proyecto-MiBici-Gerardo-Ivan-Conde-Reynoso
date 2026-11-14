# Evidencia de monitoreo P1 con Evidently

- Versión de datos: `7aa5e436d887`
- Referencia: enero-septiembre 2025 (90,230 filas)
- Periodo actual: noviembre-diciembre 2025 (22,565 filas)
- Variables analizadas: 12
- Variables con drift: 1
- Pruebas automáticas fallidas: 11

## Resultados por variable

| Variable | Método | Estadístico | Umbral | Drift |
|---|---|---:|---:|---|
| `dias_activa` | Wasserstein distance (normed) | 2.5865 | 0.10 | Sí |
| `lag_1` | Wasserstein distance (normed) | 0.0608 | 0.10 | No |
| `lag_2` | Wasserstein distance (normed) | 0.0561 | 0.10 | No |
| `media_7d` | Wasserstein distance (normed) | 0.0510 | 0.10 | No |
| `media_28d` | Wasserstein distance (normed) | 0.0307 | 0.10 | No |
| `lag_7` | Wasserstein distance (normed) | 0.0291 | 0.10 | No |
| `media_mismo_dia_4sem` | Wasserstein distance (normed) | 0.0255 | 0.10 | No |
| `lag_14` | Wasserstein distance (normed) | 0.0223 | 0.10 | No |
| `dia_semana` | Jensen-Shannon distance | 0.0186 | 0.10 | No |
| `feriado` | Jensen-Shannon distance | 0.0177 | 0.10 | No |
| `fin_semana` | Jensen-Shannon distance | 0.0072 | 0.10 | No |
| `capacidad` | Wasserstein distance (normed) | 0.0013 | 0.10 | No |

## Calidad y comprobaciones

- TRAIN: 90,230 filas, 369 estaciones.
- TEST: 22,565 filas, 370 estaciones.
- Sin nulos en las 12 características analizadas.
- Las particiones se obtuvieron de la preparación original del modelo P1.

### Tipos de pruebas fallidas

- `MaxValue`: 4
- `MeanValue`: 1
- `QuantileValue`: 3
- `RowCount`: 1
- `StdValue`: 1
- `ValueDrift`: 1

## Interpretación

`dias_activa` es la única variable que supera el umbral de drift de Evidently. Esta característica cuenta los días transcurridos en el panel de cada estación, por lo que su cambio entre periodos es coherente con el avance temporal.

Las diferencias de tamaño entre conjuntos y de valores extremos no equivalen automáticamente a fallas de calidad ni a degradación predictiva.

El criterio global de Evidently usa una proporción de variables con drift de 0.5; debe distinguirse de las alertas individuales.

Estos resultados describen data drift, no prueban concept drift. La recomendación de reentrenamiento se determinará después de comparar el desempeño de las versiones sobre el mismo periodo de evaluación.

## Reproducción

Desde la raíz del repositorio, con datos procesados disponibles:

```bash
./.venv/bin/python -m monitoring.preparar_p1
$HOME/.venvs/mibici-evidently/bin/python -m monitoring.reporte_drift_p1
./.venv/bin/python -m monitoring.resumir_drift_p1
./.venv/bin/python -m monitoring.exportar_evidencias_p1
```

El entorno Evidently se prepara con las dependencias fijadas en `monitoring/requirements.txt`.
