# Model Card — P1: Predicción de demanda

**Proyecto:** MiBici — Equipo 32

**Fecha de revisión:** 27 de noviembre de 2026

**Estado:** Champion registrado; monitoreo retrospectivo disponible.

## 1. Identificación y linaje

| Campo | Valor |
|---|---|
| Modelo registrado | `mibici-equipo32-p1-demanda` |
| Versión vigente | 1 |
| Alias | `champion` |
| Algoritmo | PoissonRegressor |
| Experimento | `mibici-equipo32` |
| Run ID | `371bf2379bfa4a74b8effbe02d2ee3ae` |
| Logged Model ID | `m-f1e2eda1d16a4663a7ec1f50d75859d9` |
| URI | `models:/mibici-equipo32-p1-demanda/1` |
| Versión de datos | `7aa5e436d887` |
| Commit de entrenamiento | `e02f9450d245390024f328467f0732b98e20025d` |

El modelo fue seleccionado durante la fase de experimentación
y posteriormente registrado en MLflow. No se encuentra
expuesto mediante la API P3 del proyecto.

## 2. Propósito

Estimar cuántos viajes se iniciarán en cada estación
de MiBici durante el día siguiente.

La unidad de predicción es estación-día y el horizonte
es de un día. El resultado representa un conteo esperado
no negativo, que puede expresarse como un valor continuo.

## 3. Usos previstos

- Apoyo a la planificación diaria de operaciones.
- Anticipación de demanda por estación.
- Priorización de estaciones para revisión operativa.
- Comparación de patrones de demanda.
- Apoyo a decisiones supervisadas de redistribución.

### Usos no recomendados

- Pronosticar trayectos o comportamientos individuales.
- Estimar directamente el inventario físico de bicicletas.
- Emitir pronósticos de largo plazo sin actualizar rezagos.
- Sustituir decisiones humanas de distribución de recursos.
- Aplicar el modelo a otras ciudades sin validación específica.

## 4. Datos de entrenamiento y evaluación

Fuente: [Datos abiertos de MiBici](https://mibici.net/es/datos-abiertos/).

Se utilizaron viajes de 2025 y el catálogo de estaciones
del proyecto. Se excluyeron atributos personales no necesarios
para la predicción y se conservaron viajes con duración
entre 60 y 86,400 segundos.

| Partición | Periodo | Observaciones evaluables |
|---|---|---:|
| TRAIN | Enero-septiembre 2025 | 90,230 |
| VALIDATION | Octubre 2025 | 11,439 |
| TEST | Noviembre-diciembre 2025 | 22,565 |
| Evaluación externa | Enero-octubre 2026 | 122,365 |

Las particiones son temporales. La selección del modelo
se realizó mediante VALIDATION. TEST se reservó para
la evaluación final.

La evaluación de 2026 no se utilizó para ajustar,
seleccionar ni reentrenar el modelo.

## 5. Características de entrada

| Grupo | Variables |
|---|---|
| Calendario | `dia_semana`, `fin_semana`, `feriado` |
| Rezagos | `lag_1`, `lag_2`, `lag_7`, `lag_14` |
| Medias móviles | `media_7d`, `media_28d` |
| Estacionalidad | `media_mismo_dia_4sem` |
| Historial observado | `dias_activa` |
| Infraestructura | `capacidad` |

Total: 12 características.

Los rezagos y medias utilizan observaciones anteriores
al día objetivo. El calendario del día objetivo es
conocido de antemano.

`dias_activa` mide los días acumulados desde el inicio
de la actividad observada de la estación en el panel,
no su antigüedad real desde la inauguración.

El preprocesamiento incluye codificación del día de la
semana, transformación log1p de características de conteo
y StandardScaler ajustado durante el entrenamiento.

## 6. Configuración del modelo

| Parámetro | Valor |
|---|---|
| Algoritmo | PoissonRegressor |
| alpha | 0.001 |
| max_iter | 1000 |
| Horizonte | 1 día |
| Rezagos | 1, 2, 7 y 14 días |
| Ventanas móviles | 7 y 28 días |

Se seleccionó una solución adecuada para un objetivo de
conteo, con menor complejidad y desempeño favorable en
VALIDATION respecto a los candidatos evaluados.

## 7. Evaluación predictiva

| Métrica | TRAIN | VALIDATION | TEST |
|---|---:|---:|---:|
| MAE | 5.195047 | 5.103751 | 5.409786 |
| RMSE | 8.497768 | 8.294217 | 9.139358 |
| WAPE | 15.7305% | 14.2595% | 17.5258% |
| R2 | 0.934822 | 0.949304 | 0.923406 |
| Sesgo | 0.000043 | -0.655719 | 1.357662 |

El sesgo se expresa como predicción menos valor observado.

### Comparación contra baseline semanal

El baseline utiliza los viajes registrados siete días antes
en la misma estación.

| Partición | MAE baseline | MAE champion |
|---|---:|---:|
| VALIDATION | 6.231576 | 5.103751 |
| TEST | 7.219455 | 5.409786 |

El modelo seleccionado reduce el error absoluto respecto
al baseline en ambos periodos.

## 8. Comparación histórica de versiones

| Campo | Champion | Challenger |
|---|---|---|
| Versión | 1 | 2 |
| Alias | `champion` | `challenger` |
| Run ID | `371bf2379bfa4a74b8effbe02d2ee3ae` | `31ce19ba37554c7399d20de74a0fb31a` |
| Periodo de ajuste | Enero-septiembre 2025 | Enero-octubre 2025 |
| Filas de ajuste | 90,230 | 101,669 |
| MAE TEST | 5.409786 | 5.475699 |
| WAPE TEST | 17.5258% | 17.7393% |

El challenger reutiliza la familia del modelo y sus
hiperparámetros, incorporando octubre de 2025 al ajuste.

La comparación retrospectiva utiliza 22,565 observaciones
comunes y 61 días de evaluación.

- Diferencia MAE champion menos challenger: -0.065913.
- IC95% aproximado: [-0.113995, -0.020573].
- Bootstrap pareado por bloques de siete días.
- 3,000 repeticiones y semilla 42.
- Challenger superior en MAE diario: 11 de 61 días.

El challenger no cumple la regla preestablecida de promoción.
Se conserva el champion v1.

Esta comparación no constituye un experimento A/B online
aleatorizado.

## 9. Monitoreo temporal externo

| Indicador | TEST 2025 | Enero-octubre 2026 |
|---|---:|---:|
| MAE | 5.409786 | 6.068723 |
| RMSE | 9.139358 | 11.396509 |
| WAPE | 17.5258% | 19.4508% |
| Sesgo | 1.357662 | -1.632986 |
| MAE baseline | 7.219455 | 7.758607 |

Versión del conjunto externo: `c915fc2638a4`.

La evaluación comprende 304 días, 3,840,981 viajes utilizables
y 472 estaciones evaluadas. Se retiraron 758 copias
operacionales redundantes sin alterar los archivos fuente.

El MAE aumenta 12.18% respecto al TEST histórico, pero el
champion reduce el error del baseline externo en 21.78%
y lo supera durante los diez meses evaluados.

### Data drift

Evidently detecta cuatro variables con drift:

| Variable | Distancia normalizada | Umbral |
|---|---:|---:|
| dias_activa | 4.670011 | 0.10 |
| media_28d | 0.110376 | 0.10 |
| media_mismo_dia_4sem | 0.106309 | 0.10 |
| lag_14 | 0.102734 | 0.10 |

Cuatro de doce variables presentan drift (33.33%), por debajo
del umbral global de 50%.

Las pruebas de calidad y drift de Evidently reportan
76 resultados satisfactorios y 33 alertas. No equivalen
a 33 variables con drift ni demuestran concept drift.

### Impacto predictivo de dias_activa

El rango de TRAIN es de 28 a 272 días. El 91.76% de las
observaciones externas supera este soporte.

El efecto marginal estimado de añadir 365 días, manteniendo
constantes las demás características, reduce aproximadamente
2.18% la predicción.

El análisis de sensibilidad por recorte no se adopta
como modificación del modelo vigente.

## 10. Limitaciones y riesgos

- El sistema representa con cero los días sin salidas,
  sin distinguir ausencia de demanda de suspensión operativa.
- El catálogo de capacidad es estático y no reconstruye
  las capacidades vigentes para cada fecha.
- Las estaciones nuevas necesitan suficiente historial
  para construir las características.
- Las estaciones sin catálogo presentan errores relativos
  elevados y requieren revisión antes del uso operacional.
- Los registros históricos pueden publicarse con retraso;
  la disponibilidad efectiva de los rezagos debe verificarse.
- No se incluyen explícitamente meteorología, eventos
  extraordinarios ni interrupciones del servicio.
- No se proporcionan intervalos predictivos calibrados.
- El cambio de distribución no prueba causalidad ni
  deterioro de la relación subyacente entre variables.
- No se realizó un análisis de equidad por características
  demográficas. No se pretende inferir atributos personales.

## 11. Gobernanza y mantenimiento

La política inicial considera:

- Alerta de desempeño ante un aumento de MAE de al menos 15%.
- Umbral absoluto inicial: 6.221254 viajes por estación-día.
- Dos meses consecutivos de deterioro justifican evaluar
  un posible reentrenamiento.
- Monitoreo de calidad y extrapolación.
- Evaluación de drift semanal como frecuencia propuesta.
- Revisión mensual de desempeño con etiquetas disponibles.
- Promoción y reversión bajo supervisión humana.

La aplicación retrospectiva de la política genera alertas
en junio, septiembre y octubre de 2026, con recomendación
de evaluar reentrenamiento en octubre.

Los umbrales fueron definidos después del análisis externo
y requieren validación prospectiva.

No se realizó un nuevo entrenamiento, registro ni promoción.
La evaluación de 2026 corresponde únicamente al champion;
no constituye una nueva comparación entre versiones.

## 12. Reproducibilidad y evidencias

- [Configuración](../../mibici/config.py).
- [Feature engineering](../../mibici/features.py).
- [Entrenamiento P1](../../mibici/modeling/p1_regresion.py).
- [MLflow Project](../../MLproject).
- [Monitoreo histórico](../evidencias/monitoreo_p1/README.md).
- [Comparación A/B](../evidencias/ab_p1/README.md).
- [Resultados A/B](../evidencias/ab_p1/comparacion_ab_p1.json).
- [Monitoreo externo](../evidencias/monitoreo_final/README.md).
- [Reporte Evidently 2026](../evidencias/monitoreo_final/evidently_p1_2026.html).
- [Política de Continuous Training](../../monitoring/README_politica_ct_p1.md).
