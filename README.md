# TC5061 — Proyecto MiBici

Proyecto integrador de **Operaciones de Aprendizaje Automático (TC5061)** enfocado en la aplicación de prácticas de MLOps sobre los datos abiertos de **MiBici**, sistema de bicicleta pública del Área Metropolitana de Guadalajara.

La solución implementa un flujo reproducible mediante **MLflow Projects, MLflow Tracking y MLflow Model Registry**, manteniendo trazabilidad entre código, datos, experimentos y modelos seleccionados.

## Preguntas de negocio

| Pregunta | Objetivo | Tipo de tarea | Usuario |
|---|---|---|---|
| P1 | ¿Cuántos viajes se iniciarán en cada estación el día siguiente? | Regresión | Operador del sistema |
| P2 | ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes en las próximas 2 horas? | Clasificación | Equipo de rebalanceo |
| P3 | ¿Qué grupos de estaciones comparten patrones de uso y qué tan estables son en el tiempo? | Clustering | Planeación y operación |

Los datos utilizados corresponden a enero-diciembre de 2025 y contienen aproximadamente 4.5 millones de viajes.

El ML Canvas y otros materiales de referencia se encuentran en `references/`.

---

## Estructura principal del proyecto

```text
.
├── MLproject
├── README.md
├── requirements.txt
├── python_env.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── references/
├── reports/
├── tests/
└── mibici/
    ├── __init__.py
    ├── config.py
    ├── dataset.py
    ├── features.py
    ├── model_registry.py
    ├── plots.py
    └── modeling/
        ├── __init__.py
        ├── p1_regresion.py
        ├── p2_clasificacion.py
        ├── p3_clustering.py
        ├── predict.py
        └── train.py
```

La lógica reutilizable se mantiene dentro del paquete `mibici`. Los notebooks se utilizan únicamente para exploración y análisis preliminar.

---

# Reproducibilidad

## Requisitos

El proyecto fue desarrollado utilizando Python y dependencias fijadas en `requirements.txt`.

Se recomienda crear un entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

La consistencia del entorno puede comprobarse mediante:

```bash
python -m pip check
```

---

## Reproducción desde un clon limpio

Clonar el repositorio:

```bash
git clone https://github.com/GIvanConder/TC5061-Proyecto-MiBici-Gerardo-Ivan-Conde-Reynoso.git
cd TC5061-Proyecto-MiBici-Gerardo-Ivan-Conde-Reynoso
```

Crear y activar el entorno:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

En Windows PowerShell, la activación equivalente es:

```powershell
.venv\Scripts\Activate.ps1
```

---

# MLflow Tracking

El proyecto utiliza un servidor local de MLflow respaldado por una base de datos SQLite.

Desde la raíz del repositorio:

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --host 127.0.0.1 \
  --port 5000
```

En otra terminal:

```bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

La interfaz de MLflow queda disponible en:

```text
http://127.0.0.1:5000
```

Todas las corridas de esta entrega se almacenan en el experimento:

```text
mibici-equipo32
```

La base `mlflow.db`, los directorios internos de MLflow, los modelos serializados y los datos generados se encuentran excluidos del repositorio mediante `.gitignore`.

---

# MLflow Project

El archivo `MLproject` define tres puntos de entrada:

- `ingest`: ingesta y preparación de datos.
- `train`: entrenamiento y experimentación para P1, P2 y P3.
- `register`: registro de los modelos seleccionados en MLflow Model Registry.

## Ingesta

```bash
mlflow run . \
  -e ingest \
  --env-manager local
```

La ingesta descarga y procesa los datos definidos en `mibici/config.py` y genera:

```text
data/manifest.json
```

El manifiesto permite identificar la versión exacta de los datos utilizados.

Para los experimentos finales de esta entrega:

```text
data_version = 7aa5e436d887
```

---

# P1 — Predicción de demanda

## Objetivo

Predecir cuántos viajes se iniciarán en cada estación al día siguiente.

La granularidad es:

```text
estación × día
```

y el horizonte de predicción es de un día.

## Partición temporal

- TRAIN: enero-septiembre de 2025.
- VALIDATION: octubre de 2025.
- TEST: noviembre-diciembre de 2025.

Las variables incluyen calendario, capacidad de estación, rezagos y medias históricas construidas únicamente con información disponible antes del día objetivo.

## Experimentación

Se evaluaron:

- baseline ingenuo semanal;
- PoissonRegressor;
- HistGradientBoostingRegressor con pérdida Poisson;
- RandomForestRegressor;
- búsqueda de hiperparámetros para HGB mediante nested runs.

La búsqueda mantiene TEST fuera del proceso de selección.

## Modelo seleccionado

El modelo seleccionado es:

```text
PoissonRegressor
```

Resultados principales:

```text
VAL MAE  = 5.1038
TEST MAE = 5.4098
```

La selección considera el desempeño respecto al baseline, estabilidad y simplicidad del modelo para una variable objetivo de conteo.

## Reproducción del modelo seleccionado

```bash
mlflow run . \
  --experiment-name mibici-equipo32 \
  -e train \
  -P question=p1 \
  -P model=poisson \
  -P params_json='{}' \
  --env-manager local
```

---

# P2 — Riesgo de disponibilidad

## Objetivo

Clasificar si una estación estará en riesgo de quedarse sin bicicletas o sin anclajes durante las próximas dos horas.

La condición de riesgo utiliza el movimiento neto futuro relativo a la capacidad de la estación.

## Partición temporal

- TRAIN: enero-septiembre de 2025.
- VALIDATION: octubre de 2025.
- TEST: noviembre-diciembre de 2025.

El umbral de decisión se selecciona exclusivamente utilizando VALIDATION.

## Experimentación

Se evaluaron:

- LogisticRegression balanceada como baseline;
- HistGradientBoostingClassifier;
- RandomForestClassifier;
- HistGradientBoostingClassifier tuneado mediante nested runs.

La búsqueda de hiperparámetros utiliza `val_f1` como métrica principal y `val_pr_auc` como criterio secundario.

TEST permanece fuera del proceso de selección.

## Modelo seleccionado

El modelo seleccionado es un HistGradientBoostingClassifier tuneado.

Resultados principales:

```text
VAL F1         = 0.4691
VAL PR-AUC     = 0.4869

TEST F1        = 0.4007
TEST PR-AUC    = 0.3994
TEST Recall    = 0.4849
TEST Precision = 0.3414
```

## Reproducción del modelo seleccionado

```bash
mlflow run . \
  --experiment-name mibici-equipo32 \
  -e train \
  -P question=p2 \
  -P model=hgb \
  -P params_json='{"learning_rate":0.05,"max_iter":350,"max_leaf_nodes":63,"min_samples_leaf":30,"l2_regularization":2.0}' \
  --env-manager local
```

---

# P3 — Segmentación de estaciones

## Objetivo

Agrupar estaciones de MiBici de acuerdo con sus patrones de uso y evaluar la estabilidad de dichos grupos en el tiempo.

## Partición temporal

- TRAIN: 1 de enero al 19 de octubre de 2025.
- TEST: 20 de octubre al 31 de diciembre de 2025.
- Ventana histórica reducida: 1 de abril al 19 de octubre de 2025.

Se eliminan viajes con duración inferior a 60 segundos o superior a 24 horas, así como viajes que cruzan los límites temporales definidos.

## Características

Para cada estación se generan 12 variables relacionadas con:

- intensidad de salidas y llegadas;
- balance relativo;
- uso durante fines de semana;
- distribución temporal de salidas y llegadas durante mañana, mediodía, tarde y noche.

Las variables de intensidad utilizan transformación `log1p`.

Posteriormente se utiliza `StandardScaler`, ajustado exclusivamente con TRAIN.

## Experimentación

Se evaluaron:

- K-Means `k=2` como baseline;
- K-Means `k=3`;
- K-Means `k=4`;
- K-Means `k=5`;
- Agglomerative Clustering `k=4` como comparación adicional.

También se ejecutó una búsqueda mediante nested runs para:

```text
k = 3, 4, 5, 6
```

La búsqueda utiliza:

```text
train_silhouette
```

como criterio principal y:

```text
train_davies_bouldin
```

como criterio secundario.

TEST no participa en la selección.

## Modelo seleccionado

El modelo seleccionado es:

```text
K-Means
n_clusters = 4
n_init = 20
```

Resultados principales:

```text
TEST Silhouette         = 0.1666
TEST Davies-Bouldin     = 1.7448
ARI train-test          = 0.7927
NMI train-test          = 0.7361
ARI histórico reducido = 0.9415
NMI histórico reducido = 0.9167
```

Aunque el baseline `k=2` presenta mayor Silhouette, `k=4` ofrece mejor Davies-Bouldin, mayor estabilidad temporal e histórica y una segmentación más granular de las estaciones.

## Reproducción del modelo seleccionado

```bash
mlflow run . \
  --experiment-name mibici-equipo32 \
  -e train \
  -P question=p3 \
  -P model=kmeans_k4 \
  -P params_json='{}' \
  --env-manager local
```

---

# Nested runs

Las búsquedas de hiperparámetros se organizan mediante corridas anidadas en MLflow.

Cada búsqueda utiliza:

```text
parent run
└── child run por combinación
```

Esto permite comparar configuraciones sin perder el contexto de la búsqueda.

En P3, por ejemplo, la búsqueda queda organizada como:

```text
p3_busqueda_kmeans
├── p3_kmeans_trial_01  k=3
├── p3_kmeans_trial_02  k=4
├── p3_kmeans_trial_03  k=5
└── p3_kmeans_trial_04  k=6
```

La misma estrategia se utiliza para las búsquedas realizadas en P1 y P2.

---

# Artefactos y trazabilidad

Las corridas candidatas registran en MLflow los elementos necesarios para reproducir y analizar los resultados.

Entre los artefactos registrados se encuentran:

- modelo mediante `log_model`;
- firma del modelo;
- ejemplo de entrada;
- manifiesto de datos;
- configuración resuelta;
- gráficas de evaluación;
- resultados auxiliares de cada pregunta.

Los modelos registran además:

```text
git_commit
data_version
question
model_family
stage_of_work
team
```

## Artefactos de P1

Incluyen, según el modelo:

- métricas;
- predicciones;
- MAE por día;
- comparación de demanda total real contra predicha;
- importancia de variables.

## Artefactos de P2

Incluyen:

- matriz de confusión;
- curva precision-recall;
- reporte de clasificación;
- configuración;
- manifiesto de datos.

## Artefactos de P3

Las corridas finales de K-Means registran:

```text
clusters/
    *_clusters_train.csv
    *_perfiles_clusters.csv

plots/
    *_perfiles_clusters.png

config/
    p3_resolved_config.json

data/
    manifest.json
```

Los perfiles permiten interpretar las características medias que distinguen a cada grupo de estaciones.

---

# Model Registry

Los modelos seleccionados de P1, P2 y P3 se registran mediante el entrypoint:

```bash
mlflow run . \
  --experiment-name mibici-equipo32 \
  -e register \
  --env-manager local
```

El proyecto utiliza **aliases** para administrar el ciclo de vida de los modelos.

Se utiliza:

```text
@champion
```

para identificar la versión actualmente seleccionada para producción.

Los modelos registrados son:

| Pregunta | Modelo registrado | Versión | Alias |
|---|---|---:|---|
| P1 | `mibici-equipo32-p1-demanda` | 1 | `champion` |
| P2 | `mibici-equipo32-p2-riesgo` | 1 | `champion` |
| P3 | `mibici-equipo32-p3-grupos` | 1 | `champion` |

Los tres modelos conservan linaje hacia:

- corrida de origen;
- commit de Git;
- versión de datos;
- pregunta atendida.

## Carga de modelos en producción

Los modelos pueden cargarse directamente mediante nombre y alias.

P1:

```text
models:/mibici-equipo32-p1-demanda@champion
```

P2:

```text
models:/mibici-equipo32-p2-riesgo@champion
```

P3:

```text
models:/mibici-equipo32-p3-grupos@champion
```

La carga de los tres modelos fue verificada mediante MLflow como `PyFuncModel`.

---

# Privacidad

Los archivos originales de MiBici contienen algunos atributos que no son necesarios para las preguntas de negocio del proyecto.

Como medida de minimización de datos, dichos atributos se excluyen de los datasets utilizados para modelado.

Los datos se utilizan únicamente con fines académicos y no se redistribuyen mediante este repositorio.

---

# Archivos excluidos del repositorio

El repositorio no versiona:

- datos raw, interim o processed;
- `mlflow.db`;
- `mlruns/`;
- `mlartifacts/`;
- modelos serializados;
- resultados temporales o artefactos generados durante entrenamiento.

Estos elementos pueden reconstruirse mediante los comandos documentados anteriormente.

---

# Git y control de versiones

El desarrollo se organizó mediante ramas específicas y Pull Requests.

La secuencia principal de la segunda entrega incluye:

```text
#1  Infraestructura reproducible con MLflow Projects
#2  Experimentación avanzada P1
#3  Experimentación avanzada P2
#4  Experimentación reproducible P3
#5  Artefactos visuales y tercer candidato P3
#6  Model Registry reproducible
```

Cada cambio funcional se desarrolló fuera de `main` y posteriormente se integró mediante Pull Request.

Debido a una situación particular del equipo, el personal docente autorizó que cada integrante desarrollara y entregara una versión independiente de la actividad. La presente implementación corresponde a la versión individual de Gerardo Iván Conde Reynoso.

---

# Autor

**Gerardo Iván Conde Reynoso**

Maestría en Inteligencia Artificial Aplicada
Tecnológico de Monterrey
TC5061 — Operaciones de aprendizaje automático
