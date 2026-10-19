"""Configuración central del proyecto MiBici — Equipo 32.

Todas las rutas, fechas, URLs y semillas viven aquí.
Regla del equipo: ningún otro archivo define estas constantes.
"""

from pathlib import Path

# ---------- Rutas ----------
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ---------- Reproducibilidad ----------
SEED = 42

# ---------- Datos: rango elegido por el equipo ----------
# Enero-diciembre 2025. Las URLs se copian del portal de datos
# abiertos de MiBici; el ID numérico de la ruta cambia por archivo.
VIAJES_URLS = {
    "2025_01": "https://www.mibici.net/site/assets/files/97653/datos_abiertos_2025_01.csv",
    "2025_02": "https://www.mibici.net/site/assets/files/99879/datos_abiertos_2025_02.csv",
    "2025_03": "https://www.mibici.net/site/assets/files/101986/datos_abiertos_2025_03.csv",
    "2025_04": "https://www.mibici.net/site/assets/files/103813/datos_abiertos_2025_04.csv",
    "2025_05": "https://www.mibici.net/site/assets/files/105513/datos_abiertos_2025_05.csv",
    "2025_06": "https://www.mibici.net/site/assets/files/107088/datos_abiertos_2025_06.csv",
    "2025_07": "https://www.mibici.net/site/assets/files/108898/datos_abiertos_2025_07.csv",
    "2025_08": "https://www.mibici.net/site/assets/files/111021/datos_abiertos_2025_08.csv",
    "2025_09": "https://www.mibici.net/site/assets/files/113216/datos_abiertos_2025_09.csv",
    "2025_10": "https://www.mibici.net/site/assets/files/115104/datos_abiertos_2025_10.csv",
    "2025_11": "https://www.mibici.net/site/assets/files/116773/datos_abiertos_2025_11.csv",
    "2025_12": "https://www.mibici.net/site/assets/files/118153/datos_abiertos_2025_12.csv"
}

NOMENCLATURA_URL = "https://www.mibici.net/site/assets/files/1118/nomenclatura_2026_09.csv"

# ---------- Archivos de salida ----------
VIAJES_PARQUET = PROCESSED_DATA_DIR / "viajes.parquet"
ESTACIONES_PARQUET = PROCESSED_DATA_DIR / "estaciones.parquet"
MANIFEST_PATH = DATA_DIR / "manifest.json"

# ---------- Partición temporal (sin fuga del futuro) ----------
# Entrenamiento: ene-oct 2025 · Prueba: nov-dic 2025
TRAIN_END = "2025-10-31"
TEST_START = "2025-11-01"
# ---------- P3: clustering de estaciones ----------
P3_TRAIN_START = "2025-01-01 00:00:00"
P3_TRAIN_END = "2025-10-19 23:59:59"
P3_TEST_START = "2025-10-20 00:00:00"
P3_TEST_END = "2026-01-01 00:00:00"
P3_REDUCED_START = "2025-04-01 00:00:00"

P3_N_CLUSTERS = 4
P3_N_INIT = 20
P3_MAX_INACTIVITY_DAYS = 60
P3_MIN_HISTORY_DAYS = 30
P3_MIN_DURATION_SECONDS = 60
P3_MAX_DURATION_SECONDS = 86400

P3_MLFLOW_EXPERIMENT = "mibici-equipo32" #usar el mismo nombre de experimiento en todas las preguntas
P3_MLFLOW_RUN_NAME = "p3_kmeans_final"
P3_QUESTION = "Agrupar estaciones MiBici según sus patrones de uso y evaluar su estabilidad temporal."

P3_OUTPUT_DIR = REPORTS_DIR / "p3"
P3_PCA_PATH = FIGURES_DIR / "pca_p3_mibici_final.png"
P3_MAP_PATH = P3_OUTPUT_DIR / "mapa_p3_mibici_final.html"
P3_GEO_PATH = P3_OUTPUT_DIR / "resultado_p3_geo.parquet"
P3_TABLE_PATH = P3_OUTPUT_DIR / "tabla_maestra_clusters.csv"
P3_MODEL_PATH = MODELS_DIR / "p3_model_bundle.joblib"

# ---------- P3: modelos de comparación ----------
P3_BASELINE_K = 2
P3_AGGLO_LINKAGE = "ward"

P3_MLFLOW_BASELINE_RUN_NAME = "p3_baseline_kmeans_k2"
P3_MLFLOW_CANDIDATE_RUN_NAME = "p3_candidato_agglomerative_k4"

# P2
P2_TRAIN_START = "2025-01-01"  # TRAIN ene-sep
P2_VAL_START = "2025-10-01"  # VAL oct
P2_TEST_START = TEST_START  # TEST nov-dic
P2_TEST_END = "2026-01-01"

P2_HORIZONTE_HORAS = 2  # H: horas hacia adelante
P2_FRACCION_CAPACIDAD = 0.4

P2_MIN_DURATION_SECONDS = 60
P2_MAX_DURATION_SECONDS = 86400

P2_MLFLOW_EXPERIMENT = "mibici-equipo32"
P2_MLFLOW_BASELINE_RUN_NAME = "p2_baseline_logreg"
P2_MLFLOW_CANDIDATE_RUN_NAME = "p2_candidato_hgb"
P2_QUESTION = (
    "¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes "
    "en las próximas horas?"
)

P2_OUTPUT_DIR = REPORTS_DIR / "p2"

# ---------- Equipo (etiqueta común en todos los runs de MLflow) ----------
TEAM_ID = "equipo32"

# ---------- P1: regresión de viajes diarios por estación (Johanna Camargo) ----------
# Granularidad: estación × día. Horizonte: el día siguiente (la predicción
# del día D se emite al cierre de D-1), por lo que todos los rezagos son >= 1 día.
P1_TRAIN_START = "2025-01-01"  # TRAIN ene-sep
P1_VAL_START = "2025-10-01"  # VAL oct (selección y comparación de modelos)
P1_TEST_START = TEST_START  # TEST nov-dic (evaluación final, sin tocar)
P1_TEST_END = "2026-01-01"

P1_HORIZONTE_DIAS = 1
P1_LAGS_DIAS = [1, 2, 7, 14]
P1_VENTANAS_MEDIA_DIAS = [7, 28]
P1_BASELINE_LAG_DIAS = 7  # ingenuo estacional: mismo día de la semana anterior

P1_MIN_DURATION_SECONDS = 60
P1_MAX_DURATION_SECONDS = 86400

# Días de descanso obligatorio (art. 74 LFT) y Jueves/Viernes Santo de 2025.
P1_FERIADOS = [
    "2025-01-01", "2025-02-03", "2025-03-17", "2025-04-17", "2025-04-18",
    "2025-05-01", "2025-09-16", "2025-11-17", "2025-12-25",
]  # fmt: skip

P1_LINEAL_ALPHA = 1e-3
P1_HGB_PARAMS = {
    "loss": "poisson",
    "learning_rate": 0.06,
    "max_iter": 400,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 50,
    "l2_regularization": 1.0,
    "early_stopping": False,
}

P1_MLFLOW_EXPERIMENT = "mibici-equipo32"
P1_MLFLOW_BASELINE_RUN_NAME = "p1_baseline_naive_semanal"
P1_MLFLOW_LINEAL_RUN_NAME = "p1_candidato_poisson_lineal"
P1_MLFLOW_CANDIDATE_RUN_NAME = "p1_candidato_hgb_poisson"
P1_QUESTION = "¿Cuántos viajes se iniciarán en cada estación el día siguiente?"

P1_OUTPUT_DIR = REPORTS_DIR / "p1"
