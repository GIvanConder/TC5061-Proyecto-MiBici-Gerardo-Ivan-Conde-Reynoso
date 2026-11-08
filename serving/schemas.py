"""Esquemas de entrada y salida para la API de MiBici P3."""

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


class P3Input(BaseModel):
    """Caracteristicas ya transformadas y estandarizadas."""

    model_config = ConfigDict(extra="forbid")

    salidas_dia: FiniteFloat
    llegadas_dia: FiniteFloat
    balance_relativo: FiniteFloat
    share_fin_semana: FiniteFloat
    salida_manana_share: FiniteFloat
    llegada_manana_share: FiniteFloat
    salida_mediodia_share: FiniteFloat
    llegada_mediodia_share: FiniteFloat
    salida_tarde_share: FiniteFloat
    llegada_tarde_share: FiniteFloat
    salida_noche_share: FiniteFloat
    llegada_noche_share: FiniteFloat


class P3Prediction(BaseModel):
    """Respuesta de la prediccion del modelo KMeans."""

    cluster: int = Field(ge=0, le=3)
    model_name: str
    model_version: int
