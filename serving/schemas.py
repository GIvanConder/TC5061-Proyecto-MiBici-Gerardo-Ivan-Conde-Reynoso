"""Esquemas de entrada y salida para la API de MiBici P3."""

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat


class P3Input(BaseModel):
    """Caracteristicas originales, previas al escalado de P3."""

    model_config = ConfigDict(extra="forbid")

    salidas_dia: FiniteFloat = Field(ge=0)
    llegadas_dia: FiniteFloat = Field(ge=0)
    balance_relativo: FiniteFloat = Field(ge=-1, le=1)
    share_fin_semana: FiniteFloat = Field(ge=0, le=1)
    salida_manana_share: FiniteFloat = Field(ge=0, le=1)
    llegada_manana_share: FiniteFloat = Field(ge=0, le=1)
    salida_mediodia_share: FiniteFloat = Field(ge=0, le=1)
    llegada_mediodia_share: FiniteFloat = Field(ge=0, le=1)
    salida_tarde_share: FiniteFloat = Field(ge=0, le=1)
    llegada_tarde_share: FiniteFloat = Field(ge=0, le=1)
    salida_noche_share: FiniteFloat = Field(ge=0, le=1)
    llegada_noche_share: FiniteFloat = Field(ge=0, le=1)


class P3Prediction(BaseModel):
    """Respuesta de la prediccion del modelo KMeans."""

    cluster: int = Field(ge=0, le=3)
    model_name: str
    model_version: int
