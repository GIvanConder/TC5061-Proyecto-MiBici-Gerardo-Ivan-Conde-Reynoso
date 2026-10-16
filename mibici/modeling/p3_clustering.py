"""Clustering de estaciones MiBici para la pregunta P3."""

from pathlib import Path

from loguru import logger
import typer

from mibici.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "p3_features.parquet",
):
    """Ejecuta el pipeline de clustering de P3."""
    logger.info("Iniciando pipeline P3...")
    logger.info(f"Features esperadas en: {features_path}")
    logger.success("Módulo P3 configurado correctamente.")


if __name__ == "__main__":
    app()
