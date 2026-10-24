import json

from loguru import logger
import typer

app = typer.Typer(add_completion=False)

VALID_QUESTIONS = {"p1", "p2", "p3"}


@app.command()
def main(
    question: str = typer.Option(..., help="Pregunta del proyecto: p1, p2 o p3."),
    model: str = typer.Option(..., help="Modelo o familia de modelo a ejecutar."),
    params_json: str = typer.Option(
        "{}",
        help="Hiperparámetros del modelo en formato JSON.",
    ),
):
    question = question.strip().lower()
    model = model.strip().lower()

    if question not in VALID_QUESTIONS:
        raise typer.BadParameter(
            f"Pregunta inválida: {question}. Usa p1, p2 o p3.",
            param_hint="question",
        )

    try:
        params = json.loads(params_json)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(
            f"params_json no es JSON válido: {exc.msg}",
            param_hint="params-json",
        ) from exc

    if not isinstance(params, dict):
        raise typer.BadParameter(
            "params_json debe contener un objeto JSON.",
            param_hint="params-json",
        )

    logger.info(f"Pregunta: {question}")
    logger.info(f"Modelo: {model}")
    logger.info(f"Hiperparámetros: {params}")

    if question == "p1":
        from mibici.modeling.p1_regresion import (
            ejecutar_busqueda_hgb_p1,
            ejecutar_modelo_p1,
        )

        if model == "hgb_search":
            if params:
                raise typer.BadParameter(
                    "hgb_search usa la rejilla definida en config.py "
                    "y no acepta params_json.",
                    param_hint="params-json",
                )

            resultados = ejecutar_busqueda_hgb_p1()
            typer.echo(
                json.dumps(
                    resultados,
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return

        try:
            metricas = ejecutar_modelo_p1(model, params)
        except ValueError as exc:
            raise typer.BadParameter(
                str(exc),
                param_hint="model",
            ) from exc

        typer.echo(json.dumps(metricas, ensure_ascii=False, indent=2))
        return

    raise typer.BadParameter(
        f"{question} todavía no está conectado al nuevo orquestador.",
        param_hint="question",
    )


if __name__ == "__main__":
    app()
