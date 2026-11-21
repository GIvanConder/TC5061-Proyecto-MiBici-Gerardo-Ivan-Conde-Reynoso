"""Verifica la integridad del artefacto P3 para construir la imagen."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELO = ROOT / "serving" / "models" / "p3"


def main():
    metadata = json.loads(
        (MODELO / "deployment_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    assert metadata["model_name"] == "mibici-equipo32-p3-grupos"
    assert metadata["model_version"] == 2
    assert metadata["model_alias"] == "serving"
    assert metadata["input_semantics"] == "raw_p3_features"
    assert len(metadata["feature_names"]) == 12

    hashes = metadata["file_sha256"]
    assert len(hashes) == 8
    assert "model.skops" in hashes
    assert "MLmodel" in hashes

    for nombre, esperado in hashes.items():
        archivo = (MODELO / nombre).resolve()

        assert archivo.is_relative_to(MODELO.resolve())
        assert archivo.is_file(), f"Falta: {nombre}"

        actual = hashlib.sha256(
            archivo.read_bytes()
        ).hexdigest()

        assert actual == esperado, (
            f"SHA-256 incorrecto: {nombre}"
        )

        print(f"SHA-256 OK: {nombre}")

    print(
        json.dumps(
            {
                "model_name": metadata["model_name"],
                "model_version": metadata["model_version"],
                "run_id": metadata["run_id"],
                "archivos_verificados": len(hashes),
                "resultado": "APROBADO",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
