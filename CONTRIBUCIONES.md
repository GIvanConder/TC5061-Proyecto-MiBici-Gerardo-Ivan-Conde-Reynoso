# Contribuciones del Equipo 32

Registro de qué hizo cada integrante, construido a partir del historial de Git y de los Pull Requests
del repositorio (`git log --all` y pestaña *Pull requests* de GitHub). Sirve como evidencia de la
evaluación individual: commits propios, PR abiertos y revisiones hechas a otros integrantes.

| Integrante | Matrícula | GitHub | Responsabilidad |
|---|---|---|---|
| Rolando Calderón Carballo | A01840537 | `a01840537` (dueño del repo) | Infraestructura: estructura CCDS, ingesta, versionado de datos, configuración, README, ML Canvas y arquitectura |
| Gerardo Iván Conde Reynoso | A01840602 | `GIvanConder` | P3: clustering de estaciones por patrón de uso |
| Manuel Alejandro Cardona Esquivel | A01795954 | `alexcard33` | P2: clasificación de riesgo de vaciado/llenado |
| Leydy Johanna Camargo Medina | A01840540 | `JohannaCamargo2805` | P1: regresión de viajes diarios por estación; cierre de la entrega |

## Rolando Calderón Carballo (`a01840537`)

| Commit | Rama | Qué hizo |
|---|---|---|
| `75e3826` | `main` | `chore`: estructura base generada con cookiecutter-data-science |
| `4e4f6cb` | `feat/ingesta` | `feat`: ingesta reproducible de viajes y estaciones con versionado por manifiesto (`dataset.py`, `config.py`) |
| `ba4bcef` | `fix/experimento-unico-y-manifiesto` | `fix`: experimento único `mibici-equipo32` y `data/manifest.json` versionado en Git |
| `da671ef` | `docs/canvas-y-arquitectura` | `docs`: README completo, ML Canvas y diagrama de arquitectura en `references/` |

- **PR abiertos:** #1 `feat: ingesta reproducible y configuracion central`, #3 `fix: experimento unico del equipo y manifiesto versionado`, #4 `docs: ML Canvas y diagrama de arquitectura`.
- **Revisiones:** aprobó el PR #2 (P3, Gerardo).
- **Integró:** PR #1.

## Gerardo Iván Conde Reynoso (`GIvanConder`)

| Commit | Rama | Qué hizo |
|---|---|---|
| `bcd2039` | `feature/p3-clustering` | `feat(p3)`: inicializa el módulo de clustering |
| `42bc1e0` | `feature/p3-clustering` | `feat(p3)`: pipeline reproducible de clustering (`p3_clustering.py`, funciones `p3_*` de `features.py`, configuración P3, sección P3 del README) |

- **PR abiertos:** #2 `feat(p3): implementa pipeline reproducible de clustering`.
- **Revisiones:** aprobó el PR #1 (ingesta, Rolando) y el PR #5 (P2, Manuel).
- **Integró:** PR #2.

## Manuel Alejandro Cardona Esquivel (`alexcard33`)

| Commit | Rama | Qué hizo |
|---|---|---|
| `9289545` | `feature/acardona/p2-clasificacion` | P2: `p2_clasificacion.py`, funciones `p2_*` de `features.py`, configuración P2, objetivo `p2` del Makefile y `.gitignore` |

- **PR abiertos:** #5 `Agregado pregunta 2`.
- **Integró:** PR #3 y PR #5.
- **Nota:** el commit `9289545` se hizo con el correo `mcardona@icwgroup.com`, que no está asociado a la
  cuenta `alexcard33`; GitHub lo muestra sin usuario. Al agregar ese correo en *Settings → Emails* de
  GitHub, el commit queda atribuido a su cuenta.

## Leydy Johanna Camargo Medina (`JohannaCamargo2805`)

Rama `feat/p1-regresion-y-cierre` (PR de cierre de la entrega):

| Commit | Qué hizo |
|---|---|
| `d32c7e1` | `fix(build)`: corrige el Makefile (el objetivo `p2` tenía un error de separador) y agrega `data`, `p1`, `p2`, `p3`, `train` y `mlflow-ui` |
| `3325b7d` | `fix(dataset)`: elimina pseudónimo de usuario, género y año de nacimiento antes de guardar los Parquet (minimización de datos) |
| `8b1f983` | `docs(features)`: documenta todas las funciones de P2 y P3 y la regla común de no fuga |
| `4a592fa` | `feat(p1)`: P1 completa: configuración, funciones `p1_*` en `features.py`, `modeling/p1_regresion.py` (baseline ingenuo semanal, Poisson lineal y HGB Poisson), pruebas `tests/test_features_p1.py` |
| `b9f1a31` | `feat(mlflow)`: agrega la etiqueta `equipo = equipo32` a los runs de P2 y P3 |
| `b089c6c` | `docs(notebooks)`: notebook de exploración `1.0-ljcm-exploracion-demanda-p1.ipynb` |
| `6754760` | `docs(readme)`: reproducción desde un clon limpio, secciones P1 y P2, privacidad y tabla del equipo con matrícula y usuario |

- Ejecutó los runs oficiales de P1, P2 y P3 desde el commit `6754760` (árbol limpio) en el experimento
  `mibici-equipo32` con `data_version = 7aa5e436d887`, y tomó las capturas de MLflow del reporte.
- **Integró:** PR #4.

## Revisiones cruzadas

| PR | Autor | Revisó / aprobó | Integró |
|---|---|---|---|
| #1 | Rolando | Gerardo (aprobado) | Rolando |
| #2 | Gerardo | Rolando (aprobado) | Gerardo |
| #3 | Rolando | — | Manuel |
| #4 | Rolando | — | Johanna |
| #5 | Manuel | Gerardo (aprobado) | Manuel |
| Cierre (P1) | Johanna | pendiente: Manuel | — |
