#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = tc5061-proyecto-equipo32
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Ingesta reproducible y versionado de datos (data/manifest.json)
.PHONY: data
data:
	$(PYTHON_INTERPRETER) -m mibici.dataset

## P1: regresión de viajes diarios por estación (MLflow)
.PHONY: p1
p1:
	$(PYTHON_INTERPRETER) -m mibici.modeling.p1_regresion

## P2: clasificación de riesgo de vaciado/llenado (MLflow)
.PHONY: p2
p2:
	$(PYTHON_INTERPRETER) -m mibici.modeling.p2_clasificacion

## P3: clustering de estaciones por patrón de uso (MLflow)
.PHONY: p3
p3:
	$(PYTHON_INTERPRETER) -m mibici.modeling.p3_clustering

## Experimento base completo: P1, P2 y P3 en mibici-equipo32
.PHONY: train
train: p1 p2 p3

## Interfaz de MLflow en http://localhost:5000
.PHONY: mlflow-ui
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z0-9_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
