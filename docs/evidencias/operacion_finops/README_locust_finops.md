# Locust y FinOps - Entregable 7

## Objetivo

Evaluar el servicio P3 bajo carga concurrente y estimar
el costo por inferencia a partir de su rendimiento,
consumo de CPU y memoria.

## Diseno experimental

Se compararon uno y dos workers de Uvicorn, utilizando
la misma imagen, modelo P3 v2 y perfiles sinteticos.

Cada experimento comprende:

- Reposo: 30 segundos.
- Calentamiento: 2 usuarios durante 30 segundos.
- Carga baja: 5 usuarios durante 120 segundos.
- Carga media: 20 usuarios durante 180 segundos.
- Carga alta: 50 usuarios durante 180 segundos.

La carga es cerrada, con tiempos de espera de 0.1 a 0.3 s.
Cada respuesta se valida en su contenido, modelo,
version y cluster esperado.

Los experimentos se ejecutaron localmente en WSL2,
con Locust y los contenedores compartiendo hardware.

## Resultados

La comparacion se encuentra en:

    locust/comparacion_workers.csv
    locust/comparacion_workers.json

Cada ejecucion conserva sus CSV, logs, reportes HTML,
mediciones de Podman y graficas.

Los datos del CSV periodico de Locust pueden diferir
ligeramente de las estadisticas al finalizar la prueba.

Para la primera ejecucion se consolido el resumen final
de consola. Para la segunda se exportaron las estadisticas
mediante el evento test_stop, verificando su concordancia
con los conteos y percentiles finales de consola.

Se preservaron los archivos originales.

## Hallazgos

Con 50 usuarios, dos workers aumentaron el rendimiento
aproximadamente 70.68% y redujeron la latencia p95 60.47%,
a cambio de incrementar el consumo de memoria de FastAPI.

Se registraron 90,239 inferencias exitosas y 0 fallos en los escenarios principales.
Consultar resumen_finops.json para el total calculado.

MLflow mantuvo un consumo cercano a 1.96 GB de memoria,
aunque no interviene directamente en la inferencia.

## Reproducir

Instalar Locust en un entorno independiente del proyecto:

    python -m venv "$HOME/.local/share/mibici-tools/locust"

    "$HOME/.local/share/mibici-tools/locust/bin/python" -m pip install -r requirements-load.txt

    export LOCUST_BIN="$HOME/.local/share/mibici-tools/locust/bin/locust"

Configurar el proveedor de Podman Compose y levantar
la arquitectura.

Para una nueva medicion con un worker:

    MIBICI_API_WORKERS=1 podman compose up -d --build

    MIBICI_EXPERIMENT_LABEL=1_worker .venv/bin/python scripts/ejecutar_carga.py

    .venv/bin/python scripts/analizar_carga.py

Para repetir con dos workers:

    MIBICI_API_WORKERS=2 podman compose up -d --force-recreate api

    MIBICI_EXPERIMENT_LABEL=2_workers .venv/bin/python scripts/ejecutar_carga.py

    .venv/bin/python scripts/analizar_carga.py

Verificar el estado de salud y permitir la estabilizacion
de los procesos antes de iniciar cada nueva medicion.

Los archivos ejecucion_original.txt y ejecucion_optimizada.txt
identifican los experimentos historicos utilizados en FinOps.
No deben sobrescribirse inadvertidamente.

Para regenerar la estimacion economica historica:

    .venv/bin/python scripts/analizar_finops.py

## FinOps

Se emplearon tarifas publicas de Amazon Lightsail
Compute Optimized como referencia de infraestructura.

La utilizacion efectiva representa la fraccion equivalente
del rendimiento medido que se aprovecha a lo largo del mes.
No corresponde directamente a la utilizacion de CPU.

El costo se estima dividiendo el precio mensual entre las
inferencias exitosas proyectadas para 30 dias.

Tambien se incluye un calculo a demanda mensual fija,
para evitar atribuir ahorros a capacidad no utilizada.

Se considera un SLO exploratorio de p95 <= 200 ms y
tasa de errores <= 0.1%.

El SLO fue propuesto despues de observar los resultados.
No constituye un requisito validado de negocio.

## Decisiones operativas

- Mantener un worker como configuracion predeterminada.
- Considerar dos workers para cargas concurrentes elevadas.
- Dimensionar CPU y memoria conjuntamente.
- Evaluar FastAPI y MLflow como componentes independientes.
- Evitar infraestructura sobredimensionada.
- Realizar nuevas mediciones antes de contratar recursos.

El benchmark no garantiza el mismo rendimiento en AWS.
No se desplego ni contrato infraestructura comercial.

## Fuentes

- https://docs.locust.io/en/stable/
- https://docs.podman.io/en/latest/markdown/podman-stats.1.html
- https://www.finops.org/wg/finops-for-ai-overview/
- https://aws.amazon.com/lightsail/pricing/

## Auditoria automatizada de resultados historicos

Desde la raiz del repositorio:

    .venv/bin/python scripts/auditar_reproducibilidad.py

La auditoria regenera ocho tablas y dieciseis graficas
en una copia temporal extraida de Git.

Se comparan los valores de las tablas con las evidencias
versionadas. La validacion de graficas comprueba su
generacion y existencia, no equivalencia visual pixel a pixel.

No se necesitan contenedores activos ni repetir Locust
para esta auditoria historica.

Para repetir las mediciones de carga se utiliza
Podman Compose, el puerto predeterminado 8000 y
los nombres de contenedor del proyecto mibici-finops.

Los CSV y logs experimentales nuevos pueden estar
excluidos por las reglas generales de .gitignore.
Incorporarlos selectivamente cuando deban conservarse
como evidencias, sin versionar datasets originales.

## Limitaciones experimentales

Los escenarios representan una ejecucion por configuracion.
No se calcularon intervalos de confianza entre repeticiones.

El rendimiento incluye la comunicacion HTTP local y
el procesamiento de caracteristicas ya calculadas.

No se midio la generacion de caracteristicas desde viajes
originales ni se evaluo el rendimiento en infraestructura AWS.

Para clasificacion periodica de estaciones, el procesamiento
batch puede ser mas eficiente que las solicitudes individuales.
La API resulta apropiada para consultas bajo demanda.
