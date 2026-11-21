# CI/CD - Construye | Microservicios, CI/CD y FinOps

## Objetivo

Automatizar la validacion, construccion y despliegue temporal
de la arquitectura operacional de MiBici.

## Workflow

Ruta: `.github/workflows/ci-cd.yml`.

Eventos:

- push
- pull_request
- workflow_dispatch

## Entorno

- Runner hospedado Ubuntu.
- Python 3.14.
- Dependencias declaradas en requirements-ci.txt.
- Permisos de repositorio de solo lectura.

## Etapas

1. Recuperacion del repositorio.
2. Instalacion y validacion de dependencias.
3. Verificacion SHA-256 del artefacto P3.
4. Ejecucion de PyTest y reporte JUnit.
5. Validacion de Docker Compose.
6. Construccion de la imagen de inferencia.
7. Despliegue temporal de API y MLflow.
8. Pruebas HTTP y equivalencia predictiva.
9. Comprobacion de conectividad entre servicios.
10. Captura de logs y limpieza del entorno.
11. Publicacion de evidencias como artefactos de Actions.

## Evidencia de ejecucion

En GitHub, abrir Actions, seleccionar MiBici CI-CD y
consultar una ejecucion concreta.

El artefacto mibici-ci-evidence contiene los resultados
de pruebas, metadatos, logs e integracion.

## Alcance del despliegue

El workflow levanta el sistema completo dentro de un
runner temporal. No representa un despliegue permanente
en infraestructura de produccion.

No requiere acceso al Registry historico, datos originales
ni secretos de servicios externos.

No modifica los alias ni reentrena los modelos.

## Auditoria experimental automatizada

El workflow incorpora una comprobacion de reproducibilidad
de los resultados historicos de Locust y FinOps.

La auditoria regenera y compara las tablas numericas
utilizando los archivos versionados y publica los resultados
en los artefactos de la ejecucion.

No ejecuta nuevamente pruebas de carga.

Los artefactos de Actions tienen una retencion configurada
de 30 dias. Las evidencias experimentales principales
se conservan de forma independiente en Git.
