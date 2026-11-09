# Hogar, colegio y puntaje en el Saber 11

Proyecto integrador de Ciencia de Datos de la Maestría en Inteligencia Artificial de la Universidad de La Salle (Bogotá). Estudia en qué medida las condiciones socioeconómicas del hogar (estrato, educación de los padres, internet y computador) y las características del colegio (naturaleza, zona y jornada) explican y permiten anticipar el puntaje global del examen Saber 11 de 2022-2, con una serie de 2014-2 a 2022-2 como contexto.

El trabajo avanza en cuatro fases. La fase 1, que este repositorio fija en la etiqueta `entregable-1`, contiene el planteamiento, la ficha del conjunto de datos y el análisis exploratorio, con su réplica en R.

## Estructura

| Carpeta o archivo | Contenido |
| --- | --- |
| `notebooks/fase-1-eda.ipynb` | Cuaderno ejecutado de la fase 1, con todas sus salidas. Su fuente es `notebooks/fase-1-eda.py` (formato jupytext). |
| `src/saber11/` | Paquete del proyecto: carga reproducible de los datos (`datos.py`), diccionario de las 51 columnas (`diccionario.py`) y registro de las cifras que citan los informes (`cifras.py`). |
| `src/r/` | `instalar.R` instala los paquetes de R y registra sus versiones; `fase-1.R` replica en R los descriptivos, las frecuencias, las correlaciones y las pruebas χ² del cuaderno y dibuja sus gráficos en ggplot2. |
| `tests/` | Pruebas automáticas con pytest. |
| `data/agregados/` | Datos agregados que el tablero leerá desde el repositorio. Ver su propio README. |
| `data/` | Los recortes con filas individuales se generan aquí y no se versionan. |
| `figuras/fase-1/` | Figuras de la fase 1 (las de R en `r/`). |
| `resultados/` | Cifras de la fase 1 (`cifras-fase-1.json` y `.tex`) y salida del script de R (`r/fase-1.json`). |
| `dashboard/` | Tablero de las fases siguientes. |
| `pyproject.toml`, `uv.lock`, `requirements.txt` | Entorno de Python con versiones exactas. `r-versiones.txt` tiene las de R. |

## Reproducir la fase 1

Los pasos van en este orden, porque cada uno usa la salida del anterior: datos, R y cuaderno. Todos se corren desde la raíz del repositorio.

1. **Entorno de Python** (3.13). Con [uv](https://docs.astral.sh/uv/):

   ```bash
   uv sync
   ```

   O con pip, en un entorno virtual nuevo:

   ```bash
   pip install -r requirements.txt
   pip install -e . --no-deps
   ```

   La segunda línea instala el paquete `saber11` del proyecto, que `requirements.txt` no incluye. El entorno trae dependencias de las fases siguientes, entre ellas PyTorch con CUDA; la fase 1 no las usa.

2. **Datos.** Se descargan del portal de datos abiertos y se verifican contra los conteos esperados:

   ```bash
   uv run python -m saber11.datos --fuente api --verificar --materializar
   ```

   El comando baja por la API los seis periodos del proyecto (unos 4,3 millones de filas, porque el portal publica filas duplicadas que el paquete quita), deja solo los resultados en estado `PUBLICAR` y escribe `data/saber11-20224.parquet` y `data/saber11-serie.parquet`. Si ya se tiene la exportación completa del portal convertida a Parquet, `--fuente local --parquet <ruta>` hace lo mismo sin descargar.

3. **R** (4.6). Una sola vez, los paquetes; después, el script de la fase:

   ```bash
   Rscript src/r/instalar.R
   Rscript src/r/fase-1.R
   ```

   El script escribe `resultados/r/fase-1.json`, que el cuaderno compara con sus propios valores.

4. **Cuaderno.**

   ```bash
   uv run jupytext --to ipynb notebooks/fase-1-eda.py
   uv run jupyter nbconvert --to notebook --execute --inplace notebooks/fase-1-eda.ipynb
   ```

   La última celda imprime cuántas comprobaciones superó. Las cifras deben coincidir con las publicadas, salvo los tiempos y los pesos de archivo de la comparación de herramientas, que dependen de la máquina.

5. **Pruebas.**

   ```bash
   uv run pytest -q
   ```

## Fuente de los datos y licencias

Los datos son del Instituto Colombiano para la Evaluación de la Educación (ICFES): Instituto Colombiano para la Evaluación de la Educación. (2023). *Resultados únicos Saber 11* [Conjunto de datos]. Datos Abiertos Colombia. https://www.datos.gov.co/Educaci-n/Resultados-nicos-Saber-11/kgxf-xxbe

Esos datos tienen licencia [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Los agregados de `data/agregados/`, que se derivan de ellos, conservan la misma licencia. El código del repositorio tiene licencia MIT (ver `LICENSE`).

El repositorio no contiene filas individuales: solo agregados con 10 o más estudiantes por celda, en atención a la Ley 1581 de 2012 de protección de datos personales.
