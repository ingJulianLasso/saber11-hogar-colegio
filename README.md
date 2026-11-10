# Hogar, colegio y puntaje en el Saber 11

Proyecto integrador de Ciencia de Datos de la Maestría en Inteligencia Artificial de la Universidad de La Salle (Bogotá). Estudia en qué medida las condiciones socioeconómicas del hogar (estrato, educación de los padres, internet y computador) y las características del colegio (naturaleza, zona y jornada) explican y permiten anticipar el puntaje global del examen Saber 11 de 2022-2, con una serie de 2014-2 a 2022-2 como contexto.

El trabajo avanza en cuatro fases. La fase 1 (etiqueta `entregable-1`) contiene el planteamiento, la ficha del conjunto de datos y el análisis exploratorio, con su réplica en R. La fase 2 (etiqueta `entregable-2`) prepara los datos, fija los modelos base y ajusta la regresión simple y múltiple del puntaje global en statsmodels, scikit-learn y R, con la primera parte del dashboard.

Dashboard: https://saber11-hogar-colegio.streamlit.app

## Estructura

| Carpeta o archivo | Contenido |
| --- | --- |
| `notebooks/fase-1-eda.ipynb` | Cuaderno ejecutado de la fase 1, con todas sus salidas. Su fuente es `notebooks/fase-1-eda.py` (formato jupytext). |
| `notebooks/fase-2-preparacion.ipynb` | Cuaderno ejecutado de la fase 2: exclusiones, mecanismo de los faltantes, imputación, atípicos, variables derivadas, partición, laboratorio descriptivo, modelos base, regresión con sus supuestos, paridad con R, comparación de herramientas y agregados del dashboard. Su fuente es `notebooks/fase-2-preparacion.py`. |
| `src/saber11/` | Paquete del proyecto: carga reproducible de los datos (`datos.py`), diccionario de las 51 columnas (`diccionario.py`), registro de las cifras que citan los informes (`cifras.py`) y, desde la fase 2, las reglas de limpieza (`limpieza.py`), los predictores y transformadores (`variables.py`), la partición recuperable (`particion.py`), los `Pipeline` y las métricas (`modelos.py`), los archivos intermedios que lee R (`intermedios.py`), el estilo de las figuras (`graficos.py`) y las comprobaciones de los cuadernos (`comprobaciones.py`). |
| `src/r/` | `instalar.R` instala los paquetes de R y registra sus versiones; `fase-1.R` replica en R los descriptivos, las frecuencias, las correlaciones y las pruebas χ² del cuaderno y dibuja sus gráficos en ggplot2; `fase-2.R` replica con `lm` la regresión simple y la múltiple y el laboratorio descriptivo. |
| `tests/` | Pruebas automáticas con pytest. |
| `data/agregados/` | Datos agregados que lee el dashboard. Ver su propio README. |
| `data/saber11-20224.parquet`, `data/saber11-serie.parquet` | Recortes del conjunto de datos que leen el cuaderno y el script de R: 2022-2 y la serie de 2014-2 a 2022-2. Vienen con el repositorio, así que no hace falta descargar nada. |
| `figuras/fase-1/`, `figuras/fase-2/` | Figuras de cada fase (las de R en `r/`; la comparación de herramientas de la fase 2 en `herramientas/`, con sus HTML de Plotly). |
| `resultados/` | Cifras de cada fase (`cifras-fase-N.json` y `.tex`), salidas de los scripts de R (`r/`), niveles de referencia de la regresión (`fase-2-niveles.json`) y el registro de evaluaciones en prueba (`evaluaciones-prueba.json`). |
| `dashboard/` | Dashboard en Streamlit (`app.py`) con sus dependencias propias (`requirements.txt`). |
| `pyproject.toml`, `uv.lock`, `requirements.txt` | Entorno de Python con versiones exactas. `r-versiones.txt` tiene las de R. |

## Reproducir la fase 1

Los pasos van en este orden, porque cada uno usa la salida del anterior: entorno, R y cuaderno. El paso de datos es opcional, porque los recortes ya vienen en `data/`. Todos se corren desde la raíz del repositorio.

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

2. **Datos (opcional).** Los recortes ya están en `data/`. Para regenerarlos desde el portal de datos abiertos y verificarlos contra los conteos esperados:

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

## Reproducir la fase 2

Con el entorno y los paquetes de R de la fase 1, los pasos van en este orden: intermedios, R, cuaderno y pruebas.

1. **Intermedios para R.** Escribe el entrenamiento ya preparado (`data/intermedio/fase-2-entrenamiento.parquet`, ignorado por git) y los niveles de referencia de la regresión:

   ```bash
   uv run python -m saber11.intermedios
   ```

2. **R.**

   ```bash
   Rscript src/r/fase-2.R
   ```

   Escribe `resultados/r/fase-2.json` y las figuras de ggplot2. También corre con *Source* en RStudio.

3. **Cuaderno.**

   ```bash
   uv run jupytext --to ipynb notebooks/fase-2-preparacion.py
   uv run jupyter nbconvert --to notebook --execute --inplace notebooks/fase-2-preparacion.ipynb
   ```

   La partición se recupera sin identificador: su huella (SHA-256) está en `resultados/cifras-fase-2.json` como `huellaParticion`, y `tests/test_particion.py` la recalcula.

4. **Dashboard en local.**

   ```bash
   uv run streamlit run dashboard/app.py
   ```

5. **Pruebas.**

   ```bash
   uv run pytest -q
   ```

## Fuente de los datos y licencias

Los datos son del Instituto Colombiano para la Evaluación de la Educación (ICFES): Instituto Colombiano para la Evaluación de la Educación. (2023). *Resultados únicos Saber 11* [Conjunto de datos]. Datos Abiertos Colombia. https://www.datos.gov.co/Educaci-n/Resultados-nicos-Saber-11/kgxf-xxbe

Esos datos tienen licencia [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Los recortes de `data/` y los agregados de `data/agregados/`, que se derivan de ellos, conservan la misma licencia, con atribución al ICFES. El código del repositorio tiene licencia MIT (ver `LICENSE`).

Los recortes de `data/` tienen una fila por estudiante, tal como las publica el ICFES, sin nombre, documento, colegio ni municipio: conservan 27 de las 51 columnas y solo los resultados en estado `PUBLICAR`, sin las filas duplicadas del portal. El cuaderno no imprime filas individuales, y los agregados tienen 10 o más estudiantes por celda, en atención a la Ley 1581 de 2012 de protección de datos personales.
