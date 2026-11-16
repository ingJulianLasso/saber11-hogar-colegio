# Hogar, colegio y puntaje en el Saber 11

Proyecto integrador de Ciencia de Datos de la Maestría en Inteligencia Artificial de la Universidad de La Salle (Bogotá). Estudia en qué medida las condiciones socioeconómicas del hogar (estrato, educación de los padres, internet y computador) y las características del colegio (naturaleza, zona y jornada) explican y permiten anticipar el puntaje global del examen Saber 11 de 2022-2, con una serie de 2014-2 a 2022-2 como contexto.

El trabajo avanza en cuatro fases. La fase 1 (etiqueta `entregable-1`) contiene el planteamiento, la ficha del conjunto de datos y el análisis exploratorio, con su réplica en R. La fase 2 (etiqueta `entregable-2`) prepara los datos, fija los modelos base y ajusta la regresión simple y múltiple del puntaje global en statsmodels, scikit-learn y R, con la primera parte del dashboard. La fase 3 (etiqueta `entregable-3`) somete a prueba cuatro hipótesis con sus tamaños de efecto en scipy.stats, statsmodels y R; compara, para el puntaje global, la regresión regularizada, Random Forest, XGBoost y un perceptrón multicapa en PyTorch y, para la naturaleza del colegio, cuatro clasificadores; explica el mejor modelo con SHAP y permutación; segmenta a los estudiantes por su desempeño en las cinco áreas con PCA, K-Means, Ward y DBSCAN, y suma al dashboard las pestañas de pruebas de hipótesis y de segmentación. La fase 4 (etiqueta `entregable-4`) reúne en un cuaderno de síntesis las cifras de las tres fases, las métricas de prueba de los quince modelos (estadística clásica frente a aprendizaje automático) y la paridad con R; exporta un modelo compacto de predicción, y lleva el dashboard a su versión final, con resumen ejecutivo, predicción en vivo, método y lecturas por audiencia. Un solo comando reproduce todo (ver «Reproducir todo»).

Dashboard: https://saber11-hogar-colegio.streamlit.app

## Estructura

| Carpeta o archivo | Contenido |
| --- | --- |
| `notebooks/fase-1-eda.ipynb` | Cuaderno ejecutado de la fase 1, con todas sus salidas. Su fuente es `notebooks/fase-1-eda.py` (formato jupytext). |
| `notebooks/fase-2-preparacion.ipynb` | Cuaderno ejecutado de la fase 2: exclusiones, mecanismo de los faltantes, imputación, atípicos, variables derivadas, partición, laboratorio descriptivo, modelos base, regresión con sus supuestos, paridad con R, comparación de herramientas y agregados del dashboard. Su fuente es `notebooks/fase-2-preparacion.py`. |
| `notebooks/fase-3-modelado.ipynb` | Cuaderno ejecutado de la fase 3: pruebas de hipótesis con sus supuestos, tamaños de efecto y paridad con R; regresión regularizada, Random Forest y XGBoost con búsqueda de hiperparámetros; clasificación; perceptrón multicapa; SHAP, permutación y errores por grupo; PCA, K-Means, Ward y DBSCAN con su paridad con R; comparación de herramientas; agregados del dashboard, y la evaluación única en prueba. Su fuente es `notebooks/fase-3-modelado.py`. |
| `notebooks/fase-4-sintesis.ipynb` | Cuaderno ejecutado de la fase 4: cifras consolidadas de las fases 1 a 3, métricas de prueba de los quince modelos, estadística clásica frente a aprendizaje automático, paridad entre Python y R consolidada, agregado por jornada, modelo de predicción del tablero con su margen, memoria del dashboard y tamaño de efecto frente a SHAP. No vuelve a evaluar nada en prueba. Su fuente es `notebooks/fase-4-sintesis.py`. |
| `src/saber11/` | Paquete del proyecto: carga reproducible de los datos (`datos.py`), diccionario de las 51 columnas (`diccionario.py`), registro de las cifras que citan los informes (`cifras.py`) y, desde la fase 2, las reglas de limpieza (`limpieza.py`), los predictores y transformadores (`variables.py`), la partición recuperable (`particion.py`), los `Pipeline` y las métricas (`modelos.py`), los archivos intermedios que lee R (`intermedios.py`), el estilo de las figuras (`graficos.py`) y las comprobaciones de los cuadernos (`comprobaciones.py`); desde la fase 3, las pruebas de hipótesis con sus tamaños de efecto (`hipotesis.py`), la búsqueda de hiperparámetros y su caché (`ajuste.py`), la comparación estadística de modelos (`comparacion.py`), el PCA y la segmentación (`segmentacion.py`), el perceptrón multicapa (`mlp.py`) y SHAP, la permutación y los errores por grupo (`interpretabilidad.py`); desde la fase 4, el margen, los controles de privacidad y el formulario del modelo del tablero (`tablero.py`), las capturas del dashboard (`capturas.py`) y la reproducción completa (`reproducir.py`). |
| `src/r/` | `instalar.R` instala los paquetes de R y registra sus versiones; `fase-1.R` replica en R los descriptivos, las frecuencias, las correlaciones y las pruebas χ² del cuaderno y dibuja sus gráficos en ggplot2; `fase-2.R` replica con `lm` la regresión simple y la múltiple y el laboratorio descriptivo; `fase-3-hipotesis.R` replica las cuatro pruebas de hipótesis con `t.test`, `oneway.test`, `aov` y `TukeyHSD`, `kruskal.test`, `car::Anova` y `chisq.test`, y `fase-3-multivariante.R`, el PCA con `prcomp`, las medias por clúster y el reparto de los clústeres; los dos grafican en ggplot2. |
| `tests/` | Pruebas automáticas con pytest. |
| `data/agregados/` | Datos agregados que lee el dashboard. Ver su propio README. |
| `data/saber11-20224.parquet`, `data/saber11-serie.parquet` | Recortes del conjunto de datos que leen los cuadernos y los scripts de R: 2022-2 y la serie de 2014-2 a 2022-2. Vienen con el repositorio, así que no hace falta descargar nada. |
| `figuras/fase-1/` a `figuras/fase-4/` | Figuras de cada fase (las de R en `r/`; la comparación de herramientas de las fases 2 y 3 en `herramientas/`, la de la fase 2 con sus HTML de Plotly; las versiones interactivas de la fase 3, en HTML de Plotly, en `interactivas/`; en la fase 4, la comparación con y sin IA, el tamaño de efecto frente a SHAP y una captura por pestaña del dashboard, `tablero-*.png`). |
| `resultados/` | Cifras de cada fase (`cifras-fase-N.json` y `.tex`; `cifras-fase-4` reúne las de las cuatro fases en un solo archivo de macros), la tabla de paridad consolidada (`paridad-consolidada.tex`), salidas de los scripts de R (`r/`), niveles de referencia de la regresión (`fase-2-niveles.json`) y los registros de evaluaciones en prueba de la fase 2 (`evaluaciones-prueba.json`) y de la fase 3 (`evaluaciones-prueba-fase-3.json`). |
| `dashboard/` | Dashboard en Streamlit (`app.py`) con siete pestañas: Resumen, Exploración (panorama, serie, territorio y jornada), Pruebas de hipótesis, Modelos (regresión, con y sin IA, clasificación e interpretabilidad), Predicción, Segmentación y Método y fuente, más un selector que adapta las lecturas a cuatro audiencias. Sus dependencias directas están en `requirements.in` y todas, con versión exacta, en `requirements.txt`; el modelo de la pestaña Predicción y su ficha (variables, niveles admitidos, margen y versiones), en `modelo/`. |
| `pyproject.toml`, `uv.lock`, `requirements.txt` | Entorno de Python con versiones exactas. `r-versiones.txt` tiene las de R. |

## Instalación

Hacen falta [uv](https://docs.astral.sh/uv/) y [R](https://cran.r-project.org/) 4.6, con `Rscript` en la ruta de ejecutables (`PATH`). Python 3.13 no hace falta instalarlo aparte: si la máquina no lo tiene, uv lo descarga.

| Sistema | Qué instala y cómo corre |
| --- | --- |
| Linux con GPU de NVIDIA | PyTorch 2.11 con CUDA 13.0, que necesita un controlador compatible con esa versión. XGBoost y la red neuronal de la fase 3 entrenan en la GPU de más memoria. Es la configuración con la que se produjeron los resultados (RTX 5080). |
| Linux sin GPU | La misma rueda de PyTorch, que sin GPU corre en la CPU. Los cuadernos lo detectan solos. |
| Windows | La rueda de PyTorch con CUDA 13.0, en la GPU de NVIDIA si la hay y, si no, en la CPU. El instalador de R no agrega `Rscript` al `PATH`: hay que agregar la carpeta `bin` de R. |
| macOS con Apple silicon | PyTorch 2.11 desde PyPI (el índice de PyTorch con CUDA no publica ruedas para macOS); todo corre en la CPU. |
| macOS con procesador Intel | No soportado: PyTorch 2.11 no publica ruedas para esa plataforma. |

Sin GPU, XGBoost y la red neuronal tardan más y sus cifras pueden diferir un poco de las publicadas. En Windows y macOS las versiones del entorno resuelven (comprobado con `uv pip compile --python-platform`), pero los cuadernos no se han ejecutado en esos sistemas.

1. **Entorno de Python.** Desde la raíz del repositorio:

   ```bash
   uv sync --locked
   ```

   `--locked` instala exactamente las versiones de `uv.lock` y falla si el archivo no corresponde a `pyproject.toml`. O con pip, en un entorno virtual nuevo de Python 3.13:

   ```bash
   pip install -r requirements.txt
   pip install -e . --no-deps
   ```

   La segunda línea instala el paquete `saber11` del proyecto, que `requirements.txt` no incluye.

2. **Paquetes de R.** Una sola vez:

   ```bash
   Rscript src/r/instalar.R
   ```

   Instala en la biblioteca de usuario los paquetes que falten y escribe en `r-versiones.txt` las versiones de esa máquina, así que `git diff r-versiones.txt` muestra si difieren de las publicadas. En Windows y macOS, CRAN entrega los paquetes compilados; en Linux se compilan, y en Arch necesitan `r`, `gcc-fortran` y `nlopt` (en otras distribuciones, sus equivalentes).

## Datos

Los recortes ya están en `data/` y reproducir el proyecto no descarga nada. Para regenerarlos desde el portal de datos abiertos y verificar los conteos de cada periodo:

```bash
uv run python -m saber11.datos --fuente api --verificar --materializar
```

El comando baja por la API los seis periodos del proyecto (4.352.174 filas, porque el portal publica filas duplicadas que el paquete quita), comprueba las filas únicas y las que quedan fuera del estado `PUBLICAR` de cada periodo, deja solo los resultados publicados y escribe `data/saber11-20224.parquet` y `data/saber11-serie.parquet`. En una prueba del 8 de noviembre de 2026, el 2022-2 (1.065.888 filas) bajó en 2 min 30 s y su recorte salió idéntico, fila por fila, al del repositorio; con una conexión parecida, los seis periodos toman unos 10 minutos. Si ya se tiene la exportación completa del portal convertida a Parquet, `--fuente local --parquet <ruta>` hace lo mismo sin descargar.

## Reproducir todo

Con el entorno y los paquetes de R instalados, un solo comando corre el proyecto de principio a fin, en el orden de las secciones siguientes:

```bash
uv run python -m saber11.reproducir
```

Son doce pasos: la comprobación de los recortes de `data/` (las filas de cada periodo deben ser las únicas del portal menos las que quedan fuera de `PUBLICAR`); en cada fase, los intermedios, los scripts de R y el cuaderno, que jupytext convierte a `.ipynb` y nbconvert ejecuta en su lugar; el cuaderno de síntesis de la fase 4, y las pruebas. El comando se detiene en el primer paso que falla, porque cada uno usa la salida de los anteriores, y termina con el tiempo de cada paso. Con una RTX 5080, el cuaderno de la fase 3 tarda unos 50 minutos y domina el total.

- `--api` antepone la descarga de la sección «Datos»: regenera los recortes y verifica los conteos del portal.
- `--lista` imprime los pasos con sus comandos, sin ejecutarlos.
- `--recortes` solo comprueba los recortes de `data/`.

Al terminar, `git status` muestra lo que se regeneró (cuadernos, cifras, figuras, agregados y el modelo del tablero) y `git diff` deja comparar con lo publicado. Las cifras deben coincidir, salvo los tiempos y los pesos de archivo de las comparaciones de herramientas y la memoria del dashboard (claves que empiezan por `medida`), y algunas del Random Forest desde el octavo dígito significativo, porque las predicciones de sus árboles se suman en paralelo.

## Reproducir la fase 1

Con el entorno y los paquetes de R instalados, los pasos van en este orden, porque el cuaderno compara sus valores con los de R. Todos se corren desde la raíz del repositorio.

1. **R.**

   ```bash
   Rscript src/r/fase-1.R
   ```

   El script escribe `resultados/r/fase-1.json`, que el cuaderno compara con sus propios valores.

2. **Cuaderno.**

   ```bash
   uv run jupytext --to ipynb notebooks/fase-1-eda.py
   uv run jupyter nbconvert --to notebook --execute --inplace notebooks/fase-1-eda.ipynb
   ```

   La última celda imprime cuántas comprobaciones superó. Las cifras deben coincidir con las publicadas, salvo los tiempos y los pesos de archivo de la comparación de herramientas, que dependen de la máquina.

3. **Pruebas.**

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

## Reproducir la fase 3

Con el entorno y los paquetes de R de la fase 1, los pasos van en este orden: intermedios, R, cuaderno y pruebas. No hace falta volver a ejecutar el cuaderno de la fase 2: el de la fase 3 recupera la partición y la muestra de entrenamiento por su huella y lee las cifras de la fase 2 desde `resultados/`.

1. **Intermedios para R.** El primer comando es el de la fase 2. El segundo escribe las variables de las cuatro hipótesis con su marco (niveles y orden) y, como `segmentacion.K_KMEANS` ya está fijada, las áreas estandarizadas del entrenamiento con su clúster y las celdas del plano PCA, todo en `data/intermedio/` (ignorada por git):

   ```bash
   uv run python -m saber11.intermedios
   uv run python -m saber11.intermedios --fase 3
   ```

2. **R.**

   ```bash
   Rscript src/r/fase-3-hipotesis.R
   Rscript src/r/fase-3-multivariante.R
   ```

   Escriben `resultados/r/fase-3-hipotesis.json` y `resultados/r/fase-3-multivariante.json`, que el cuaderno compara con sus propios valores, y las figuras de ggplot2 en `figuras/fase-3/r/`. También corren con *Source* en RStudio.

3. **Cuaderno.**

   ```bash
   uv run jupytext --to ipynb notebooks/fase-3-modelado.py
   uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/fase-3-modelado.ipynb
   ```

   Con una GPU de NVIDIA, XGBoost y la red neuronal entrenan en la de más memoria; sin ella, en CPU, con más tiempo y diferencias menores en sus cifras. La primera ejecución tarda unos 50 minutos con una RTX 5080. Las búsquedas de hiperparámetros, los intervalos bootstrap y el entrenamiento de la red quedan en `data/intermedio/cache/` (ignorada por git), y con ella una nueva ejecución tarda unos 4 minutos; la caché lleva la versión del código en su clave, así que un cambio en `src/saber11/` la recalcula. Las cifras deben coincidir con las publicadas, salvo los tiempos y los pesos de archivo de la comparación de herramientas; algunas del Random Forest pueden variar desde el octavo dígito significativo, porque las predicciones de sus árboles se suman en paralelo. Cada modelo final se evalúa una sola vez en el conjunto de prueba (sección 22), en el registro `resultados/evaluaciones-prueba-fase-3.json`, que empieza vacío en cada ejecución.

4. **Dashboard en local.** Con el mismo comando de la fase 2. Las pestañas «Pruebas de hipótesis» y «Segmentación» leen los diez agregados nuevos de `data/agregados/`.

5. **Pruebas.**

   ```bash
   uv run pytest -q
   ```

   Entre ellas, `tests/test_prueba_unica.py` comprueba que ningún modelo se evalúe dos veces en prueba y que el cuaderno solo use ese conjunto donde lo recupera (sección 1) y donde evalúa (sección 22).

## Reproducir la fase 4

La fase 4 no corre scripts de R ni vuelve a calcular lo de las fases anteriores: su cuaderno lee las cifras, los registros de prueba y las salidas de R que dejaron en `resultados/`, así que basta con que existan.

1. **Cuaderno de síntesis.**

   ```bash
   uv run jupytext --to ipynb notebooks/fase-4-sintesis.py
   uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/fase-4-sintesis.ipynb
   ```

   No necesita GPU. Escribe `resultados/cifras-fase-4.json` y `.tex` (las cifras de las cuatro fases, con un solo nombre por cifra), `resultados/paridad-consolidada.tex`, los agregados `metricas-modelos.csv` y `jornada-estrato-naturaleza-20224.csv` y las figuras de `figuras/fase-4/`. También elige el modelo del tablero (el de menor RMSE de validación entre los que no guardan observaciones), lo reajusta en la CPU con hojas de 10 o más estudiantes, calcula su margen con los residuos de validación y lo exporta a `dashboard/modelo/` con su ficha. El reajuste se guarda en la caché de `data/intermedio/cache/`.

2. **Dashboard en local.** Con el entorno del proyecto:

   ```bash
   uv run streamlit run dashboard/app.py
   ```

   O, en Linux, con el entorno liviano que instala Streamlit Community Cloud, que no incluye PyTorch ni el paquete `saber11` (la app agrega `src/` a la ruta de importación):

   ```bash
   pip install -r dashboard/requirements.txt
   streamlit run dashboard/app.py
   ```

   `dashboard/requirements.txt` se compila para Linux, como el servidor de Streamlit Community Cloud, desde `dashboard/requirements.in` y con las versiones del entorno principal, para que el modelo se cargue con las mismas versiones con que se entrenó (el comando está en la cabecera de los dos archivos). En Windows y macOS, la app corre con el entorno del proyecto.

3. **Capturas del dashboard** (opcional; necesita Chrome o Chromium, y la variable `BROWSER_PATH` elige otro ejecutable). Levanta la app y guarda una captura por pestaña en `figuras/fase-4/tablero-*.png`:

   ```bash
   uv run python -m saber11.capturas
   ```

4. **Pruebas.** Las del proyecto y, aparte y en Linux, las del dashboard en su entorno liviano y sin GPU:

   ```bash
   uv run pytest -q
   CUDA_VISIBLE_DEVICES= uv run --isolated --no-project --python 3.13 --with-requirements dashboard/requirements.txt --with pytest pytest -q tests/test_dashboard.py
   ```

   `tests/test_modelo_tablero.py` comprueba que el modelo exportado pese menos de 50 MB, no guarde filas de estudiantes, tenga hojas de 10 o más estudiantes, prediga sin GPU con las versiones de su ficha y dé una estimación entre 0 y 500.

## Enlaces

- Dashboard: https://saber11-hogar-colegio.streamlit.app
- Datos: https://www.datos.gov.co/Educaci-n/Resultados-nicos-Saber-11/kgxf-xxbe
- uv: https://docs.astral.sh/uv/
- R: https://cran.r-project.org/

## Fuente de los datos y licencias

Los datos son del Instituto Colombiano para la Evaluación de la Educación (ICFES): Instituto Colombiano para la Evaluación de la Educación. (2023). *Resultados únicos Saber 11* [Conjunto de datos]. Datos Abiertos Colombia. https://www.datos.gov.co/Educaci-n/Resultados-nicos-Saber-11/kgxf-xxbe

Esos datos tienen licencia [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Los recortes de `data/`, los agregados de `data/agregados/` y el modelo de `dashboard/modelo/`, que se derivan de ellos, conservan la misma licencia, con atribución al ICFES. El código del repositorio tiene licencia MIT (ver `LICENSE`).

Los recortes de `data/` tienen una fila por estudiante, tal como las publica el ICFES, sin nombre, documento, colegio ni municipio: conservan 27 de las 51 columnas y solo los resultados en estado `PUBLICAR`, sin las filas duplicadas del portal. Los cuadernos no imprimen filas individuales, ninguna figura dibuja un punto por estudiante (el biplot y el plano del PCA van por celdas, y el dendrograma, por hojas, todas de 10 o más) y los agregados tienen 10 o más estudiantes por celda, en atención a la Ley 1581 de 2012 de protección de datos personales. El único modelo entrenado que publica el repositorio es el del tablero (`dashboard/modelo/regresion.joblib`): un XGBoost que no guarda observaciones y cuyas hojas cubren 10 o más estudiantes de entrenamiento cada una.
