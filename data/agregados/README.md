# Datos agregados del Saber 11

Los dos primeros archivos los escribe la sección 10 de `notebooks/fase-1-eda.ipynb`; los seis siguientes, la sección 17 de `notebooks/fase-2-preparacion.ipynb`, y los diez últimos, la sección 21 de `notebooks/fase-3-modelado.ipynb`. Son los únicos datos que lee el dashboard (`dashboard/app.py`). No contienen filas individuales.

| Archivo | Una fila por | Columnas |
| --- | --- | --- |
| `serie-estrato-naturaleza.csv` | Periodo, estrato de la vivienda y naturaleza del colegio, de 2014-2 a 2022-2 | `periodo`, `estrato`, `naturaleza`, `n` y la media, la mediana, los cuartiles (`q1`, `q3`) y la desviación estándar del puntaje global |
| `departamento-20224.csv` | Departamento de la sede del colegio, en 2022-2 | `cod_depto` (código DANE), `departamento` y las mismas medidas |
| `estrato-naturaleza-zona-20224.csv` | Estrato (con «Sin dato»), naturaleza y zona del colegio, en 2022-2, más sus marginales con «Todos» | `estrato`, `naturaleza`, `zona` y las mismas medidas |
| `departamento-zona-20224.csv` | Departamento por zona, con la zona también en «Todos» | `cod_depto`, `departamento`, `zona` y las mismas medidas |
| `distribucion-puntaje-20224.csv` | Intervalo de 10 puntos del puntaje global por naturaleza y zona (con sus marginales); los extremos quedan abiertos hasta que todas las celdas llegan a 10 | `desde`, `hasta`, `intervalo`, `naturaleza`, `zona` y `n` |
| `regresion-coeficientes.csv` | Término de la regresión múltiple de la fase 2 | `termino`, `bloque`, `variable`, `nivel`, `coeficiente`, `ic_inferior` e `ic_superior` (IC clásico del 95 %), `ic_agrupado_inferior` e `ic_agrupado_superior` (IC del 95 % con errores agrupados por departamento, vacío en los términos del propio departamento) y `n`, los estudiantes de entrenamiento con ese nivel o ese faltante |
| `regresion-grupos.csv` | Estrato declarado y naturaleza, en el conjunto de validación | `n`, puntaje medio `observado` y `predicho`, y `rmse` |
| `regresion-metricas.csv` | Modelo de regresión y conjunto (entrenamiento, validación y prueba) | `n`, `r2`, `rmse` y `mae` |
| `hipotesis-medias-20224.csv` | Grupo de H1 (internet en el hogar) y de H2 (educación de la madre), en 2022-2, incluidos los grupos descriptivos que no entran en la prueba («Sin dato» y, en H2, «No sabe» y «No Aplica») | `hipotesis`, `variable`, `orden` (el de la figura), `grupo`, `etiqueta`, `probado` (si el grupo entra en la prueba), `n` y, del puntaje global, `media`, `desviacion` e intervalo t del 95 % (`inferior`, `superior`) |
| `hipotesis-interaccion-20224.csv` | Estrato (con «Sin Estrato») y naturaleza del colegio, en 2022-2 (H3) | `estrato`, `naturaleza` y las mismas medidas |
| `hipotesis-tukey-20224.csv` | Par de niveles de la educación de la madre (45 pares) y método, Tukey o Games-Howell (H2) | `metodo`, `grupo_a`, `grupo_b`, `par`, `consecutivo` (si los dos niveles son vecinos en la escala), `diferencia` de medias (b menos a), intervalo simultáneo del 95 % (`inferior`, `superior`), `p` ajustado y `n`, el del menor de los dos grupos |
| `hipotesis-residuos-20224.csv` | Celda de las dos tablas de H4: estrato por naturaleza del colegio e internet por zona | `tabla`, `fila`, `columna`, `n` y `residuo` estandarizado ajustado |
| `pca-celdas-20224.csv` | Celda no vacía del plano de los dos primeros componentes del PCA: rejilla de 10 por 10 entre los percentiles 2 y 98 de cada eje, cuyas celdas extremas también recogen a quienes quedan más afuera | `celda_x`, `celda_y`, sus límites (`x_inferior`, `x_superior`, `y_inferior`, `y_superior`), `n`, el `cluster` de K-Means mayoritario en la celda y su `proporcion` |
| `pca-cargas-20224.csv` | Componente del PCA (cinco) y área (cinco) | `componente`, `area`, `carga`, `varianza` explicada por el componente (en porcentaje) y `n` |
| `cluster-perfiles-20224.csv` | Clúster de K-Means | `cluster`, `n`, `porcentaje` de los estudiantes, media de cada área en la escala de la prueba (`lectura_critica`, `matematicas`, `ciencias_naturales`, `sociales_ciudadanas`, `ingles`) y estandarizada (las mismas con prefijo `z_`), y `puntaje_global` medio |
| `cluster-estrato-20224.csv`, `cluster-zona-20224.csv`, `cluster-naturaleza-20224.csv` | Nivel del estrato (con «Sin Estrato»), de la zona o de la naturaleza del colegio, y clúster de K-Means | La dimensión (`estrato`, `zona` o `naturaleza`), `cluster`, `n` y `porcentaje` de los estudiantes del nivel que caen en el clúster |

Los archivos de pruebas de hipótesis resumen los casos completos de cada prueba sobre el 2022-2 depurado en la fase 2, con los valores observados, sin imputar; los de PCA y clústeres, los estudiantes del conjunto de entrenamiento con las cinco áreas.

Reglas:

- Solo entran los resultados en estado `PUBLICAR` y las filas con las dimensiones de cada archivo registradas. «Sin Estrato» es una categoría del formulario y se conserva.
- **Cada archivo es un cruce completo con 10 o más estudiantes en todas sus celdas, por diseño y sin supresión.** Una celda oculta se podría recuperar restando de los totales, así que el cruce que no cumple no se publica: por eso no hay un archivo de departamento por naturaleza, y el dashboard dice que esa combinación no está publicada. Los cuadernos se detienen si una celda no llega al mínimo, y `tests/test_agregados.py` lo vuelve a comprobar. La rejilla del plano PCA no es un producto completo porque sus celdas vacías no tienen estudiantes y no se escriben; que no falte ninguna lo prueba la suma de los `n`, igual a todos los estudiantes del plano.
- Las medidas se redondean a dos decimales; las métricas de `regresion-metricas.csv` y los archivos de la fase 3, a cuatro, para que el dashboard, que los muestra con uno o dos, no redondee dos veces.

Fuente: Instituto Colombiano para la Evaluación de la Educación. (2023). *Resultados únicos Saber 11* [Conjunto de datos]. Datos Abiertos Colombia. https://www.datos.gov.co/Educaci-n/Resultados-nicos-Saber-11/kgxf-xxbe

Licencia: los datos del ICFES tienen licencia [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), y estos agregados, que se derivan de ellos, se comparten con la misma licencia y con atribución al ICFES.
