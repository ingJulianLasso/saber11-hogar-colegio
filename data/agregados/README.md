# Datos agregados del Saber 11

Los dos primeros archivos los escribe la sección 10 de `notebooks/fase-1-eda.ipynb`, y los seis siguientes, la sección 17 de `notebooks/fase-2-preparacion.ipynb`. Son los únicos datos que lee el dashboard (`dashboard/app.py`). No contienen filas individuales.

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

Reglas:

- Solo entran los resultados en estado `PUBLICAR` y las filas con las dimensiones de cada archivo registradas. «Sin Estrato» es una categoría del formulario y se conserva.
- **Cada archivo es un cruce completo con 10 o más estudiantes en todas sus celdas, por diseño y sin supresión.** Una celda oculta se podría recuperar restando de los totales, así que el cruce que no cumple no se publica: por eso no hay un archivo de departamento por naturaleza, y el dashboard dice que esa combinación no está publicada. Los cuadernos se detienen si una celda no llega al mínimo, y `tests/test_agregados.py` lo vuelve a comprobar.
- Las medidas se redondean a dos decimales; las métricas de `regresion-metricas.csv`, a cuatro.

Fuente: Instituto Colombiano para la Evaluación de la Educación. (2023). *Resultados únicos Saber 11* [Conjunto de datos]. Datos Abiertos Colombia. https://www.datos.gov.co/Educaci-n/Resultados-nicos-Saber-11/kgxf-xxbe

Licencia: los datos del ICFES tienen licencia [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), y estos agregados, que se derivan de ellos, se comparten con la misma licencia y con atribución al ICFES.
