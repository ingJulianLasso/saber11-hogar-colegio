# Datos agregados del Saber 11

Estos archivos los escribe la sección 10 de `notebooks/fase-1-eda.ipynb` y son los que leerá el tablero desde el repositorio. No contienen filas individuales.

| Archivo | Una fila por | Columnas |
| --- | --- | --- |
| `serie-estrato-naturaleza.csv` | Periodo, estrato de la vivienda y naturaleza del colegio, de 2014-2 a 2022-2 | `periodo`, `estrato`, `naturaleza`, `n` y la media, la mediana, los cuartiles (`q1`, `q3`) y la desviación estándar del puntaje global |
| `departamento-20224.csv` | Departamento de la sede del colegio, en 2022-2 | `cod_depto` (código DANE), `departamento` y las mismas medidas |

Reglas:

- Solo entran los resultados en estado `PUBLICAR` y las filas con las dimensiones de cada archivo registradas. «Sin Estrato» es una categoría del formulario y se conserva.
- **Cada celda tiene 10 o más estudiantes, por diseño y sin supresión.** Una celda oculta se podría recuperar restando de los totales, así que los agregados se arman con dimensiones que no dejan celdas pequeñas. El cuaderno se detiene si alguna no llega al mínimo, y `tests/test_agregados.py` lo vuelve a comprobar.
- Las medidas se redondean a dos decimales.

Fuente: Instituto Colombiano para la Evaluación de la Educación. (2023). *Resultados únicos Saber 11* [Conjunto de datos]. Datos Abiertos Colombia. https://www.datos.gov.co/Educaci-n/Resultados-nicos-Saber-11/kgxf-xxbe

Licencia: los datos del ICFES tienen licencia [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), y estos agregados, que se derivan de ellos, se comparten con la misma licencia y con atribución al ICFES.
