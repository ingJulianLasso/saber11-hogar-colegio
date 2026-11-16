# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Actividades 7, 8 y 9. Síntesis del proyecto, modelo del tablero y cifras finales
#
# **Actividad 7:** Aplicaciones Prácticas en el Mundo Real  
# **Actividad 8:** Integración de Inteligencia Artificial en Análisis de Datos  
# **Actividad 9:** Evaluación final. Presentación Final del Proyecto de Ciencia de Datos  
# **Proyecto integrador:** fase 4, visualización y sustentación  
# **Autor:** Julián Andrés Lasso Figueroa  
# **Correo institucional:** jlasso96@unisalle.edu.co  
# **Programa:** Maestría en Inteligencia Artificial, Universidad de La Salle  
# **Asignatura:** Ciencia de Datos, Unidad 3, estudio de casos y proyectos aplicados  
# **Docente:** Fabián Camilo Castro Riveros  
# **Fecha:** noviembre de 2026
#
# ---

# %% [markdown]
# ## 0. Planteamiento y configuración
#
# ### 0.1. Qué hace este cuaderno
#
# Las fases 1 a 3 dejaron sus resultados en archivos: las cifras de cada cuaderno (`resultados/cifras-fase-{1,2,3}.json`), los dos registros de evaluación en prueba, las salidas de los scripts de R y los agregados del dashboard. Esta fase no vuelve a calcular nada de eso. El cuaderno es liviano y hace seis cosas:
#
# 1. Une las cifras de las tres fases en un solo registro, para que el documento del dashboard (A7), el informe final (A8) y las diapositivas (A9) carguen un único archivo de macros (§1).
# 2. Reúne las métricas de prueba de los quince modelos en un solo agregado, sin volver a evaluar ninguno (§2), y compara la estadística clásica con los modelos de aprendizaje automático (§3).
# 3. Consolida en una tabla la paridad entre Python y R de las tres fases (§4).
# 4. Agrega el puntaje por jornada, la característica del colegio que nombra la pregunta principal y que ningún agregado traía (§5).
# 5. Exporta el modelo de predicción del tablero, con su ficha y su margen, mide la memoria del dashboard (§6) y compara el tamaño de efecto de las hipótesis con la importancia SHAP de ese modelo (§7).
# 6. Escribe las cifras de la fase y comprueba que todo existe (última sección).
#
# Ninguna celda lee el conjunto de prueba: sus métricas salen de los registros que las fases 2 y 3 escribieron al evaluarlo una sola vez, y el modelo del tablero se elige y se mide en validación.

# %% [markdown]
# ### 0.2. Entorno
#
# El cuaderno corre con el entorno del repositorio (`uv sync`) después de los cuadernos de las fases 1 a 3 y de sus scripts de R, porque lee lo que ellos escriben (el orden está en el `README.md`). No usa la GPU.

# %%
import json
import platform
import warnings
from itertools import combinations
from importlib.metadata import version

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from saber11 import ajuste, datos, graficos, interpretabilidad, limpieza, modelos, particion, tablero
from saber11 import variables as v
from saber11.cifras import Cifras, en_espanol
from saber11.comprobaciones import Comprobaciones

print(f"{'Python':<14}{platform.python_version()}")
for paquete in ("pandas", "numpy", "scikit-learn", "xgboost", "joblib", "matplotlib"):
    print(f"{paquete:<14}{version(paquete)}")

# %% [markdown]
# ### 0.3. Estilo, rutas y comprobaciones
#
# Las figuras conservan la estética de las fases anteriores (`saber11.graficos`) y se dibujan al ancho de una columna IEEE, con la letra con que se imprimen. Las rutas se imprimen relativas a la raíz del repositorio. `comprobar` detiene el cuaderno si una condición falla.

# %%
graficos.aplicar_estilo(informe=True)
RAIZ = datos.RAIZ_PROYECTO
RUTA_FIGURAS = RAIZ / "figuras" / "fase-4"
RUTA_RESULTADOS = RAIZ / "resultados"
RUTA_AGREGADOS = datos.DESTINO / "agregados"
FUENTE = "Fuente: ICFES, Saber 11, 2022-2 (datos.gov.co)."
MINIMO_POR_CELDA = 10

figuras = graficos.Figuras(RUTA_FIGURAS)
control = Comprobaciones()
comprobar = control.comprobar


def relativa(ruta):
    """Ruta relativa a la raíz del repositorio, para no imprimir rutas locales."""
    return ruta.relative_to(RAIZ).as_posix()


def formatear(valor: object, decimales: int) -> object:
    """Número con coma decimal; los enteros, sin decimales."""
    if isinstance(valor, (bool, np.bool_)) or not isinstance(valor, (int, float, np.integer, np.floating)) or pd.isna(valor):
        return valor
    return en_espanol(valor, 0 if float(valor).is_integer() else decimales)


def tabla(marco: pd.DataFrame, decimales: int = 2) -> None:
    """Imprime una tabla con coma decimal."""
    print(marco.to_string(formatters={c: (lambda x, d=decimales: formatear(x, d)) for c in marco.columns}))


print(f"Figuras en {relativa(RUTA_FIGURAS)}; cifras en {relativa(RUTA_RESULTADOS)}; agregados en {relativa(RUTA_AGREGADOS)}")

# %% [markdown]
# ## 1. Cifras consolidadas de las fases 1 a 3
#
# Cada fase volcó sus cifras en un JSON y en un `.tex` con una macro de LaTeX por cifra. Cargar los tres `.tex` en un mismo documento fallaría, porque algunas claves se repiten y `\newcommand` no admite dos definiciones. `Cifras.consolidar` une los tres JSON con una regla: una clave que se repite con el mismo texto queda una vez, y una que cambia de texto entre fases sale solo con el sufijo de cada fase (`FaseUno`, `FaseDos` o `FaseTres`), para que ninguna macro tenga dos significados. Las cifras nuevas de esta fase se agregan a ese registro y se vuelcan al final en `resultados/cifras-fase-4.{json,tex}`, el único archivo de macros de los tres documentos.

# %%
RUTAS_CIFRAS = [RUTA_RESULTADOS / f"cifras-fase-{fase}.json" for fase in (1, 2, 3)]
por_fase = {ruta.stem: json.loads(ruta.read_text(encoding="utf-8")) for ruta in RUTAS_CIFRAS}
cifras = Cifras.consolidar(RUTAS_CIFRAS)

apariciones = pd.Series([clave for datos_fase in por_fase.values() for clave in datos_fase]).value_counts()
repetidas = apariciones[apariciones > 1].index
textos = pd.DataFrame({fase: {clave: datos_fase[clave]["texto"] for clave in repetidas if clave in datos_fase} for fase, datos_fase in por_fase.items()})
con_texto_distinto = textos[textos.nunique(axis=1) > 1]
claves_esperadas = len(apariciones) - len(con_texto_distinto) + int(apariciones[con_texto_distinto.index].sum())
print(pd.Series({fase: len(datos_fase) for fase, datos_fase in por_fase.items()}, name="Cifras").to_string())
print(f"\nClaves repetidas entre fases: {len(repetidas)}; con el mismo texto: {len(repetidas) - len(con_texto_distinto)}; con texto distinto: {len(con_texto_distinto)}")
print(con_texto_distinto.fillna("").to_string())
print(f"\nCifras consolidadas: {en_espanol(len(cifras))}")
comprobar("Las claves con texto distinto entre fases solo quedan con el sufijo de cada fase", not any(clave in cifras for clave in con_texto_distinto.index))
comprobar("El registro consolidado tiene una clave por cifra distinta de las tres fases", len(cifras) == claves_esperadas)
CONSOLIDADAS = len(cifras)


def cifra(clave: str, valor: float | str, decimales: int = 0) -> str:
    """Registra una cifra nueva de la fase 4; falla si la clave ya viene de una fase anterior."""
    if clave in cifras:
        raise ValueError(f"La macro {clave!r} ya existe en las cifras consolidadas")
    return cifras.cifra(clave, valor, decimales)


cifra("clavesConSufijoDeFase", len(con_texto_distinto));

# %% [markdown]
# **Lectura.** Las claves que cambian de texto entre fases no son errores: son cifras distintas que dos fases guardaron con el mismo nombre, como los valores comparados con R o los agregados nuevos de cada fase. En el registro consolidado quedan solo con el sufijo de su fase (por ejemplo, `\paridadValoresFaseUno` y `\paridadValoresFaseDos`), así que un documento que cite la versión sin sufijo no compila, en lugar de mostrar en silencio la cifra de otra fase.

# %% [markdown]
# ## 2. Métricas de prueba de los quince modelos
#
# La prueba se tocó una sola vez por modelo: la fase 2 evaluó sus cinco bases y la fase 3 sus diez modelos finales, y cada una guardó las métricas en su registro (`resultados/evaluaciones-prueba.json` y `resultados/evaluaciones-prueba-fase-3.json`). Aquí se leen esos registros, sin predecir nada. `Cifras.metricas_de_prueba` busca, para cada métrica, la macro que la fase ya escribió con el mismo valor (por ejemplo `\rmsePruebaXgboost`) y solo crea una nueva, con el nombre `<métrica>Prueba<Modelo>`, si la fase no la tenía. Así cada métrica tiene un solo nombre en los documentos.
#
# Los modelos se reparten en dos grupos para la A8, que pide visualizaciones «con y sin el uso de IA»:
#
# - **Sin IA (estadística clásica):** el modelo que predice la media o la clase mayoritaria (Dummy), la regresión lineal simple, la múltiple por mínimos cuadrados, la logística base y el análisis discriminante lineal (LDA). Todos se ajustan en forma cerrada o con un solo problema de optimización convexo, sin búsqueda de hiperparámetros; el LDA es el discriminante de Fisher, un método estadístico clásico, aunque se ajustó en la fase 3.
# - **Con IA (aprendizaje automático):** Ridge, Lasso y ElasticNet, cuya penalización se elige por validación cruzada; Random Forest y XGBoost, con búsqueda de hiperparámetros; la máquina de vectores de soporte (SVM) y la red neuronal (MLP).

# %%
REGISTROS_PRUEBA = [RUTA_RESULTADOS / "evaluaciones-prueba.json", RUTA_RESULTADOS / "evaluaciones-prueba-fase-3.json"]
SIN_IA, CON_IA = "Sin IA", "Con IA"
REGRESION, CLASIFICACION = "Regresión", "Clasificación"
MODELOS = pd.DataFrame(
    [
        ("dummyRegresion", REGRESION, SIN_IA, "Media del entrenamiento", 2),
        ("linealSimple", REGRESION, SIN_IA, "Lineal simple", 2),
        ("regresionMultiple", REGRESION, SIN_IA, "Regresión múltiple", 2),
        ("ridge", REGRESION, CON_IA, "Ridge", 3),
        ("lasso", REGRESION, CON_IA, "Lasso", 3),
        ("elasticNet", REGRESION, CON_IA, "ElasticNet", 3),
        ("randomForestRegresion", REGRESION, CON_IA, "Random Forest", 3),
        ("xgboostRegresion", REGRESION, CON_IA, "XGBoost", 3),
        ("mlpRegresion", REGRESION, CON_IA, "MLP", 3),
        ("dummyClasificacion", CLASIFICACION, SIN_IA, "Clase mayoritaria", 2),
        ("logisticaBase", CLASIFICACION, SIN_IA, "Logística base", 2),
        ("ldaClasificacion", CLASIFICACION, SIN_IA, "LDA", 3),
        ("randomForestClasificacion", CLASIFICACION, CON_IA, "Random Forest", 3),
        ("xgboostClasificacion", CLASIFICACION, CON_IA, "XGBoost", 3),
        ("svmClasificacion", CLASIFICACION, CON_IA, "SVM", 3),
    ],
    columns=["modelo", "tarea", "enfoque", "etiqueta", "fase"],
).set_index("modelo")
COLUMNA_METRICA = {"r2": "r2", "rmse": "rmse", "mae": "mae", "accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1", "f1Macro": "f1_macro", "auc": "auc", "aucPr": "auc_pr"}

registros = [fila for ruta in REGISTROS_PRUEBA for fila in json.loads(ruta.read_text(encoding="utf-8"))]
metricas_json = {fila["modelo"]: fila["metricas"] for fila in registros}
comprobar("Los dos registros de prueba tienen los quince modelos, sin repetir ninguno", len(registros) == len(metricas_json) == len(MODELOS) and set(metricas_json) == set(MODELOS.index))

antes_de_las_metricas = len(cifras)
macros_prueba: dict[tuple[str, str], str] = {}
for ruta in REGISTROS_PRUEBA:
    macros_prueba |= cifras.metricas_de_prueba(ruta)
creadas = len(cifras) - antes_de_las_metricas
comprobar("Cada métrica de prueba tiene una macro propia, con el valor del registro", len(set(macros_prueba.values())) == len(macros_prueba) and all(cifras.valor(clave) == metricas_json[modelo][metrica] for (modelo, metrica), clave in macros_prueba.items()))
print(f"Métricas de prueba: {len(macros_prueba)}; macros que ya existían en su fase: {len(macros_prueba) - creadas}; creadas aquí: {creadas}")
print(pd.DataFrame([{"Modelo": modelo, "Métrica": metrica, "Macro": f"\\{clave}", "Texto": cifras.texto(clave)} for (modelo, metrica), clave in macros_prueba.items()]).head(12).to_string(index=False))
cifra("metricasPruebaConsolidadas", len(macros_prueba))
cifra("modelosEvaluadosPrueba", len(MODELOS));

# %% [markdown]
# El agregado `data/agregados/metricas-modelos.csv` deja las métricas de prueba de los quince modelos en una tabla, con la tarea, el grupo (sin IA o con IA) y el número de estudiantes de prueba (`n`), que es el mismo en todos. Lo leen el dashboard (tarjetas y pestaña Modelos) y la figura de §3. Las métricas que una fase no calculó para un modelo (el F1 macro y el AUC-PR de las bases de la fase 2) quedan vacías.

# %%
FILAS_PRUEBA = int(cifras.valor("filasPrueba"))
metricas = MODELOS.join(pd.DataFrame.from_dict(metricas_json, orient="index").rename(columns=COLUMNA_METRICA)).reset_index()
metricas.insert(5, "n", FILAS_PRUEBA)
metricas = metricas[["tarea", "enfoque", "modelo", "etiqueta", "fase", "n", *COLUMNA_METRICA.values()]]
comprobar("Cada tarea tiene modelos sin IA y con IA", metricas.groupby("tarea")["enfoque"].nunique().eq(2).all())
comprobar("Los modelos de regresión tienen R², RMSE y MAE, y los de clasificación, exactitud, F1 y AUC", metricas.loc[metricas["tarea"] == REGRESION, ["r2", "rmse", "mae"]].notna().all().all() and metricas.loc[metricas["tarea"] == CLASIFICACION, ["accuracy", "f1", "auc"]].notna().all().all())
ruta_metricas = RUTA_AGREGADOS / "metricas-modelos.csv"
comprobar(f"metricas-modelos.csv: un modelo por fila y {MINIMO_POR_CELDA} o más estudiantes en cada una", not metricas["modelo"].duplicated().any() and metricas["n"].min() >= MINIMO_POR_CELDA)
metricas.round(6).to_csv(ruta_metricas, index=False)  # seis decimales: el dashboard redondea una sola vez
print(f"Escrito {relativa(ruta_metricas)}: {len(metricas)} filas, n = {en_espanol(FILAS_PRUEBA)}")
tabla(metricas.set_index("modelo")[["tarea", "enfoque", "r2", "rmse", "mae", "accuracy", "f1", "auc"]], 3)

# %% [markdown]
# ## 3. Estadística clásica frente a aprendizaje automático
#
# La figura pone lado a lado, en prueba, los modelos sin IA y con IA de las dos tareas: el RMSE de la regresión del puntaje global (en puntos; menor es mejor) y el AUC-ROC de la clasificación del colegio oficial o no oficial (mayor es mejor). Cada punto es un modelo, no un estudiante, y el eje horizontal no empieza en cero para que se vean las diferencias, que son de pocos puntos o centésimas.

# %%
COLOR_ENFOQUE = {SIN_IA: graficos.COLOR_APOYO, CON_IA: graficos.COLOR_PRINCIPAL}
ETIQUETA_ENFOQUE = {SIN_IA: "Sin IA: estadística clásica", CON_IA: "Con IA: aprendizaje automático"}
PANELES = (
    (REGRESION, "rmse", True, "Regresión del puntaje global", "RMSE de prueba (puntos; menor es mejor)", 1, 2),
    (CLASIFICACION, "auc", False, "Clasificación: oficial o no oficial", "AUC-ROC de prueba (mayor es mejor)", 3, 0.1),
)
figura, ejes = plt.subplots(2, 1, figsize=(graficos.ANCHO_COLUMNA, 3.75), height_ratios=[9, 6])
for eje, (tarea, metrica, ascendente, titulo, etiqueta_x, decimales, paso) in zip(ejes, PANELES, strict=True):
    panel = metricas[metricas["tarea"] == tarea].sort_values(metrica, ascending=ascendente).reset_index(drop=True)
    posiciones = np.arange(len(panel))
    minimo, maximo = panel[metrica].min(), panel[metrica].max()
    margen = 0.12 * (maximo - minimo)
    eje.hlines(posiciones, minimo - margen, panel[metrica], color="#D9D9D9", linewidth=0.6, zorder=1)
    eje.scatter(panel[metrica], posiciones, s=16, color=panel["enfoque"].map(COLOR_ENFOQUE), zorder=2)
    for posicion, valor in zip(posiciones, panel[metrica], strict=True):
        eje.annotate(en_espanol(valor, decimales), (valor, posicion), xytext=(4, 0), textcoords="offset points", va="center", fontsize=6)
    eje.set_yticks(posiciones, panel["etiqueta"])
    eje.invert_yaxis()
    eje.set_xlim(minimo - margen, maximo + 2.6 * margen)
    eje.xaxis.set_major_locator(MultipleLocator(paso))
    eje.set_title(titulo, loc="left")
    eje.set_xlabel(etiqueta_x)
    eje.xaxis.set_major_formatter(graficos.formato_fijo(1 if tarea == CLASIFICACION else 0))
    eje.grid(axis="y", visible=False)
figura.legend(
    handles=[Line2D([], [], marker="o", linestyle="", color=COLOR_ENFOQUE[e], markersize=4, label=ETIQUETA_ENFOQUE[e]) for e in (SIN_IA, CON_IA)],
    loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.2, columnspacing=1.0, fontsize=6.5,
)
figura.tight_layout(h_pad=1.2, rect=(0, 0, 1, 0.95))
graficos.anotar_fuente(ejes[-1], f"{FUENTE} Prueba: {en_espanol(FILAS_PRUEBA)} estudiantes; cada modelo se evaluó una sola vez.")
ruta_con_sin_ia = figuras.guardar(figura, "con-sin-ia")
plt.close(figura)
print(f"Escrita {relativa(ruta_con_sin_ia)}")

# %%
mejores = {
    (tarea, enfoque): grupo.sort_values(metrica, ascending=ascendente).iloc[0]
    for tarea, metrica, ascendente in ((REGRESION, "rmse", True), (CLASIFICACION, "auc", False))
    for enfoque, grupo in metricas[metricas["tarea"] == tarea].groupby("enfoque")
}
resumen_ia = pd.DataFrame({
    f"{tarea}, {enfoque}": {"Mejor modelo": fila["etiqueta"], "R² o AUC": fila["r2"] if tarea == REGRESION else fila["auc"], "RMSE o F1": fila["rmse"] if tarea == REGRESION else fila["f1"]}
    for (tarea, enfoque), fila in mejores.items()
}).T
tabla(resumen_ia, 3)
mejora_rmse = mejores[REGRESION, SIN_IA]["rmse"] - mejores[REGRESION, CON_IA]["rmse"]
mejora_r2 = mejores[REGRESION, CON_IA]["r2"] - mejores[REGRESION, SIN_IA]["r2"]
mejora_auc = mejores[CLASIFICACION, CON_IA]["auc"] - mejores[CLASIFICACION, SIN_IA]["auc"]
mejora_f1 = mejores[CLASIFICACION, CON_IA]["f1"] - mejores[CLASIFICACION, SIN_IA]["f1"]
print(f"\nCon IA frente a sin IA, en prueba: RMSE {en_espanol(mejora_rmse, 2)} puntos menos; R² {en_espanol(mejora_r2, 3)} más; AUC {en_espanol(mejora_auc, 3)} más; F1 {en_espanol(mejora_f1, 3)} más")
comprobar("En las dos tareas, el mejor modelo con IA supera en prueba al mejor sin IA", mejora_rmse > 0 and mejora_auc > 0)
for (tarea, enfoque), fila in mejores.items():
    cifra(f"mejor{'Regresion' if tarea == REGRESION else 'Clasificacion'}{enfoque.replace(' ', '')}", fila["etiqueta"])
cifra("mejoraRmsePruebaConIa", mejora_rmse, 2)
cifra("mejoraRCuadradoPruebaConIa", mejora_r2, 3)
cifra("mejoraAucPruebaConIa", mejora_auc, 3)
cifra("mejoraFUnoPruebaConIa", mejora_f1, 3)
cifra("pctMejoraRmsePruebaConIa", 100 * mejora_rmse / mejores[REGRESION, SIN_IA]["rmse"], 1)
cifra("pctVarianzaSinExplicarPrueba", 100 * (1 - mejores[REGRESION, CON_IA]["r2"]), 0)
por_modelo = metricas.set_index("modelo")
comprobar("Ridge, Lasso y ElasticNet quedan a menos de 0,001 puntos de RMSE de la múltiple en prueba", (por_modelo.loc[["ridge", "lasso", "elasticNet"], "rmse"] - por_modelo.loc["regresionMultiple", "rmse"]).abs().max() < 0.001)
comprobar("En prueba, la SVM y el LDA quedan por debajo de la logística base en AUC", por_modelo.loc[["svmClasificacion", "ldaClasificacion"], "auc"].max() < por_modelo.loc["logisticaBase", "auc"]);

# %% [markdown]
# **Lectura.** En las dos tareas, el mejor modelo con IA supera al mejor sin IA, pero por poco. En la regresión, XGBoost baja el RMSE de prueba de 42,8 puntos (la múltiple) a 40,8, unos 2 puntos menos (el 4,6 %), y sube el R² de 0,319 a 0,379; Ridge, Lasso y ElasticNet quedan iguales a la múltiple, porque con 372.792 estudiantes de entrenamiento para 95 coeficientes la múltiple casi no sobreajusta y la penalización no tiene qué corregir. En la clasificación, la ventaja de XGBoost sobre la logística base es de 0,009 de AUC, y la SVM y el LDA quedan por debajo de la logística. Lo que agrega el aprendizaje automático son las interacciones y las relaciones no lineales que captan los ensambles, como la brecha entre colegios oficiales y no oficiales que crece con el estrato (H3), no un salto en la capacidad de anticipar el puntaje: con las variables del hogar, del colegio y del estudiante, cerca del 62 % de la varianza del puntaje global queda sin explicar con cualquiera de los dos enfoques.

# %% [markdown]
# ## 4. Paridad entre Python y R, consolidada
#
# En cada fase, un script de R replicó con sus propias funciones lo que calculó Python, y el cuaderno de la fase comparó valor por valor con una tolerancia de 10⁻⁶: relativa en los estadísticos y absoluta (en puntos) en los coeficientes de la regresión de la fase 2. La tabla consolidada lee las salidas de R (`resultados/r/*.json`), de las que toma la versión de R y de sus paquetes y cuántos valores exporta cada script, y las cifras de paridad que dejó cada fase, de las que toma cuántos valores se compararon y la mayor diferencia. R exporta más de lo que se compara (errores estándar, intervalos y valores p que Python no siempre calcula igual), así que los valores comparados nunca superan a los exportados. La tabla queda en `resultados/paridad-consolidada.tex`, lista para el anexo de la A8.

# %%
RUTA_R = RUTA_RESULTADOS / "r"
salidas_r = {nombre: json.loads((RUTA_R / f"{nombre}.json").read_text(encoding="utf-8")) for nombre in ("fase-1", "fase-2", "fase-3-hipotesis", "fase-3-multivariante")}


def valores_numericos(objeto: object) -> int:
    """Cuenta los números de una salida de R (sin contar los booleanos)."""
    if isinstance(objeto, dict):
        return sum(valores_numericos(valor) for valor in objeto.values())
    if isinstance(objeto, list):
        return sum(valores_numericos(valor) for valor in objeto)
    return int(isinstance(objeto, (int, float)) and not isinstance(objeto, bool))


def numero_de_cifra(clave: str) -> float:
    """Valor numérico de una cifra que la fase guardó como texto («1,2e-10»)."""
    valor = cifras.valor(clave)
    return float(valor.replace(",", ".")) if isinstance(valor, str) else float(valor)


def cientifico(valor: float) -> str:
    """Notación científica para LaTeX con coma decimal: 3,4e-08 da $3{,}4\\times10^{-8}$."""
    mantisa, exponente = f"{valor:.1e}".split("e")
    return f"${mantisa.replace('.', '{,}')}\\times10^{{{int(exponente)}}}$"


BLOQUES_PARIDAD = [
    {
        "salida": "fase-1", "buzon": "A2",
        "replica": r"Descriptivos de los seis puntajes, frecuencias, correlaciones de Pearson y Spearman y $\chi^2$ de H4",
        "valores": cifras.valor("paridadValoresFaseUno"), "relativa": 10.0 ** cifras.valor("paridadCota"), "cota": True, "absoluta": None,
    },
    {
        "salida": "fase-2", "buzon": "A3 y A4",
        "replica": r"Descriptivos y frecuencias antes y después de preparar; \texttt{lm} simple y múltiple con \texttt{broom}",
        "valores": cifras.valor("paridadValoresFaseDos"), "relativa": max(numero_de_cifra("paridadMaxDescriptivos"), numero_de_cifra("paridadMaxRCuadrado")),
        "cota": False, "absoluta": numero_de_cifra("paridadMaxCoeficientes"),
    },
    {
        "salida": "fase-3-hipotesis", "buzon": "A5",
        "replica": r"\texttt{t.test}, \texttt{oneway.test}, \texttt{aov} con \texttt{TukeyHSD}, \texttt{car::Anova} tipo II, \texttt{kruskal.test} y \texttt{chisq.test} de H1 a H4",
        "valores": cifras.valor("paridadValoresHipotesis"), "relativa": numero_de_cifra("paridadMaxRelativaHipotesis"), "cota": False, "absoluta": None,
    },
    {
        "salida": "fase-3-multivariante", "buzon": "A6",
        "replica": r"\texttt{prcomp}, medias por clúster de K-Means y de Ward y \texttt{chisq.test} de los tres cruces",
        "valores": cifras.valor("paridadValoresMultivariante"), "relativa": numero_de_cifra("paridadMaxRelativaMultivariante"), "cota": False, "absoluta": None,
    },
]
paridad = pd.DataFrame(BLOQUES_PARIDAD).set_index("salida").astype({"valores": int})
paridad["exportados"] = [sum(valores_numericos(valor) for clave, valor in salidas_r[salida].items() if clave != "medicion") for salida in paridad.index]
paridad["version_r"] = [salidas_r[salida]["version_r"] for salida in paridad.index]
print(paridad[["buzon", "valores", "exportados", "relativa", "absoluta", "version_r"]].to_string(formatters={"relativa": "{:.1e}".format, "absoluta": lambda x: "" if pd.isna(x) else f"{x:.1e}"}))
TOLERANCIA_PARIDAD = 1e-6
comprobar("Paridad con R: las cuatro salidas de R usan la misma versión de R", paridad["version_r"].nunique() == 1)
comprobar("Paridad con R: en cada salida, los valores comparados no superan a los que exporta R", (paridad["valores"] <= paridad["exportados"]).all())
comprobar(
    f"Paridad con R: todas las diferencias están por debajo de la tolerancia de {TOLERANCIA_PARIDAD:g} de las fases",
    (paridad["relativa"] < TOLERANCIA_PARIDAD).all() and paridad["absoluta"].dropna().lt(TOLERANCIA_PARIDAD).all(),
)


def celda_diferencia(fila: pd.Series) -> str:
    texto = f"$<10^{{{round(np.log10(fila['relativa']))}}}$" if fila["cota"] else cientifico(fila["relativa"])
    return texto if pd.isna(fila["absoluta"]) else f"{texto}; {cientifico(fila['absoluta'])} abs."


version_r = paridad["version_r"].iloc[0]
lineas = [
    "% Generado por notebooks/fase-4-sintesis.ipynb (sección 4): no editar a mano.",
    f"% R {version_r}; ggplot2 {salidas_r['fase-1']['version_ggplot2']}; car {salidas_r['fase-3-hipotesis']['version_car']}.",
    r"\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{4.3cm}rl@{}}",
    r"\toprule",
    r"Buzón & Qué replica R & Valores & Mayor diferencia \\",
    r"\midrule",
    *[f"{fila['buzon']} & {fila['replica']} & {en_espanol(fila['valores'])} & {celda_diferencia(fila)} \\\\" for _, fila in paridad.iterrows()],
    r"\midrule",
    f"Total & R {version_r} & {en_espanol(paridad['valores'].sum())} & {cientifico(paridad['relativa'].max())} \\\\",
    r"\bottomrule",
    r"\end{tabular}",
]
ruta_paridad = RUTA_RESULTADOS / "paridad-consolidada.tex"
ruta_paridad.write_text("\n".join(lineas) + "\n", encoding="utf-8")
print(f"\nEscrita {relativa(ruta_paridad)}")
print(ruta_paridad.read_text(encoding="utf-8"))
comprobar("La tabla de paridad tiene una fila por salida de R y el total", ruta_paridad.read_text(encoding="utf-8").count("\\\\") == len(paridad) + 2)
cifra("paridadValoresConsolidada", int(paridad["valores"].sum()))
cifra("paridadValoresExportadosR", int(paridad["exportados"].sum()))
cifra("paridadSalidasR", len(paridad))
cifra("paridadMaxRelativaConsolidada", f"{paridad['relativa'].max():.1e}".replace(".", ","))
cifra("paridadMaxAbsolutaConsolidada", f"{paridad['absoluta'].max():.1e}".replace(".", ","));

# %% [markdown]
# **Lectura.** Las cuatro salidas de R corrieron con la misma versión de R y reprodujeron los 848 valores que compararon las fases, con una diferencia relativa de 3,4 × 10⁻⁸ o menor; la mayor está en las pruebas de hipótesis de la A5 y queda casi 30 veces por debajo de la tolerancia de 10⁻⁶. En los coeficientes de la regresión de la fase 2, la diferencia absoluta es de 1,2 × 10⁻¹⁰ puntos. Python y R dan los mismos resultados hasta el redondeo de la máquina, así que cualquiera de los dos sirve para replicar el análisis.

# %% [markdown]
# ## 5. Agregado por jornada
#
# La pregunta principal nombra la jornada entre las características del colegio, pero ningún agregado del dashboard la traía. Se busca el cruce más fino de la jornada con el estrato, la naturaleza y la zona que cumpla la regla de los agregados: un cruce completo, con sus marginales «Todos», y 10 o más estudiantes en todas sus celdas. Los candidatos se prueban del más fino al más grueso (más dimensiones primero y, entre los del mismo número, más celdas y después la menor celda más grande), y se publica el primero que cumple. Las medidas son las de los agregados de la fase 2: estudiantes, media, mediana, cuartiles y desviación estándar del puntaje global.

# %%
TODOS, SIN_DATO = "Todos", "Sin dato"
MEDIDAS_PUNTAJE = {
    "n": "size",
    "media": "mean",
    "mediana": "median",
    "q1": lambda serie: serie.quantile(0.25),
    "q3": lambda serie: serie.quantile(0.75),
    "desviacion": "std",
}


def cubo(marco: pd.DataFrame, dimensiones: list[str], medidas: dict) -> pd.DataFrame:
    """Cruce completo de las dimensiones más sus marginales, con «Todos» en la dimensión que no se agrupa."""
    partes = []
    for tamano in range(len(dimensiones), -1, -1):
        for agrupadas in combinations(dimensiones, tamano):
            vista = marco.assign(**{d: TODOS for d in dimensiones if d not in agrupadas})
            partes.append(vista.groupby(dimensiones, observed=True)["puntaje"].agg(**medidas).reset_index())
    return pd.concat(partes, ignore_index=True)


def es_cruce_completo(marco: pd.DataFrame, dimensiones: list[str]) -> bool:
    return len(marco) == int(np.prod([marco[d].nunique() for d in dimensiones])) and not marco.duplicated(dimensiones).any()


limpio, _ = limpieza.limpiar(pd.read_parquet(datos.RECORTE_MODELADO))
comprobar("El análisis tiene las filas de la fase 2", len(limpio) == cifras.valor("filasAnalisis"))
analisis = pd.DataFrame({
    "jornada": limpio["COLE_JORNADA"],
    "estrato": limpio[v.ESTRATO].fillna(SIN_DATO),
    "naturaleza": limpio["COLE_NATURALEZA"],
    "zona": limpio["COLE_AREA_UBICACION"],
    "puntaje": limpio[v.OBJETIVO_REGRESION],
})
comprobar("Las dimensiones del agregado no tienen faltantes, salvo el estrato, que va como «Sin dato»", not analisis.isna().any().any())

candidatos = []
for tamano in range(3, -1, -1):
    for otras in combinations(["estrato", "naturaleza", "zona"], tamano):
        dimensiones = ["jornada", *otras]
        conteos = cubo(analisis, dimensiones, {"n": "size"})
        candidatos.append({
            "dimensiones": " × ".join(dimensiones), "lista": dimensiones, "numero": len(dimensiones), "celdas": len(conteos),
            "menor": int(conteos["n"].min()), "cumple": es_cruce_completo(conteos, dimensiones) and conteos["n"].min() >= MINIMO_POR_CELDA,
        })
candidatos = pd.DataFrame(candidatos).sort_values(["numero", "celdas", "menor"], ascending=False).reset_index(drop=True)
tabla(candidatos.set_index("dimensiones")[["celdas", "menor", "cumple"]])
elegido = candidatos[candidatos["cumple"]].iloc[0]
dimensiones_jornada = elegido["lista"]
jornada = cubo(analisis, dimensiones_jornada, MEDIDAS_PUNTAJE)
ruta_jornada = RUTA_AGREGADOS / f"{'-'.join(dimensiones_jornada)}-20224.csv"
comprobar(
    f"{ruta_jornada.name}: cruce completo, con {MINIMO_POR_CELDA} o más estudiantes en cada una de sus {len(jornada)} celdas",
    es_cruce_completo(jornada, dimensiones_jornada) and jornada["n"].min() >= MINIMO_POR_CELDA,
)
comprobar("El marginal «Todos» del agregado de jornada suma los estudiantes del análisis", jornada.loc[jornada[dimensiones_jornada].eq(TODOS).all(axis=1), "n"].item() == len(limpio))
jornada.round(2).to_csv(ruta_jornada, index=False)
print(f"\nElegido: {elegido['dimensiones']}. Escrito {relativa(ruta_jornada)}: {len(jornada)} filas, la menor celda con {en_espanol(jornada['n'].min())} estudiantes")
por_jornada = jornada[jornada[dimensiones_jornada[1:]].eq(TODOS).all(axis=1)].set_index("jornada").drop(index=TODOS).sort_values("media", ascending=False)
tabla(por_jornada[["n", "media", "mediana", "desviacion"]], 1)
cifra("archivoAgregadoJornada", ruta_jornada.name)
cifra("dimensionesAgregadoJornada", len(dimensiones_jornada))
cifra("filasAgregadoJornada", len(jornada))
cifra("celdaMinimaJornada", jornada["n"].min())
cifra("candidatosJornadaDescartados", int((~candidatos["cumple"]).sum()))
cifra("jornadaMediaMasAlta", por_jornada.index[0])
cifra("jornadaMediaMasBaja", por_jornada.index[-1])
cifra("brechaMediaJornadas", por_jornada["media"].iloc[0] - por_jornada["media"].iloc[-1], 1)
comprobar("La jornada completa tiene la media más alta, y la nocturna y la sabatina las dos más bajas", por_jornada.index[0] == "COMPLETA" and set(por_jornada.index[-2:]) == {"NOCHE", "SABATINA"});

# %% [markdown]
# **Lectura.** El cruce más fino que cumple la regla es jornada por estrato y naturaleza del colegio: 189 filas con los marginales, la menor con 32 estudiantes. Con la zona como cuarta dimensión, una celda queda con un solo estudiante y el cruce no se publica; entre los dos cruces de tres dimensiones que cumplen, el de la naturaleza tiene la menor celda más grande (32 frente a 14 del de la zona). Por jornada, la completa tiene la media más alta (280,8 puntos) y la nocturna y la sabatina las más bajas (210,6 y 212,7), unos 70 puntos por debajo. Es una diferencia descriptiva, que mezcla la jornada con el estrato, la naturaleza y la edad de quienes estudian en cada una; el agregado deja que el dashboard la abra por estrato y naturaleza.

# %% [markdown]
# ## 6. Modelo de predicción del tablero
#
# La pestaña Predicción del dashboard estima el puntaje global esperado de un perfil de hogar, colegio y estudiante. La regla para elegir el modelo se fijó antes de ver resultados: el mejor de regresión de la fase 3 por RMSE de validación entre los que no guardan observaciones y pesan menos de 50 MB; se exporta para predecir en la CPU, porque Streamlit Community Cloud no tiene GPU, y si es de árboles, toda hoja tiene que cubrir 10 o más estudiantes, la misma regla de las celdas de los agregados. Si el modelo de la fase 3 no la cumple, se reajusta con ese mínimo y se declara la diferencia en validación. El modelo del tablero nunca se evalúa en prueba: la prueba ya se usó una vez por modelo, y las tarjetas del tablero muestran las métricas de prueba de la fase 3.
#
# ### 6.1. Elección
#
# Los seis modelos de regresión de la fase 3 se ordenan por su RMSE de validación, el que registró la fase. Ninguno guarda observaciones (no hay vecinos más cercanos ni vectores de soporte entre ellos), así que el primero es el elegido si pesa menos de 50 MB.

# %%
CANDIDATOS_TABLERO = {"xgboostRegresion": "Xgboost", "mlpRegresion": "Mlp", "randomForestRegresion": "RandomForest", "lasso": "Lasso", "elasticNet": "ElasticNet", "ridge": "Ridge"}
eleccion = pd.DataFrame(
    {"Modelo": MODELOS.loc[modelo, "etiqueta"], "RMSE de validación": cifras.valor(f"rmseValidacion{sufijo}")}
    for modelo, sufijo in CANDIDATOS_TABLERO.items()
).set_axis(list(CANDIDATOS_TABLERO)).sort_values("RMSE de validación")
tabla(eleccion, 3)
comprobar("El mejor modelo de regresión de la fase 3 por RMSE de validación es XGBoost", eleccion.index[0] == "xgboostRegresion")

# %% [markdown]
# ### 6.2. Reajuste en la CPU con hojas de 10 o más estudiantes
#
# XGBoost con pérdida cuadrática pesa cada estudiante con 1 en la segunda derivada de la pérdida, así que el `Cover` de una hoja es el número de estudiantes de entrenamiento que caen en ella, y el hiperparámetro `min_child_weight` es el mínimo de estudiantes que puede tener una hoja nueva. La búsqueda de la fase 3 eligió un valor menor que 10, así que el modelo se reajusta con los mismos hiperparámetros (los de `cifras-fase-3.json`), `min_child_weight` en 10 y `device="cpu"`, sobre el mismo entrenamiento. El ajuste se guarda en la caché (`ajuste.en_cache`), así que una nueva ejecución lo reusa. Se reconstruyen la partición y los conjuntos de la fase 2 y se comprueba su huella.

# %%
# Unos pocos niveles raros de validación no aparecen en el entrenamiento; el one-hot de la fase 2
# (`handle_unknown="ignore"`) los codifica en ceros y avisa cada vez, sin cambiar el resultado.
warnings.filterwarnings("ignore", message="Found unknown categories in columns", category=UserWarning)
conjuntos = particion.particionar(limpio[v.OBJETIVO_CLASIFICACION])
comprobar("La partición tiene la huella de la fase 2", particion.huella(conjuntos) == cifras.valor("huellaParticion"))
es_ent, es_val = conjuntos.eq(particion.ENTRENAMIENTO), conjuntos.eq(particion.VALIDACION)
X_ent, X_val = (limpio.loc[m, list(v.ENTRADA_REGRESION)] for m in (es_ent, es_val))
y_ent, y_val = (limpio.loc[m, v.OBJETIVO_REGRESION] for m in (es_ent, es_val))
comprobar("Entrenamiento y validación tienen los tamaños de la fase 2", len(X_ent) == cifras.valor("filasEntrenamiento") and len(X_val) == cifras.valor("filasValidacion"))

PARAMETROS_XGBOOST = {"n_estimators": "NEstimators", "learning_rate": "LearningRate", "max_depth": "MaxDepth", "min_child_weight": "MinChildWeight", "subsample": "Subsample", "colsample_bytree": "ColsampleBytree", "reg_lambda": "RegLambda"}
ENTEROS = {"n_estimators", "max_depth"}
de_la_fase_3 = {p: (int(cifras.valor(f"mejorXgboost{s}")) if p in ENTEROS else cifras.valor(f"mejorXgboost{s}")) for p, s in PARAMETROS_XGBOOST.items()}
del_tablero = de_la_fase_3 | {"min_child_weight": max(float(tablero.MINIMO_POR_HOJA), de_la_fase_3["min_child_weight"])}
print(pd.DataFrame({"Fase 3": de_la_fase_3, "Tablero": del_tablero}).to_string())
comprobar("La fase 3 no garantizaba hojas de 10 o más estudiantes, así que el modelo se reajusta", de_la_fase_3["min_child_weight"] < tablero.MINIMO_POR_HOJA)

PREDICTORES = v.PREDICTORES_REGRESION[modelos.ESPECIFICACION]
ajustado = ajuste.en_cache(ajuste.ajustado)
ajuste_tablero = ajustado(modelos.pipeline(PREDICTORES, XGBRegressor(**del_tablero, device="cpu", random_state=ajuste.SEMILLA)), X_ent, y_ent)
modelo_tablero = ajuste_tablero.modelo
prediccion_val = modelo_tablero.predict(X_val)
metricas_tablero = modelos.metricas_regresion(y_val, prediccion_val)
rmse_fase_3 = cifras.valor("rmseValidacionXgboost")
diferencia_pct = 100 * (metricas_tablero["rmse"] - rmse_fase_3) / rmse_fase_3
cobertura_hojas = tablero.cobertura_minima_de_hojas(modelo_tablero.named_steps["modelo"])
print(f"\nAjuste en la CPU: {en_espanol(ajuste_tablero.segundos, 1)} s (primera ejecución, guardada en la caché)")
print(f"Validación: RMSE {en_espanol(metricas_tablero['rmse'], 3)} (fase 3: {en_espanol(rmse_fase_3, 3)}; diferencia {en_espanol(diferencia_pct, 2)} %), R² {en_espanol(metricas_tablero['r2'], 3)}, MAE {en_espanol(metricas_tablero['mae'], 2)}")
print(f"Menor cobertura de una hoja: {en_espanol(cobertura_hojas)} estudiantes")
comprobar(f"Toda hoja del modelo cubre {tablero.MINIMO_POR_HOJA} o más estudiantes de entrenamiento", cobertura_hojas >= tablero.MINIMO_POR_HOJA)
comprobar("El RMSE de validación del reajuste difiere menos del 1 % del de la fase 3", abs(diferencia_pct) < 1)
cifra("rmseValidacionTablero", metricas_tablero["rmse"], 2)
cifra("rCuadradoValidacionTablero", metricas_tablero["r2"], 3)
cifra("maeValidacionTablero", metricas_tablero["mae"], 2)
cifra("diferenciaRmseTableroPct", diferencia_pct, 2)
cifra("minChildWeightFaseTres", de_la_fase_3["min_child_weight"], 2)
cifra("minChildWeightTablero", del_tablero["min_child_weight"])
cifra("coberturaMinimaHojaTablero", cobertura_hojas)
cifra("medidaAjusteTableroSegundos", ajuste_tablero.segundos, 0);

# %% [markdown]
# **Lectura.** Exigir 10 estudiantes por hoja no le cuesta precisión al modelo: el RMSE de validación del reajuste queda en 40,83 puntos, frente a 40,84 del XGBoost de la fase 3, y el R² en 0,380. El reajuste en la CPU tarda menos de un minuto, así que el modelo del tablero se reconstruye en cualquier máquina sin GPU.

# %% [markdown]
# ### 6.3. Margen de la estimación
#
# La estimación es el puntaje esperado (el promedio) de los estudiantes con ese perfil, no el de una persona. El margen dice cuánto se apartan de esa estimación los estudiantes reales: es el residuo absoluto de validación en la posición ⌈(n + 1) × 0,9⌉, el cuantil 0,9 con la corrección de muestra finita de la predicción conforme por partición. Se lee así: en validación, 9 de cada 10 estudiantes quedaron a menos de ±X puntos de la estimación de su perfil. Como la validación también sirvió para elegir el modelo, la cobertura del 90 % es aproximada y no una garantía formal, y es una cobertura promedio: en algunos grupos el margen cubre a más estudiantes que en otros, como muestra la tabla.

# %%
residuos = np.abs(y_val.to_numpy() - prediccion_val)
margen = tablero.margen_conformal(residuos)
cobertura = float(np.mean(residuos <= margen))
print(f"Margen al {en_espanol(100 * tablero.NIVEL_MARGEN)} %: ±{en_espanol(margen, 1)} puntos; cobertura en validación: {en_espanol(100 * cobertura, 1)} % de {en_espanol(len(residuos))} estudiantes")
comprobar("El margen cubre al menos al 90 % de los estudiantes de validación", cobertura >= tablero.NIVEL_MARGEN)
dentro = pd.DataFrame({
    "Naturaleza": limpio.loc[es_val, "COLE_NATURALEZA"].to_numpy(),
    "Zona": limpio.loc[es_val, "COLE_AREA_UBICACION"].to_numpy(),
    "Estrato": limpio.loc[es_val, v.ESTRATO].fillna(SIN_DATO).to_numpy(),
    "dentro": residuos <= margen,
})
cobertura_por_grupo = pd.concat(
    {dimension: dentro.groupby(dimension)["dentro"].agg(n="size", cobertura=lambda serie: 100 * serie.mean()) for dimension in ("Naturaleza", "Zona", "Estrato")},
    names=["Dimensión", "Grupo"],
)
tabla(cobertura_por_grupo, 1)
comprobar("Cada grupo de la tabla de cobertura tiene 10 o más estudiantes", cobertura_por_grupo["n"].min() >= MINIMO_POR_CELDA)
grupo_menor, grupo_mayor = cobertura_por_grupo["cobertura"].idxmin(), cobertura_por_grupo["cobertura"].idxmax()
cifra("margenNoventaTablero", margen, 1)
cifra("coberturaMargenTablero", 100 * cobertura, 1)
cifra("coberturaMenorGrupoTablero", cobertura_por_grupo["cobertura"].min(), 1)
cifra("grupoCoberturaMenorTablero", grupo_menor[1])
cifra("coberturaMayorGrupoTablero", cobertura_por_grupo["cobertura"].max(), 1)
cifra("grupoCoberturaMayorTablero", grupo_mayor[1]);

# %% [markdown]
# **Lectura.** El margen es de ±66,8 puntos. Es amplio porque las variables del formulario explican cerca del 38 % de la varianza del puntaje global y el resto depende de lo que estos datos no miden: la estimación sirve para comparar perfiles, no para anticipar el puntaje de una persona. La cobertura es del 90,0 % en toda la validación y va del 88,8 % en los colegios no oficiales al 93,2 % en la zona rural: el margen queda un poco corto donde los puntajes se dispersan más y un poco holgado donde se dispersan menos.

# %% [markdown]
# ### 6.4. Exportación y ficha
#
# El modelo se exporta con `joblib` a `dashboard/modelo/regresion.joblib`. Es un único `Pipeline` que recibe las 19 columnas de entrada tal como las llena el formulario (con los cuartos y las personas ya armonizados a número) y adentro imputa, calcula las derivadas, trata los atípicos, codifica y predice: la app no repite nada de la preparación. La ficha `dashboard/modelo/ficha.json` describe el formulario y el modelo: para cada columna, su grupo (hogar, colegio o estudiante), sus niveles o valores admitidos (solo los que tienen 10 o más estudiantes de entrenamiento) y si admite «Sin dato», que solo se ofrece donde el `Pipeline` imputa; además, los hiperparámetros, las métricas de validación propias y de la fase 3, el margen y las versiones exactas de las bibliotecas con que se entrenó, que la app debe instalar iguales. Antes de escribirlo se comprueba que no guarde observaciones: ninguna matriz de dos dimensiones con 1.000 filas o más y ninguna clase de vecinos o de vectores de soporte.

# %%
RUTA_MODELO = RAIZ / "dashboard" / "modelo"
RUTA_MODELO.mkdir(parents=True, exist_ok=True)
ruta_modelo, ruta_ficha = RUTA_MODELO / "regresion.joblib", RUTA_MODELO / "ficha.json"
comprobar("El modelo predice en la CPU", modelo_tablero.named_steps["modelo"].get_params()["device"] == "cpu")
comprobar("El modelo no guarda matrices de 1.000 filas o más ni clases de vecinos o de vectores de soporte", not tablero.matrices_grandes(modelo_tablero) and not tablero.clases_con_observaciones(modelo_tablero))
modulos_propios = sorted({type(objeto).__module__ for _, objeto in tablero.objetos(modelo_tablero) if type(objeto).__module__.startswith("saber11")})
print(f"Módulos de saber11 que necesita la app para cargar el modelo: {modulos_propios}")
comprobar("Para deserializar el modelo basta saber11.variables (que no importa PyTorch)", modulos_propios == ["saber11.variables"])
joblib.dump(modelo_tablero, ruta_modelo, compress=3)
peso_mb = ruta_modelo.stat().st_size / 1e6
recargado = joblib.load(ruta_modelo)
comprobar(f"El modelo pesa menos de {tablero.PESO_MAXIMO_MB} MB ({en_espanol(peso_mb, 1)} MB)", peso_mb < tablero.PESO_MAXIMO_MB)
comprobar("El modelo recargado predice lo mismo que el ajustado en validación", np.array_equal(recargado.predict(X_val), prediccion_val))

imputador = modelo_tablero.named_steps["imputacion"]
entradas_formulario = tablero.entradas(X_ent, imputador.con_faltantes_)
ficha = {
    "modelo": "XGBoost de la fase 3, reajustado en la CPU con hojas de 10 o más estudiantes",
    "archivo": ruta_modelo.name,
    "objetivo": v.OBJETIVO_REGRESION,
    "periodo": "2022-2",
    "estimacion": "Puntaje global esperado (promedio) de los estudiantes con el perfil; no es el resultado de una persona.",
    "entrada": entradas_formulario,
    "hiperparametros": del_tablero,
    "hiperparametros_fase_3": de_la_fase_3,
    "metricas_validacion": {**metricas_tablero, "rmse_fase_3": rmse_fase_3, "diferencia_rmse_pct": diferencia_pct, "filas": len(X_val)},
    "filas_entrenamiento": len(X_ent),
    "margen": {
        "nivel": tablero.NIVEL_MARGEN, "puntos": margen, "cobertura_validacion": cobertura,
        "lectura": f"En validación, 9 de cada 10 estudiantes quedaron a menos de ±{en_espanol(margen, 1)} puntos de la estimación de su perfil.",
    },
    "minimo_por_hoja": tablero.MINIMO_POR_HOJA,
    "cobertura_minima_hoja": cobertura_hojas,
    "peso_mb": peso_mb,
    "python": platform.python_version(),
    "versiones": tablero.versiones(),
    "fuente": "ICFES, Resultados únicos Saber 11, 2022-2 (datos.gov.co), licencia CC BY-SA 4.0.",
}
ruta_ficha.write_text(json.dumps(ficha, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
resumen_formulario = pd.DataFrame(
    {"Grupo": e["grupo"], "Tipo": e["tipo"], "Opciones": len(e.get("niveles", e.get("valores", []))), "Sin dato": "sí" if e["sin_dato"] else "no"}
    for e in entradas_formulario
).set_axis([e["etiqueta"] for e in entradas_formulario])
print(resumen_formulario.to_string())
fuera = {e["columna"]: sorted(set(X_ent[e["columna"]].dropna().astype(str)) - set(e["niveles"])) for e in entradas_formulario if e["tipo"] == "categorica"}
fuera = {columna: niveles for columna, niveles in fuera.items() if niveles}
print(f"Niveles vistos en entrenamiento con menos de {tablero.MINIMO_POR_NIVEL} estudiantes, fuera del formulario: {fuera or 'ninguno'}")

ejemplo = tablero.perfil(ficha, tablero.PERFIL_DE_EJEMPLO)
sin_datos = tablero.perfil(ficha, tablero.PERFIL_DE_EJEMPLO | {e["columna"]: None for e in entradas_formulario if e["sin_dato"]})
with warnings.catch_warnings():
    # Un nivel desconocido para el one-hot sería un «Sin dato» que no se imputó.
    warnings.filterwarnings("error", message="Found unknown categories")
    estimaciones = recargado.predict(pd.concat([ejemplo, sin_datos], ignore_index=True))
comprobar("El «Sin dato» del formulario llega imputado al modelo", not recargado.named_steps["imputacion"].transform(sin_datos).isna().any().any())
print(f"\nPerfil de ejemplo: {en_espanol(estimaciones[0], 1)} ± {en_espanol(margen, 1)} puntos; con «Sin dato» donde se admite: {en_espanol(estimaciones[1], 1)}")
comprobar("Las dos estimaciones del perfil de ejemplo están entre 0 y 500", bool(np.all((estimaciones >= 0) & (estimaciones <= 500))))
cifra("pesoModeloTablero", peso_mb, 1)
cifra("entradasFormularioTablero", len(entradas_formulario))
cifra("entradasSinDatoTablero", sum(e["sin_dato"] for e in entradas_formulario))
cifra("estimacionEjemploTablero", estimaciones[0], 1)
print(f"\nEscritos {relativa(ruta_modelo)} y {relativa(ruta_ficha)}")

# %% [markdown]
# **Lectura.** El modelo pesa 3,5 MB y solo necesita, además de scikit-learn y XGBoost, el módulo `saber11.variables` para cargarse, así que la app no instala PyTorch. Con «Sin dato» en todas las columnas que lo admiten, la estimación del perfil de ejemplo baja de 261,1 a 213,8 puntos: el `Pipeline` imputa con la moda, pero también marca cada faltante con su indicador, y en estos datos quien deja respuestas en blanco suele tener puntajes más bajos, como mostró la fase 2 al estudiar el mecanismo de los faltantes. La app tiene que decirlo junto al formulario, para que «Sin dato» no se lea como un valor neutro.

# %% [markdown]
# ### 6.5. Memoria del dashboard
#
# Streamlit Community Cloud detiene una app que pasa su tope de memoria. Como Streamlit ejecuta el guion entero en cada interacción, con todas las pestañas a la vez, la medida corre la app en un proceso aparte con `AppTest`, envía el formulario de Predicción con otra audiencia y otro filtro, e informa el pico de memoria residente del proceso (`tablero.memoria_maxima_mb`), sin GPU visible, como en Community Cloud. Incluye los agregados, las figuras y el modelo, y deja fuera el servidor web de Streamlit. El pico se lee de `VmHWM` y no de `ru_maxrss`, que en Linux arrastra la memoria de este núcleo, del que el proceso nace.

# %%
LIMITE_MEMORIA_MB = 2700  # tope de memoria de una app en Streamlit Community Cloud
memoria_mb = tablero.memoria_maxima_mb(RAIZ / "dashboard" / "app.py")
print(f"Pico de memoria del proceso que recorre el dashboard: {en_espanol(memoria_mb, 0)} MB")
comprobar(f"El dashboard cabe en el tope de memoria de Community Cloud ({en_espanol(LIMITE_MEMORIA_MB)} MB)", memoria_mb < LIMITE_MEMORIA_MB)
cifra("medidaMemoriaTablero", memoria_mb, 0)
cifra("limiteMemoriaTablero", LIMITE_MEMORIA_MB)

# %% [markdown]
# ## 7. Tamaño de efecto de las hipótesis frente a SHAP
#
# Las pruebas de hipótesis de la fase 3 midieron cuánto de la varianza del puntaje global explica cada factor por sí solo (η² en H1 y H2) o dentro del modelo de dos factores de H3 (η² parcial del estrato y de la naturaleza). SHAP mide otra cosa: cuánto mueve cada variable la estimación del modelo del tablero cuando las demás ya están en él, como la media del valor absoluto de su contribución en puntos. La comparación se hace sobre las cuatro variables de H1 a H3, con SHAP sobre los mismos 10.000 estudiantes de validación de la fase 3 (estratificados por naturaleza del colegio). El η² de H1 se calcula aquí con las medias, las desviaciones y los tamaños de sus dos grupos, que la fase 3 dejó en sus cifras; la fase reportó la d de Cohen.

# %%
FILAS_INTERPRETACION = int(cifras.valor("filasInterpretacion"))
elegidas, _ = train_test_split(np.arange(len(X_val)), train_size=FILAS_INTERPRETACION, stratify=limpio.loc[es_val, v.OBJETIVO_CLASIFICACION], random_state=ajuste.SEMILLA)
X_interpretacion = X_val.iloc[np.sort(elegidas)]
contribuciones = interpretabilidad.contribuciones_shap(modelo_tablero.named_steps["modelo"], modelo_tablero[:-1].transform(X_interpretacion))
importancia_tablero = interpretabilidad.importancia_media(interpretabilidad.por_variable(contribuciones, PREDICTORES))
puesto_shap = importancia_tablero.rank(ascending=False, method="min").astype(int)
print(f"Las tres de mayor SHAP en el modelo del tablero: {[tablero.ETIQUETAS[c] for c in importancia_tablero.index[:3]]}")
tres_fase_3 = [cifras.valor(f"variableShap{p}") for p in ("Primera", "Segunda", "Tercera")]
print(f"En la fase 3: {tres_fase_3}")
comprobar("Las tres variables de mayor SHAP son las mismas y en el mismo orden que en la fase 3", all(tablero.ETIQUETAS[c].startswith(e) for c, e in zip(importancia_tablero.index[:3], tres_fase_3, strict=True)))


def eta_cuadrado_de_grupos(n: np.ndarray, media: np.ndarray, desviacion: np.ndarray) -> float:
    """η² de un factor a partir de los tamaños, las medias y las desviaciones de sus grupos."""
    media_global = np.sum(n * media) / np.sum(n)
    entre = np.sum(n * (media - media_global) ** 2)
    dentro_grupos = np.sum((n - 1) * desviacion**2)
    return float(entre / (entre + dentro_grupos))


grupos_h1 = {grupo: [cifras.valor(f"{medida}HUno{grupo}") for medida in ("n", "media", "desviacion")] for grupo in ("Si", "No")}
eta_h1 = eta_cuadrado_de_grupos(*(np.array(valores) for valores in zip(*grupos_h1.values(), strict=True)))
EFECTOS = pd.DataFrame(
    [
        ("FAMI_TIENEINTERNET", "Internet (H1)", "η²", eta_h1),
        (v.EDUCACION_MADRE, "Educación de la madre (H2)", "η²", cifras.valor("efectoHDos")),
        ("COLE_NATURALEZA", "Naturaleza del colegio (H3)", "η² parcial", cifras.valor("efectoHTres")),
        (v.ESTRATO, "Estrato (H3)", "η² parcial", cifras.valor("efectoHTresEstrato")),
    ],
    columns=["columna", "etiqueta", "medida", "eta"],
).set_index("columna")
EFECTOS["shap"] = importancia_tablero.loc[EFECTOS.index]
EFECTOS["puesto_shap"] = puesto_shap.loc[EFECTOS.index]
EFECTOS = EFECTOS.sort_values("eta", ascending=False)
tabla(EFECTOS.assign(eta=100 * EFECTOS["eta"]).set_index("etiqueta")[["medida", "eta", "shap", "puesto_shap"]], 2)

figura, (eje_eta, eje_shap) = plt.subplots(1, 2, figsize=(graficos.ANCHO_COLUMNA, 1.55), sharey=True)
posiciones = np.arange(len(EFECTOS))
for eje, columna, titulo, escala, decimales in ((eje_eta, "eta", "η² o η² parcial (%)", 100, 1), (eje_shap, "shap", "SHAP medio (puntos)", 1, 1)):
    valores = escala * EFECTOS[columna].to_numpy()
    eje.barh(posiciones, valores, height=0.6, color=graficos.COLOR_PRINCIPAL)
    for posicion, valor in zip(posiciones, valores, strict=True):
        eje.annotate(en_espanol(valor, decimales), (valor, posicion), xytext=(2, 0), textcoords="offset points", va="center", fontsize=6)
    eje.set_xlim(0, 1.4 * valores.max())
    eje.set_title(titulo, loc="left")
    eje.grid(axis="y", visible=False)
    eje.xaxis.set_major_formatter(graficos.formato_fijo(0))
eje_eta.set_yticks(posiciones, EFECTOS["etiqueta"])
eje_eta.invert_yaxis()
eje_shap.set_xlabel("Modelo del tablero")
eje_eta.xaxis.set_major_locator(MultipleLocator(5))
eje_shap.xaxis.set_major_locator(MultipleLocator(2))
eje_eta.set_xlabel("Pruebas de la fase 3")
figura.tight_layout(w_pad=0.8)
graficos.anotar_fuente(eje_eta, f"{FUENTE} SHAP sobre {en_espanol(FILAS_INTERPRETACION)} estudiantes de validación.")
ruta_eta_shap = figuras.guardar(figura, "eta-shap")
plt.close(figura)
print(f"Escrita {relativa(ruta_eta_shap)}")
cifra("etaCuadradoHUno", eta_h1, 3)
for columna, sufijo in (("FAMI_TIENEINTERNET", "Internet"), (v.EDUCACION_MADRE, "EducacionMadre"), ("COLE_NATURALEZA", "Naturaleza"), (v.ESTRATO, "Estrato")):
    cifra(f"shapTablero{sufijo}", EFECTOS.loc[columna, "shap"], 1)
    cifra(f"puestoShapTablero{sufijo}", EFECTOS.loc[columna, "puesto_shap"])
cifra("variablesShapTablero", len(importancia_tablero));

# %% [markdown]
# **Lectura.** Las dos miradas no ordenan igual las variables. Por sí solas, la educación de la madre explica el 13,7 % de la varianza del puntaje global y el internet el 7,5 %, pero en el modelo, donde ya están el computador, el estrato, el departamento y la edad, sus contribuciones medias son de 4,4 y 2,8 puntos (puestos 6 y 12 de 19): buena parte de lo que explican por separado lo comparten con otras variables del hogar. La naturaleza del colegio va al revés: su η² parcial es pequeño (4,1 %), pero en el modelo es la cuarta variable (4,9 puntos), lo que concuerda con la interacción con el estrato que encontró H3. El tamaño de efecto de una prueba dice cuánto separa una variable los grupos por sí sola; SHAP dice cuánto mueve la estimación cuando las demás ya están, y no dice si una diferencia es significativa. Se leen juntos. Las tres variables de mayor SHAP (edad, departamento y computador) son las mismas que encontró la fase 3 con su XGBoost, así que el reajuste no cambió la lectura del modelo.

# %% [markdown]
# ## 8. Cifras y comprobaciones
#
# Última sección del cuaderno: escribe las cifras de los tres documentos de la fase 4 (las consolidadas de las fases 1 a 3 más las nuevas) y confirma que cada figura existe.

# %%
for ruta in figuras.escritas:
    comprobar(f"Existe {relativa(ruta)}", ruta.is_file())
ruta_json, ruta_tex = cifras.volcar(RUTA_RESULTADOS, "cifras-fase-4")
macros = [linea for linea in ruta_tex.read_text(encoding="utf-8").splitlines() if linea.startswith("\\newcommand")]
comprobar(f"Cifras volcadas ({en_espanol(len(cifras))}) en {relativa(ruta_json)} y {relativa(ruta_tex)}, sin macros repetidas", len(macros) == len(cifras) == len({m.split('}')[0] for m in macros}))
print(f"Cifras consolidadas de las fases 1 a 3: {en_espanol(CONSOLIDADAS)}; nuevas de la fase 4: {en_espanol(len(cifras) - CONSOLIDADAS)}")
print(control.resumen())
