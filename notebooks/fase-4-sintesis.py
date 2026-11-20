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
# Las fases 1 a 3 dejaron sus resultados en archivos: las cifras de cada cuaderno (`resultados/cifras-fase-{1,2,3}.json`), los dos registros de evaluación en prueba, las salidas de los scripts de R y los agregados del dashboard. Esta fase no vuelve a calcular nada de eso, salvo los cinco clasificadores de la fase 3, que §8 ajusta otra vez con sus hiperparámetros solo para trazar de nuevo sus curvas. El cuaderno hace siete cosas:
#
# 1. Une las cifras de las tres fases en un solo registro, para que el documento del dashboard (A7), el informe final (A8) y las diapositivas (A9) carguen un único archivo de macros (§1).
# 2. Reúne las métricas de prueba de los quince modelos en un solo agregado, sin volver a evaluar ninguno (§2), y compara la estadística clásica con los modelos de aprendizaje automático (§3).
# 3. Consolida en una tabla la paridad entre Python y R de las tres fases (§4).
# 4. Agrega el puntaje por jornada, la característica del colegio que nombra la pregunta principal y que ningún agregado traía (§5).
# 5. Exporta el modelo de predicción del tablero, con su ficha y su margen, mide la memoria del dashboard (§6) y compara el tamaño de efecto de las hipótesis con la importancia SHAP de ese modelo (§7).
# 6. Traza para el documento del dashboard, con la letra con que se imprime, las curvas de clasificación de la fase 3 con los cinco umbrales a la vista y la interacción de H3 (§8), y para las diapositivas, con la letra con que se proyectan, cinco figuras al tamaño de su hueco (§9).
# 7. Escribe las cifras de la fase y comprueba que todo existe (última sección).
#
# Ninguna celda lee el conjunto de prueba: sus métricas salen de los registros que las fases 2 y 3 escribieron al evaluarlo una sola vez, y el modelo del tablero se elige y se mide en validación.

# %% [markdown]
# ### 0.2. Entorno
#
# El cuaderno corre con el entorno del repositorio (`uv sync`) después de los cuadernos de las fases 1 a 3 y de sus scripts de R, porque lee lo que ellos escriben (el orden está en el `README.md`). No usa la GPU.

# %%
import json
import platform
import textwrap
import warnings
from itertools import combinations
from importlib.metadata import version
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor

from saber11 import ajuste, datos, graficos, interpretabilidad, intermedios, limpieza, mlp, modelos, particion, tablero
from saber11 import variables as v
from saber11.cifras import Cifras, en_espanol
from saber11.comprobaciones import Comprobaciones

print(f"{'Python':<14}{platform.python_version()}")
for paquete in ("pandas", "numpy", "scikit-learn", "xgboost", "joblib", "matplotlib"):
    print(f"{paquete:<14}{version(paquete)}")

# %% [markdown]
# ### 0.3. Estilo, rutas y comprobaciones
#
# Las figuras conservan la estética de las fases anteriores (`saber11.graficos`) y se dibujan al ancho de una columna IEEE, con la letra con que se imprimen. `guardar_para_imprimir` las guarda con un margen mínimo y comprueba que la imagen no sea más ancha que la columna (252 puntos), para que el documento no la reduzca y su letra más pequeña, de 6,5 puntos, se imprima a ese tamaño o mayor. Las rutas se imprimen relativas a la raíz del repositorio. `comprobar` detiene el cuaderno si una condición falla.

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


ANCHO_IMPRESO = 252 / 72.27  # ancho de una columna IEEE, en pulgadas


def guardar_para_imprimir(figura: plt.Figure, nombre: str) -> Path:
    """Guarda la figura con un margen mínimo y comprueba que quepa en una columna IEEE sin reducirse."""
    with plt.rc_context({"savefig.pad_inches": 0.02}):
        ruta = figuras.guardar(figura, nombre)
    ancho = plt.imread(ruta).shape[1] / plt.rcParams["savefig.dpi"]
    comprobar(f"{relativa(ruta)} mide {en_espanol(ancho, 2)} pulgadas de ancho: cabe en una columna sin reducir su letra", ancho <= ANCHO_IMPRESO)
    return ruta


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
# La figura pone lado a lado, en prueba, los modelos sin IA y con IA de las dos tareas: el RMSE de la regresión del puntaje global (en puntos; menor es mejor) y el AUC-ROC de la clasificación del colegio oficial o no oficial (mayor es mejor). Cada punto es un modelo, no un estudiante, y el eje horizontal no empieza en cero para que se vean las diferencias, que son de pocos puntos o centésimas. La A8 la imprime al ancho de una columna, así que se guarda con `guardar_para_imprimir` y los valores junto a cada punto van a 6,5 puntos, como el resto de la letra.

# %%
COLOR_ENFOQUE = {SIN_IA: graficos.COLOR_APOYO, CON_IA: graficos.COLOR_PRINCIPAL}
ETIQUETA_ENFOQUE = {SIN_IA: "Sin IA: estadística clásica", CON_IA: "Con IA: aprendizaje automático"}
PANELES = (
    (REGRESION, "rmse", True, "Regresión del puntaje global", "RMSE de prueba (puntos; menor es mejor)", 1, 2),
    (CLASIFICACION, "auc", False, "Clasificación: oficial o no oficial", "AUC-ROC de prueba (mayor es mejor)", 3, 0.1),
)
figura, ejes = plt.subplots(2, 1, figsize=(graficos.ANCHO_COLUMNA - 0.1, 3.75), height_ratios=[9, 6])
for eje, (tarea, metrica, ascendente, titulo, etiqueta_x, decimales, paso) in zip(ejes, PANELES, strict=True):
    panel = metricas[metricas["tarea"] == tarea].sort_values(metrica, ascending=ascendente).reset_index(drop=True)
    posiciones = np.arange(len(panel))
    minimo, maximo = panel[metrica].min(), panel[metrica].max()
    margen = 0.12 * (maximo - minimo)
    eje.hlines(posiciones, minimo - margen, panel[metrica], color="#D9D9D9", linewidth=0.6, zorder=1)
    eje.scatter(panel[metrica], posiciones, s=16, color=panel["enfoque"].map(COLOR_ENFOQUE), zorder=2)
    for posicion, valor in zip(posiciones, panel[metrica], strict=True):
        eje.annotate(en_espanol(valor, decimales), (valor, posicion), xytext=(4, 0), textcoords="offset points", va="center", fontsize=6.5, bbox={"boxstyle": "square,pad=0.05", "facecolor": "white", "edgecolor": "none"})
    eje.set_yticks(posiciones, panel["etiqueta"])
    eje.set_ylim(len(panel) - 0.5, -0.5)  # invertido, con medio puesto libre arriba y abajo para que ninguna etiqueta toque el borde
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
ruta_con_sin_ia = guardar_para_imprimir(figura, "con-sin-ia")
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
# La A8 imprime también las métricas de prueba de los quince modelos en una tabla de una columna, `resultados/metricas-prueba.tex`, escrita desde el mismo `metricas-modelos.csv`: tres decimales en el R² y en las métricas de clasificación y dos en el RMSE y el MAE, que están en puntos. Las celdas que una fase no calculó (el AUC-PR de las bases de la fase 2) quedan vacías.

# %%
DECIMALES_TABLA = {"r2": 3, "rmse": 2, "mae": 2, "accuracy": 3, "f1": 3, "auc": 3, "auc_pr": 3}
COLUMNAS_TABLA = {REGRESION: ("r2", "rmse", "mae"), CLASIFICACION: ("accuracy", "f1", "auc", "auc_pr")}
ENCABEZADOS_TABLA = {REGRESION: (r"$R^2$", "RMSE", "MAE", ""), CLASIFICACION: ("Exact.", "F1", "AUC-ROC", "AUC-PR")}


def celda_metrica(valor: float, metrica: str) -> str:
    """Métrica con coma decimal y signo menos tipográfico; vacía si la fase no la calculó."""
    if pd.isna(valor):
        return ""
    decimales = DECIMALES_TABLA[metrica]
    redondeado = round(float(valor), decimales) + 0.0  # suma 0,0 para no escribir «-0,000»
    return f"$-${en_espanol(-redondeado, decimales)}" if redondeado < 0 else en_espanol(redondeado, decimales)


lineas_metricas = ["% Generado por notebooks/fase-4-sintesis.ipynb (sección 3): no editar a mano.", r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule"]
for tarea in (REGRESION, CLASIFICACION):
    if tarea == CLASIFICACION:
        lineas_metricas.append(r"\midrule")
    lineas_metricas += [" & ".join((tarea, "Grupo", *ENCABEZADOS_TABLA[tarea])) + r" \\", r"\midrule"]
    for _, fila in metricas[metricas["tarea"] == tarea].iterrows():
        celdas = [celda_metrica(fila[metrica], metrica) for metrica in COLUMNAS_TABLA[tarea]]
        lineas_metricas.append(" & ".join((fila["etiqueta"], fila["enfoque"], *celdas, *[""] * (4 - len(celdas)))) + r" \\")
lineas_metricas += [r"\bottomrule", r"\end{tabular}"]
ruta_metricas_tex = RUTA_RESULTADOS / "metricas-prueba.tex"
ruta_metricas_tex.write_text("\n".join(lineas_metricas) + "\n", encoding="utf-8")
print(f"Escrita {relativa(ruta_metricas_tex)}")
print(ruta_metricas_tex.read_text(encoding="utf-8"))
comprobar("La tabla de métricas de prueba tiene un modelo por fila y un encabezado por tarea", ruta_metricas_tex.read_text(encoding="utf-8").count("\\\\") == len(metricas) + 2)

# %% [markdown]
# La A8 describe también la configuración del MLP de la fase 3. Las neuronas de cada capa oculta, el abandono, la tasa de aprendizaje y el tamaño del lote son constantes de `saber11.mlp` y no resultados, pero se registran aquí como cifras para que el informe no las escriba a mano; las épocas máximas y la paciencia ya venían en las cifras de la fase 3, y se comprueba que coinciden con el módulo.

# %%
comprobar("El MLP de la fase 3 tiene dos capas ocultas, como lo describe la A8", len(mlp.OCULTAS) == 2)
comprobar("Las épocas máximas y la paciencia de las cifras de la fase 3 son las de saber11.mlp", cifras.valor("epocasMaximasMlp") == mlp.EPOCAS and cifras.valor("pacienciaMlp") == mlp.PACIENCIA)
cifra("neuronasPrimeraCapaMlp", mlp.OCULTAS[0])
cifra("neuronasSegundaCapaMlp", mlp.OCULTAS[1])
cifra("abandonoMlp", mlp.ABANDONO, 1)
cifra("tasaAprendizajeMlp", mlp.TASA, 3)
cifra("loteMlp", mlp.LOTE);

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
    r"\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{2.9cm}rl@{}}",  # cabe en una columna de la A8
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
# Máximo aproximado de memoria de una app en Community Cloud, según «Manage your app» de la
# documentación de Streamlit (límites de febrero de 2024: «Memory: 690MB minimum, 2.7GBs maximum»).
LIMITE_MEMORIA_MB = 2700
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

# Cada panel lleva su origen y su medida en un título de dos líneas centrado sobre su eje, sin rótulo del eje x,
# y los nombres largos de las variables van en dos líneas, para que los dos ejes ganen ancho y nada se toque.
figura, (eje_eta, eje_shap) = plt.subplots(1, 2, figsize=(graficos.ANCHO_COLUMNA - 0.1, 1.7), sharey=True)
posiciones = np.arange(len(EFECTOS))
for eje, columna, titulo, escala, decimales in (
    (eje_eta, "eta", "Pruebas de la fase 3\nη² o η² parcial (%)", 100, 1),
    (eje_shap, "shap", "Modelo del tablero\nSHAP medio (puntos)", 1, 1),
):
    valores = escala * EFECTOS[columna].to_numpy()
    eje.barh(posiciones, valores, height=0.6, color=graficos.COLOR_PRINCIPAL)
    for posicion, valor in zip(posiciones, valores, strict=True):
        eje.annotate(en_espanol(valor, decimales), (valor, posicion), xytext=(2, 0), textcoords="offset points", va="center")
    eje.set_xlim(0, 1.6 * valores.max())
    eje.set_title(titulo)
    eje.grid(axis="y", visible=False)
    eje.xaxis.set_major_formatter(graficos.formato_fijo(0))
eje_eta.set_yticks(posiciones, [textwrap.fill(etiqueta, 16) for etiqueta in EFECTOS["etiqueta"]])
eje_eta.invert_yaxis()
eje_eta.xaxis.set_major_locator(MultipleLocator(5))
eje_shap.xaxis.set_major_locator(MultipleLocator(2))
figura.tight_layout(w_pad=1.6)
# Ninguna etiqueta de valor ni ningún título se sale de su panel o pisa al otro.
figura.canvas.draw()
cajas_valores = {eje: [texto.get_window_extent() for texto in eje.texts] for eje in (eje_eta, eje_shap)}
comprobar(
    "En el panel de η² frente a SHAP, cada valor queda dentro de su eje y los dos títulos no se tocan",
    all(caja.x1 < eje.get_window_extent().x1 for eje, cajas in cajas_valores.items() for caja in cajas)
    and eje_eta.title.get_window_extent().x1 < eje_shap.title.get_window_extent().x0,
)
graficos.anotar_fuente(eje_eta, f"{FUENTE} SHAP sobre {en_espanol(FILAS_INTERPRETACION)} estudiantes de validación.")
ruta_eta_shap = guardar_para_imprimir(figura, "eta-shap")
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
# ## 8. Figuras para imprimir en el documento del dashboard
#
# El documento del dashboard (A7) imprime dos figuras que no tenían una versión a su tamaño. La primera son las curvas ROC y PR de la subpestaña Clasificación, que vienen de la fase 3 (`figuras/fase-3/clasificacion-curvas.png`, ya entregada, que no se toca). Impresa, esa figura tiene dos defectos: el eje vertical lleva marcas cada 0,25 con rótulos de un decimal, así que la marca de 0,25 dice «0,2» y la línea de azar de la PR, en 0,226, parece quedar por debajo de 0,2; y los cinco puntos de umbral caen tan juntos que dos quedan tapados. La segunda es la interacción de H3, que el documento usa para leer la brecha dentro de cada estrato y que el dashboard solo tiene en Plotly.
#
# ### 8.1. Curvas de clasificación con los cinco umbrales
#
# Para trazar las mismas curvas se ajustan otra vez los cinco clasificadores de la fase 3, con los hiperparámetros que eligió su búsqueda (leídos de `cifras-fase-3.json`) y con sus mismas filas: el entrenamiento completo y, en la SVM, la muestra de la fase 2, cuya huella se comprueba. La logística, Random Forest, la SVM y el LDA son deterministas con su semilla y dan el mismo modelo, lo que se comprueba con su AUC-ROC de validación. XGBoost entrenó en la GPU en la fase 3 y aquí entrena en la CPU, donde el submuestreo de filas y columnas sortea distinto, así que su curva puede diferir en milésimas. Cada punto es el umbral que eligió la fase 3 (el puntaje que maximiza el F1 de la clase no oficial en validación), y se comprueba que dé la exhaustividad y la precisión de la fase. Las marcas de los ejes van cada 0,2 con un decimal, y una segunda fila amplía la zona de los umbrales, con un marcador distinto por modelo.

# %%
CLASE = v.PREDICTORES_CLASIFICACION
muestra_fase_2 = intermedios.muestra_de_la_fase_2(limpio, conjuntos)
en_muestra = muestra_fase_2.loc[X_ent.index].to_numpy()
y_ent_clase, y_val_clase = (limpio.loc[m, v.OBJETIVO_CLASIFICACION].eq(v.CLASE_POSITIVA).astype(int) for m in (es_ent, es_val))
X_ent_clase, X_val_clase = X_ent[list(CLASE)], X_val[list(CLASE)]
filas_clase = {"entrenamiento": (X_ent_clase, y_ent_clase), "muestra": (X_ent_clase[en_muestra], y_ent_clase[en_muestra])}
peso_no_oficial = float((y_ent_clase == 0).sum() / (y_ent_clase == 1).sum())
comprobar("La muestra de la SVM tiene el tamaño de la de la fase 2", len(filas_clase["muestra"][0]) == cifras.valor("tamanoMuestra"))
comprobar("El peso de la clase no oficial es el que usó XGBoost en la fase 3", abs(peso_no_oficial - cifras.valor("pesoNoOficialXgboost")) < 1e-9)

ENTEROS_BUSQUEDA = {"n_estimators", "max_depth", "min_samples_leaf"}


def de_la_busqueda(nombre: str, sufijo: str) -> dict:
    """Hiperparámetros que eligió la búsqueda de la fase 3, leídos de sus cifras (`mejor<Modelo><Parámetro>`)."""
    elegidos = {}
    for clave in ajuste.ESPACIOS[nombre]:
        parametro = clave.removeprefix("modelo__")
        valor = cifras.valor(f"mejor{sufijo}" + "".join(parte.capitalize() for parte in parametro.split("_")))
        elegidos[parametro] = None if valor == "sin límite" else int(valor) if parametro in ENTEROS_BUSQUEDA else valor
    return elegidos


def puntaje_clase(modelo, X: pd.DataFrame) -> np.ndarray:
    """Puntaje de la clase no oficial: la probabilidad o, en la SVM, la función de decisión."""
    if isinstance(modelo.named_steps["modelo"], SVC):
        return modelo.decision_function(X)
    return modelo.predict_proba(X)[:, 1]


SUFIJO_CLASE = {"logisticaUmbral": "LogisticaUmbral", "randomForestClasificacion": "RandomForestClasificacion", "xgboostClasificacion": "XgboostClasificacion", "svmClasificacion": "Svm", "ldaClasificacion": "Lda"}
ETIQUETA_CLASE = {"logisticaUmbral": "Logística base", "randomForestClasificacion": "Random Forest", "xgboostClasificacion": "XGBoost", "svmClasificacion": "SVM en la muestra", "ldaClasificacion": "LDA"}
bosque = RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=ajuste.SEMILLA, **de_la_busqueda("randomForestClasificacion", "RandomForestClasificacion"))
refuerzo = XGBClassifier(device="cpu", scale_pos_weight=peso_no_oficial, random_state=ajuste.SEMILLA, **de_la_busqueda("xgboostClasificacion", "XgboostClasificacion"))
clasificadores = {
    "logisticaUmbral": (modelos.logistica_base(), "entrenamiento"),
    "randomForestClasificacion": (modelos.pipeline(CLASE, bosque), "entrenamiento" if ajuste.REAJUSTE_RANDOM_FOREST else "muestra"),
    "xgboostClasificacion": (modelos.pipeline(CLASE, refuerzo), "entrenamiento"),
    "svmClasificacion": (modelos.pipeline(CLASE, SVC(kernel="rbf", class_weight="balanced", **de_la_busqueda("svmClasificacion", "Svm"))), "muestra"),
    "ldaClasificacion": (modelos.pipeline(CLASE, LinearDiscriminantAnalysis()), "entrenamiento"),
}
positivos = y_val_clase.to_numpy() == 1
curvas = {}
for nombre, (modelo, filas) in clasificadores.items():
    ajuste_clase = ajustado(modelo, *filas_clase[filas])
    puntajes = puntaje_clase(ajuste_clase.modelo, X_val_clase)
    predicho = puntajes >= cifras.valor(f"umbral{SUFIJO_CLASE[nombre]}")
    curvas[nombre] = {
        "puntajes": puntajes,
        "auc": roc_auc_score(y_val_clase, puntajes),
        "tasa_fp": predicho[~positivos].mean(),
        "exhaustividad": predicho[positivos].mean(),
        "precision": positivos[predicho].mean(),
        "segundos": ajuste_clase.segundos,
    }
diferencias = pd.DataFrame({
    ETIQUETA_CLASE[nombre]: {
        "Ajuste (s)": c["segundos"],
        "AUC-ROC": c["auc"],
        "Dif. AUC-ROC": c["auc"] - cifras.valor(f"aucValidacion{SUFIJO_CLASE[nombre]}"),
        "Dif. exhaustividad": c["exhaustividad"] - cifras.valor(f"recallValidacion{SUFIJO_CLASE[nombre]}"),
        "Dif. precisión": c["precision"] - cifras.valor(f"precisionValidacion{SUFIJO_CLASE[nombre]}"),
    }
    for nombre, c in curvas.items()
}).T
print("Diferencias frente a la fase 3, en validación:")
tabla(diferencias, 4)
DETERMINISTAS = [ETIQUETA_CLASE[n] for n in clasificadores if n != "xgboostClasificacion"]
comprobar("La logística, Random Forest, la SVM y el LDA dan el AUC-ROC de validación de la fase 3 (diferencia menor que 1e-6)", diferencias.loc[DETERMINISTAS, "Dif. AUC-ROC"].abs().max() < 1e-6)
comprobar("XGBoost en la CPU da un AUC-ROC de validación a menos de 0,002 del de la fase 3", abs(diferencias.loc["XGBoost", "Dif. AUC-ROC"]) < 0.002)
comprobar("Con el umbral de la fase 3, cada modelo da su exhaustividad y su precisión a menos de 0,01", diferencias[["Dif. exhaustividad", "Dif. precisión"]].abs().max().max() < 0.01)

COLOR_CLASE = {"logisticaUmbral": graficos.COLOR_APOYO, **dict(zip(list(clasificadores)[1:], graficos.PALETA_CATEGORICA[2:], strict=False))}
MARCADOR = {"logisticaUmbral": "o", "randomForestClasificacion": "s", "xgboostClasificacion": "D", "svmClasificacion": "^", "ldaClasificacion": "v"}


def recuadro(xs: list[float], ys: list[float], holgura: float = 0.6) -> tuple[tuple[float, float], tuple[float, float]]:
    """Límites que contienen los puntos, ampliados en `holgura` veces su recorrido a cada lado."""
    return tuple((min(z) - holgura * (max(z) - min(z)), max(z) + holgura * (max(z) - min(z))) for z in (xs, ys))


zonas = {
    "roc": recuadro([c["tasa_fp"] for c in curvas.values()], [c["exhaustividad"] for c in curvas.values()]),
    "pr": recuadro([c["exhaustividad"] for c in curvas.values()], [c["precision"] for c in curvas.values()]),
}
figura, ((eje_roc, eje_pr), (detalle_roc, detalle_pr)) = plt.subplots(2, 2, figsize=(graficos.ANCHO_COLUMNA - 0.1, 3.3), height_ratios=[1.15, 1])
for nombre, c in curvas.items():
    tasa_fp, tasa_vp, _ = roc_curve(y_val_clase, c["puntajes"])
    precision, exhaustividad, _ = precision_recall_curve(y_val_clase, c["puntajes"])
    # Sin los puntos de exhaustividad cero, que dibujarían una raya vertical sobre el eje (como en la fase 3).
    con_aciertos = exhaustividad > 0
    estilo = {"color": COLOR_CLASE[nombre], "linewidth": 0.9, "linestyle": "--" if nombre == "logisticaUmbral" else "-"}
    marca = {"marker": MARCADOR[nombre], "s": 18, "color": COLOR_CLASE[nombre], "edgecolors": "white", "linewidths": 0.5, "zorder": 3}
    for eje in (eje_roc, detalle_roc):
        eje.plot(tasa_fp, tasa_vp, **estilo)
    for eje in (eje_pr, detalle_pr):
        eje.plot(exhaustividad[con_aciertos], precision[con_aciertos], **estilo)
    detalle_roc.scatter(c["tasa_fp"], c["exhaustividad"], **marca)
    detalle_pr.scatter(c["exhaustividad"], c["precision"], **marca)
eje_roc.plot([0, 1], [0, 1], color=graficos.COLOR_APOYO, linewidth=0.5, linestyle=":")
eje_pr.axhline(y_val_clase.mean(), color=graficos.COLOR_APOYO, linewidth=0.5, linestyle=":")
for eje, detalle, zona, (etiqueta_x, etiqueta_y), (letra, titulo) in (
    (eje_roc, detalle_roc, zonas["roc"], ("Tasa de falsos positivos", "Exhaustividad"), ("a", "Curva ROC")),
    (eje_pr, detalle_pr, zonas["pr"], ("Exhaustividad", "Precisión"), ("b", "Curva PR")),
):
    (x0, x1), (y0, y1) = zona
    eje.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#4C4C4C", linewidth=0.6, zorder=4))
    eje.set(xlabel=etiqueta_x, ylabel=etiqueta_y, title=f"({letra}) {titulo}", xlim=(0, 1), ylim=(0, 1.02))
    detalle.set(xlabel=etiqueta_x, ylabel=etiqueta_y, title=f"({chr(ord(letra) + 2)}) Recuadro de ({letra})", xlim=(x0, x1), ylim=(y0, y1))
    # Rótulos con tantos decimales como pide el paso de las marcas, para que cada uno diga el valor de la suya.
    for panel, paso, decimales in ((eje, 0.2, 1), (detalle, 0.02, 2)):
        for marcas in (panel.xaxis, panel.yaxis):
            marcas.set_major_locator(MultipleLocator(paso))
            marcas.set_major_formatter(graficos.formato_fijo(decimales))
figura.tight_layout(h_pad=0.6, w_pad=1.0)
manijas = [
    Line2D([], [], color=COLOR_CLASE[n], linewidth=0.9, linestyle="--" if n == "logisticaUmbral" else "-", marker=MARCADOR[n], markersize=4, markeredgecolor="white", markeredgewidth=0.5, label=f"{ETIQUETA_CLASE[n]} ({en_espanol(c['auc'], 3)})")
    for n, c in curvas.items()
]
figura.legend(handles=manijas, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, title="Modelo (AUC-ROC en validación)", handlelength=2.2, columnspacing=1.0)
graficos.anotar_fuente(detalle_roc, FUENTE)
ruta_curvas = guardar_para_imprimir(figura, "clasificacion-curvas")
plt.close(figura)
print(f"Escrita {relativa(ruta_curvas)}")

# %% [markdown]
# **Lectura.** Las curvas de la logística, Random Forest, la SVM y el LDA son las de la fase 3: dan su mismo AUC-ROC de validación (0,762; 0,769; 0,759 y 0,756). XGBoost, ajustado en la CPU, da 0,772, dos diezmilésimas más que en la GPU, y con el umbral de la fase 3 su exhaustividad y su precisión se mueven menos de tres milésimas. Los recuadros muestran lo que la figura de la fase 3 tapaba: los cinco umbrales caen en una franja de unas tres centésimas de exhaustividad, y a igual exhaustividad XGBoost tiene la mayor precisión y el LDA la menor.

# %% [markdown]
# ### 8.2. Brecha entre colegios no oficiales y oficiales dentro de cada estrato
#
# La interacción de H3 se dibuja con el agregado que lee el dashboard (`data/agregados/hipotesis-interaccion-20224.csv`): la media del puntaje global por estrato y naturaleza del colegio, con su intervalo del 95 %. Encima de cada estrato va la ventaja de los no oficiales, que tiene que coincidir con la brecha de cada estrato que la fase 3 calculó con Welch.

# %%
interaccion = pd.read_csv(RUTA_AGREGADOS / "hipotesis-interaccion-20224.csv")
comprobar("Cada celda del agregado de la interacción tiene 10 o más estudiantes", interaccion["n"].min() >= MINIMO_POR_CELDA)
comprobar("El agregado de la interacción suma los estudiantes de H3", interaccion["n"].sum() == cifras.valor("nHTres"))
medias_h3 = interaccion.pivot(index="estrato", columns="naturaleza", values="media")
brechas_h3 = medias_h3["NO OFICIAL"] - medias_h3["OFICIAL"]
PALABRA_ESTRATO = {"Estrato 1": "Uno", "Estrato 2": "Dos", "Estrato 3": "Tres", "Estrato 4": "Cuatro", "Estrato 5": "Cinco", "Estrato 6": "Seis"}
comprobar(
    "La ventaja de los no oficiales en cada estrato es la brecha que calculó la fase 3",
    all(abs(brechas_h3[estrato] - cifras.valor(f"brechaEstrato{palabra}")) < 1e-3 for estrato, palabra in PALABRA_ESTRATO.items()),
)
NATURALEZAS = {"NO OFICIAL": ("No oficial", graficos.COLOR_PRINCIPAL, "-"), "OFICIAL": ("Oficial", graficos.COLOR_APOYO, "--")}
# Marcador pequeño y tope del intervalo más ancho que él, para que se vea dónde hay intervalo aunque sea angosto.
DIAMETRO_MARCADOR = 2.2  # en puntos
posiciones = np.arange(len(medias_h3))
figura, eje = plt.subplots(figsize=(graficos.ANCHO_COLUMNA - 0.1, 1.9))
# «Sin estrato» no sigue al estrato 6, así que su punto va suelto, sin la línea de los estratos.
ordinales = medias_h3.index.str.startswith("Estrato ")
for naturaleza, (etiqueta, color, linea) in NATURALEZAS.items():
    grupo = interaccion[interaccion["naturaleza"] == naturaleza].set_index("estrato").loc[medias_h3.index]
    errores = np.array([grupo["media"] - grupo["inferior"], grupo["superior"] - grupo["media"]])
    estilo = {"color": color, "marker": "o", "markersize": DIAMETRO_MARCADOR, "elinewidth": 0.6, "capsize": 2.2, "capthick": 0.6}
    eje.errorbar(posiciones[ordinales], grupo["media"][ordinales], yerr=errores[:, ordinales], linestyle=linea, linewidth=0.9, label=etiqueta, **estilo)
    eje.errorbar(posiciones[~ordinales], grupo["media"][~ordinales], yerr=errores[:, ~ordinales], linestyle="none", **estilo)
for posicion, brecha in zip(posiciones, brechas_h3, strict=True):
    eje.annotate(en_espanol(brecha, 1), (posicion, 1), xycoords=("data", "axes fraction"), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom")
eje.set_title("Ventaja de los no oficiales, en puntos", loc="left", pad=11)
eje.set_xticks(posiciones, [estrato.removeprefix("Estrato ").replace("Sin Estrato", "Sin estrato") for estrato in medias_h3.index])
eje.set(xlabel="Estrato de la vivienda", ylabel="Puntaje global medio")
eje.yaxis.set_major_formatter(graficos.formato_fijo(0))
eje.legend(loc="lower left", ncol=2)
figura.tight_layout()
# Con miles de estudiantes por grupo, muchos intervalos son más angostos que el marcador: se cuentan para decirlo en el pie.
figura.canvas.draw()
puntos_por_unidad = eje.get_window_extent().height * 72 / figura.dpi / np.ptp(eje.get_ylim())
semiancho_ic = (interaccion["superior"] - interaccion["inferior"]) / 2
intervalos_ocultos = int((semiancho_ic * puntos_por_unidad < DIAMETRO_MARCADOR / 2).sum())
print(f"Intervalos más angostos que el radio del marcador: {intervalos_ocultos} de {len(interaccion)}")
comprobar("La mayoría de los intervalos del 95 % de la interacción es más angosta que el marcador, como dice el pie", intervalos_ocultos > len(interaccion) / 2)
graficos.anotar_fuente(eje, f"{FUENTE} Medias con su IC del 95 %, más angosto que el marcador en {intervalos_ocultos} de los {len(interaccion)} grupos.")
ruta_brecha = guardar_para_imprimir(figura, "brecha-estrato")
plt.close(figura)
print(f"Escrita {relativa(ruta_brecha)}")
cifra("intervalosOcultosBrecha", intervalos_ocultos)
cifra("gruposBrecha", len(interaccion))
cifra("menorCeldaInteraccion", int(interaccion["n"].min()));

# %% [markdown]
# **Lectura.** La ventaja de los no oficiales es positiva en los siete grupos, pero crece de 4,5 puntos en el estrato 1 a 91,4 en el 6, porque la media de los oficiales sube hasta el estrato 3 y luego baja hasta quedar en el 6 por debajo de la del 1, mientras la de los no oficiales sube hasta el 5. Como cada grupo reúne miles de estudiantes, la mayoría de los intervalos del 95 % es más angosta que el marcador, y sus topes, más anchos que él, señalan dónde están. Es la interacción que H3 encontró con la ANOVA de dos vías: la brecha nacional, de 29,0 puntos, esconde que en el estrato 1 casi no existe y que en los estratos 5 y 6 supera los 80.

# %% [markdown]
# ## 9. Figuras para proyectar en las diapositivas
#
# Las diapositivas de la sustentación (A9) tienen formato de 16:9 y una caja de texto de 140 mm de ancho y unos 2,9 pulgadas de alto bajo el título. Las figuras de §3 y §8 se dibujaron para una columna IEEE impresa, con letra de 6,5 puntos y proporciones que no caben a lo alto de una diapositiva sin reducirse, de modo que proyectadas se leerían con una letra aún menor. Por eso cinco de ellas se trazan otra vez con los mismos datos y la misma estética, al tamaño exacto del hueco que ocupan en la diapositiva y con letra de 8,5 a 9,5 puntos: la interacción de H3, el RMSE de prueba de los nueve modelos de regresión, las curvas ROC y PR de los cinco clasificadores, los perfiles de los dos clústeres con su reparto y la curva de pérdida del MLP. `guardar_para_proyectar` comprueba que cada imagen quepa en su hueco (media diapositiva o el ancho completo), para que la diapositiva la incluya a su tamaño natural. No se calcula nada nuevo: las métricas son las de §2, las curvas las de §8.1, la interacción la de §8.2, los clústeres salen de los agregados que lee el dashboard y la historia de pérdidas del MLP se lee de la caché que dejó su entrenamiento en la fase 3, sin volver a entrenarlo.

# %%
ESTILO_DIAPOSITIVA = {
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "legend.title_fontsize": 8.5,
}
ANCHO_DIAPOSITIVA = 140 / 25.4  # caja de texto de una diapositiva de 16:9 con márgenes de 1 cm, en pulgadas
HUECOS = {"media": (0.6 * ANCHO_DIAPOSITIVA, 2.75), "completa": (ANCHO_DIAPOSITIVA, 2.6)}  # ancho y alto máximos, en pulgadas


def guardar_para_proyectar(figura: plt.Figure, nombre: str, hueco: str) -> Path:
    """Guarda la figura de una diapositiva y comprueba que quepa en su hueco sin reducirse."""
    with plt.rc_context({"savefig.pad_inches": 0.02}):
        ruta = figuras.guardar(figura, f"diapositiva-{nombre}")
    alto, ancho = (lado / plt.rcParams["savefig.dpi"] for lado in plt.imread(ruta).shape[:2])
    ancho_maximo, alto_maximo = HUECOS[hueco]
    comprobar(
        f"{relativa(ruta)} mide {en_espanol(ancho, 2)} por {en_espanol(alto, 2)} pulgadas: cabe en el hueco «{hueco}» de la diapositiva sin reducir su letra",
        ancho <= ancho_maximo and alto <= alto_maximo,
    )
    return ruta


def leyenda_enfoque() -> list[Line2D]:
    """Manijas de la leyenda de los dos grupos de modelos."""
    return [Line2D([], [], marker="o", linestyle="", color=COLOR_ENFOQUE[e], markersize=5, label=ETIQUETA_ENFOQUE[e]) for e in (SIN_IA, CON_IA)]


# Interacción de H3: medias por estrato y naturaleza, con la ventaja de los no oficiales encima.
estratos_h3 = np.arange(len(medias_h3))
ordinales_h3 = medias_h3.index.str.startswith("Estrato ")
with plt.rc_context(ESTILO_DIAPOSITIVA):
    figura, eje = plt.subplots(figsize=(3.25, 2.35))
    for naturaleza, (etiqueta, color, linea) in NATURALEZAS.items():
        medias = medias_h3[naturaleza].to_numpy()
        eje.plot(estratos_h3[ordinales_h3], medias[ordinales_h3], color=color, linestyle=linea, linewidth=1.2, marker="o", markersize=4, label=etiqueta)
        eje.plot(estratos_h3[~ordinales_h3], medias[~ordinales_h3], color=color, linestyle="none", marker="o", markersize=4)
    for estrato, brecha in zip(estratos_h3, brechas_h3, strict=True):
        eje.annotate(en_espanol(brecha, 1), (estrato, 1), xycoords=("data", "axes fraction"), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)
    eje.set_title("Ventaja de los no oficiales, en puntos", loc="left", pad=13)
    eje.set_xticks(estratos_h3, [e.removeprefix("Estrato ").replace("Sin Estrato", "Sin\nestrato") for e in medias_h3.index])
    eje.set(xlabel="Estrato de la vivienda", ylabel="Puntaje global medio")
    eje.yaxis.set_major_formatter(graficos.formato_fijo(0))
    eje.legend(loc="center right", ncol=1, handlelength=1.8, labelspacing=0.2)
    figura.tight_layout()
    graficos.anotar_fuente(eje, f"{FUENTE} Cada grupo reúne {en_espanol(int(interaccion['n'].min()))} o más estudiantes.")
    ruta_diapositiva_brecha = guardar_para_proyectar(figura, "brecha", "media")
plt.close(figura)

# RMSE de prueba de los nueve modelos de regresión, del mejor al peor, con el eje recortado como en §3.
regresion = metricas[metricas["tarea"] == REGRESION].sort_values("rmse").reset_index(drop=True)
ETIQUETA_CORTA = {"Media del entrenamiento": "Media", "Regresión múltiple": "Múltiple"}  # nombres que caben junto al eje
with plt.rc_context(ESTILO_DIAPOSITIVA):
    figura, eje = plt.subplots(figsize=(3.25, 2.5))
    puestos = np.arange(len(regresion))
    minimo, maximo = regresion["rmse"].min(), regresion["rmse"].max()
    margen = 0.1 * (maximo - minimo)
    eje.hlines(puestos, minimo - margen, regresion["rmse"], color="#D9D9D9", linewidth=0.8, zorder=1)
    eje.scatter(regresion["rmse"], puestos, s=26, color=regresion["enfoque"].map(COLOR_ENFOQUE), zorder=2)
    for puesto, valor in zip(puestos, regresion["rmse"], strict=True):
        eje.annotate(en_espanol(valor, 2), (valor, puesto), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8.5, bbox={"boxstyle": "square,pad=0.05", "facecolor": "white", "edgecolor": "none"})
    eje.set_yticks(puestos, regresion["etiqueta"].replace(ETIQUETA_CORTA))
    eje.set_ylim(len(regresion) - 0.25, -0.75)  # holgura arriba y abajo, para que ninguna etiqueta tape el marco
    eje.set_xlim(minimo - margen, maximo + 3.4 * margen)
    eje.xaxis.set_major_locator(MultipleLocator(2))
    eje.xaxis.set_major_formatter(graficos.formato_fijo(0))
    eje.set_xlabel("RMSE de prueba, en puntos")
    eje.grid(axis="y", visible=False)
    figura.legend(handles=leyenda_enfoque(), loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=1, handletextpad=0.2, labelspacing=0.2)
    figura.tight_layout(rect=(0, 0, 1, 0.84))
    graficos.anotar_fuente(eje, f"{FUENTE} Prueba: {en_espanol(FILAS_PRUEBA)} estudiantes.")
    ruta_diapositiva_regresion = guardar_para_proyectar(figura, "regresion", "media")
plt.close(figura)
comprobar("En la figura de regresión de la diapositiva, XGBoost es el primero y la media del entrenamiento el último", regresion["modelo"].iloc[0] == "xgboostRegresion" and regresion["modelo"].iloc[-1] == "dummyRegresion")

# Curvas ROC y PR de validación de los cinco clasificadores de §8.1, con el umbral elegido de cada uno.
ETIQUETA_DIAPOSITIVA = {**ETIQUETA_CLASE, "svmClasificacion": "SVM (muestra)"}
with plt.rc_context(ESTILO_DIAPOSITIVA):
    figura, (eje_roc, eje_pr, eje_leyenda) = plt.subplots(1, 3, figsize=(5.4, 2.15), width_ratios=[1, 1, 0.72])
    for nombre, c in curvas.items():
        tasa_fp, tasa_vp, _ = roc_curve(y_val_clase, c["puntajes"])
        precision, exhaustividad, _ = precision_recall_curve(y_val_clase, c["puntajes"])
        con_aciertos = exhaustividad > 0
        estilo = {"color": COLOR_CLASE[nombre], "linewidth": 1.1, "linestyle": "--" if nombre == "logisticaUmbral" else "-"}
        marca = {"marker": MARCADOR[nombre], "s": 22, "color": COLOR_CLASE[nombre], "edgecolors": "white", "linewidths": 0.5, "zorder": 3}
        eje_roc.plot(tasa_fp, tasa_vp, **estilo)
        eje_pr.plot(exhaustividad[con_aciertos], precision[con_aciertos], **estilo)
        eje_roc.scatter(c["tasa_fp"], c["exhaustividad"], **marca)
        eje_pr.scatter(c["exhaustividad"], c["precision"], **marca)
    eje_roc.plot([0, 1], [0, 1], color=graficos.COLOR_APOYO, linewidth=0.6, linestyle=":")
    eje_pr.axhline(y_val_clase.mean(), color=graficos.COLOR_APOYO, linewidth=0.6, linestyle=":")
    for eje, (etiqueta_x, etiqueta_y), titulo in (
        (eje_roc, ("Falsos positivos", "Exhaustividad"), "(a) Curva ROC"),
        (eje_pr, ("Exhaustividad", "Precisión"), "(b) Curva PR"),
    ):
        eje.set(xlabel=etiqueta_x, ylabel=etiqueta_y, xlim=(0, 1), ylim=(0, 1.02))
        eje.set_title(titulo, loc="left")
        for marcas in (eje.xaxis, eje.yaxis):
            marcas.set_major_locator(MultipleLocator(0.5))
            marcas.set_major_formatter(graficos.formato_fijo(1))
    eje_leyenda.axis("off")
    manijas = [
        Line2D([], [], color=COLOR_CLASE[n], linewidth=1.1, linestyle="--" if n == "logisticaUmbral" else "-", marker=MARCADOR[n], markersize=5, markeredgecolor="white", markeredgewidth=0.5, label=ETIQUETA_DIAPOSITIVA[n])
        for n in curvas
    ] + [Line2D([], [], color=graficos.COLOR_APOYO, linewidth=0.6, linestyle=":", label="Azar")]
    eje_leyenda.legend(handles=manijas, loc="center left", bbox_to_anchor=(0.0, 0.5), title="En validación;\nel marcador es\nel umbral elegido", alignment="left", handlelength=1.8, borderaxespad=0)
    figura.tight_layout(w_pad=1.2)
    graficos.anotar_fuente(eje_roc, f"{FUENTE} Validación: {en_espanol(len(y_val_clase))} estudiantes; positiva, la clase no oficial.")
    ruta_diapositiva_clasificacion = guardar_para_proyectar(figura, "clasificacion", "completa")
plt.close(figura)

# Perfiles de los dos clústeres de K-Means por área y porcentaje de cada grupo que cae en el clúster alto.
perfiles = pd.read_csv(RUTA_AGREGADOS / "cluster-perfiles-20224.csv").set_index("cluster")
alto, bajo = perfiles["puntaje_global"].idxmax(), perfiles["puntaje_global"].idxmin()
comprobar(
    "Los puntajes globales medios de los clústeres son los de la fase 3",
    all(abs(perfiles.loc[c, "puntaje_global"] - cifras.valor(f"globalCluster{p}")) < 0.05 for c, p in ((1, "Uno"), (2, "Dos"))),
)
AREAS = {"lectura_critica": "Lectura", "matematicas": "Matemáticas", "ciencias_naturales": "Naturales", "sociales_ciudadanas": "Sociales", "ingles": "Inglés"}
REPARTOS = (
    ("cluster-naturaleza-20224.csv", "naturaleza", {"OFICIAL": "Oficial", "NO OFICIAL": "No oficial"}),
    ("cluster-zona-20224.csv", "zona", {"URBANO": "Urbano", "RURAL": "Rural"}),
    ("cluster-estrato-20224.csv", "estrato", {f"Estrato {k}": f"Estrato {k}" for k in range(1, 7)} | {"Sin Estrato": "Sin estrato"}),
)
reparto = pd.concat(
    pd.read_csv(RUTA_AGREGADOS / archivo).loc[lambda marco: marco["cluster"] == alto].set_index(columna).loc[list(niveles)].rename(index=niveles)[["n", "porcentaje"]]
    for archivo, columna, niveles in REPARTOS
)
comprobar("Cada celda del reparto por clúster tiene 10 o más estudiantes", reparto["n"].min() >= MINIMO_POR_CELDA)
MACROS_REPARTO = {"Oficial": "clusterAltoOficial", "No oficial": "clusterAltoNoOficial", "Urbano": "clusterAltoUrbano", "Rural": "clusterAltoRural", "Estrato 1": "clusterAltoEstratoUno", "Estrato 4": "clusterAltoEstratoCuatro", "Sin estrato": "clusterAltoSinEstrato"}
comprobar("Los porcentajes del clúster alto son los que citan los documentos", all(abs(reparto.loc[g, "porcentaje"] - cifras.valor(m)) < 0.05 for g, m in MACROS_REPARTO.items()))
COLOR_CLUSTER = {bajo: graficos.COLOR_APOYO, alto: graficos.COLOR_PRINCIPAL}
NOMBRE_CLUSTER = {bajo: "Bajo", alto: "Alto"}
with plt.rc_context(ESTILO_DIAPOSITIVA):
    figura, (eje_perfil, eje_reparto) = plt.subplots(1, 2, figsize=(5.4, 2.3), width_ratios=[1, 1.05])
    posiciones_area = np.arange(len(AREAS))
    for cluster in (alto, bajo):
        medias = perfiles.loc[cluster, list(AREAS)].to_numpy(dtype=float)
        eje_perfil.plot(posiciones_area, medias, color=COLOR_CLUSTER[cluster], marker="o", markersize=4, linewidth=1.2, label=f"{NOMBRE_CLUSTER[cluster]} ({en_espanol(perfiles.loc[cluster, 'porcentaje'], 1)} %)")
    eje_perfil.set_xticks(posiciones_area, list(AREAS.values()), rotation=30, ha="right", rotation_mode="anchor")
    eje_perfil.set_ylim(30, 70)
    eje_perfil.set_ylabel("Media del área (0 a 100)")
    eje_perfil.yaxis.set_major_formatter(graficos.formato_fijo(0))
    eje_perfil.set_title("(a) Perfil por área", loc="left")
    eje_perfil.legend(loc="center", ncol=1, handlelength=1.6, labelspacing=0.2, title="Clúster", alignment="left")
    filas_reparto = np.arange(len(reparto))
    eje_reparto.barh(filas_reparto, reparto["porcentaje"], height=0.65, color=graficos.COLOR_PRINCIPAL)
    for fila, valor in zip(filas_reparto, reparto["porcentaje"], strict=True):
        eje_reparto.annotate(en_espanol(valor, 1), (valor, fila), xytext=(2, 0), textcoords="offset points", va="center", fontsize=8, bbox={"boxstyle": "square,pad=0.05", "facecolor": "white", "edgecolor": "none"})
    eje_reparto.axvline(perfiles.loc[alto, "porcentaje"], color=graficos.COLOR_APOYO, linestyle=":", linewidth=0.9)
    eje_reparto.set_yticks(filas_reparto, reparto.index)
    eje_reparto.tick_params(axis="y", labelsize=8)
    eje_reparto.set_ylim(len(reparto) - 0.4, -0.6)
    eje_reparto.set_xlim(0, 100)
    eje_reparto.xaxis.set_major_locator(MultipleLocator(25))
    eje_reparto.xaxis.set_major_formatter(graficos.formato_fijo(0))
    eje_reparto.set_xlabel("% del grupo (punteada: total)")
    eje_reparto.set_title("(b) Peso del clúster alto", loc="left")
    eje_reparto.grid(axis="y", visible=False)
    figura.tight_layout(w_pad=1.0)
    graficos.anotar_fuente(eje_perfil, f"{FUENTE} K-Means sobre las cinco áreas; entrenamiento, {en_espanol(int(perfiles['n'].sum()))} estudiantes.")
    ruta_diapositiva_segmentacion = guardar_para_proyectar(figura, "segmentacion", "completa")
plt.close(figura)
comprobar("Los estudiantes de los dos clústeres son los del análisis multivariante de la fase 3", int(perfiles["n"].sum()) == cifras.valor("filasMultivariante"))


def entrenamiento_del_mlp() -> mlp.Entrenamiento:
    """El entrenamiento del MLP que guardó la caché de la fase 3 y que reportan sus cifras; no reentrena nada."""
    guardados = [joblib.load(ruta) for ruta in sorted(ajuste.CARPETA_CACHE.glob("mlp-*/saber11/mlp/entrenar/*/output.pkl"))]
    del_reporte = [
        red
        for red in guardados
        if red.epocas_corridas == cifras.valor("epocaParada")
        and red.epoca_mejor == cifras.valor("epocaMejorMlp")
        and abs(min(red.perdidas_validacion) - cifras.valor("perdidaValidacionMlp")) < 5e-4
    ]
    comprobar("La caché del MLP guarda el entrenamiento que reportan las cifras de la fase 3 (épocas, mejor época y pérdida de validación)", len(del_reporte) > 0)
    return del_reporte[-1]


# Curva de pérdida del MLP: las mismas curvas y referencias de la fase 3, con el RMSE en puntos a la derecha.
red = entrenamiento_del_mlp()
comprobar("La pérdida mínima de validación, en puntos, es el RMSE de validación del MLP de la fase 3", abs(np.sqrt(min(red.perdidas_validacion)) * red.escala - cifras.valor("rmseValidacionMlp")) < 5e-3)
epocas_red = np.arange(1, red.epocas_corridas + 1)
REFERENCIAS_MLP = (("Múltiple", "rmseValidacionMultiple", graficos.PALETA_CATEGORICA[3]), ("XGBoost", "rmseValidacionXgboost", graficos.PALETA_CATEGORICA[2]))
with plt.rc_context(ESTILO_DIAPOSITIVA):
    figura, eje = plt.subplots(figsize=(3.2, 2.5))
    eje.plot(epocas_red, red.perdidas_entrenamiento, color=graficos.COLOR_APOYO, linewidth=1.0, label="Entrenamiento")
    eje.plot(epocas_red, red.perdidas_validacion, color=graficos.COLOR_PRINCIPAL, linewidth=1.2, label="Validación")
    eje.axvline(red.epoca_mejor, color=graficos.PALETA_CATEGORICA[1], linestyle="--", linewidth=0.8, label=f"Mejor época ({red.epoca_mejor})")
    for etiqueta, clave, color in REFERENCIAS_MLP:
        eje.axhline((cifras.valor(clave) / red.escala) ** 2, color=color, linestyle=":", linewidth=0.9, label=f"{etiqueta}, validación")
    eje.set(xlabel="Época", ylabel="MSE estandarizado", xlim=(0, red.epocas_corridas + 1))
    eje.yaxis.set_major_locator(MultipleLocator(0.05))
    eje.yaxis.set_major_formatter(graficos.formato_fijo(2))
    derecho = eje.secondary_yaxis("right", functions=(lambda m: np.sqrt(np.clip(m, 0, None)) * red.escala, lambda r: (np.asarray(r) / red.escala) ** 2))
    derecho.set_ylabel("RMSE en puntos")
    derecho.yaxis.set_major_locator(MultipleLocator(1))
    derecho.yaxis.set_major_formatter(graficos.formato_fijo(0))
    # Las curvas quedan en la mitad de abajo y la leyenda arriba, sin tapar ninguna línea.
    inferior = min(min(red.perdidas_entrenamiento), min(red.perdidas_validacion)) - 0.01
    superior = max(red.perdidas_entrenamiento)
    eje.set_ylim(inferior, superior + 0.35 * (superior - inferior))
    eje.legend(loc="upper right", handlelength=1.8, labelspacing=0.2, frameon=True, framealpha=0.95, edgecolor="none")
    figura.tight_layout()
    graficos.anotar_fuente(eje, f"{FUENTE} Validación: {en_espanol(int(cifras.valor('filasValidacion')))} estudiantes; el entrenamiento, con el abandono activo.")
    ruta_diapositiva_mlp = guardar_para_proyectar(figura, "mlp", "media")
plt.close(figura)
for ruta in (ruta_diapositiva_brecha, ruta_diapositiva_regresion, ruta_diapositiva_clasificacion, ruta_diapositiva_segmentacion, ruta_diapositiva_mlp):
    print(f"Escrita {relativa(ruta)}")

# %% [markdown]
# **Lectura.** Las cinco figuras dicen lo mismo que sus versiones impresas, con menos detalle y más letra. La interacción de H3 deja la brecha de cada estrato encima de su columna; el RMSE de prueba ordena los nueve modelos de regresión con el eje recortado, porque las diferencias son de pocos puntos; las curvas ROC y PR no llevan los recuadros ampliados de §8.1, que no cabrían, pero conservan el umbral de cada modelo, y los perfiles casi paralelos de los dos clústeres, junto con el reparto, muestran que el clúster alto es una franja de nivel con más estudiantes de colegios no oficiales, urbanos y de estratos medios. La curva de pérdida del MLP es la misma de la fase 3, leída de su caché: la validación toca su mínimo en la época marcada y se queda en la línea de XGBoost, que es lo que la diapositiva cuenta.

# %% [markdown]
# ## 10. Cifras y comprobaciones
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
