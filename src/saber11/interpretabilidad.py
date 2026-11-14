"""Interpretabilidad y errores de la fase 3: SHAP, permutación y errores por grupo.

SHAP y la permutación corren sobre una muestra estratificada de validación, nunca sobre la
prueba: describen el modelo, no estiman su error. Los dos se expresan por variable
original: las contribuciones SHAP de las columnas codificadas (los niveles del one-hot y el
indicador de faltante) se suman en su variable, y la permutación baraja la columna de
entrada antes del `Pipeline`, así que mueve a la vez la variable y su indicador. Las medias
por categoría y los errores por grupo solo dejan los grupos con 10 o más.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

from saber11 import modelos
from saber11 import variables as v

with warnings.catch_warnings():
    # shap importa `tqdm.auto`, que en Jupyter avisa si falta ipywidgets; aquí no se usa ninguna
    # barra de progreso, así que el aviso solo ensuciaría la salida de error del cuaderno.
    warnings.filterwarnings("ignore", message="IProgress not found")
    import shap

SEMILLA = 42
REPETICIONES_PERMUTACION = 5
MINIMO_POR_GRUPO = 10


def contribuciones_shap(modelo: BaseEstimator, X_codificado: pd.DataFrame) -> pd.DataFrame:
    """Contribuciones SHAP de un modelo de árboles de regresión, una columna por columna codificada."""
    valores = shap.TreeExplainer(modelo).shap_values(X_codificado)
    return pd.DataFrame(valores, index=X_codificado.index, columns=X_codificado.columns)


def por_variable(contribuciones: pd.DataFrame, predictores: tuple[str, ...]) -> pd.DataFrame:
    """Suma, fila por fila, las contribuciones de las columnas codificadas de cada variable original."""
    origen = [modelos.variable_de_origen(c, predictores) for c in contribuciones.columns]
    return contribuciones.T.groupby(origen, sort=False).sum().T


def importancia_media(por_variable: pd.DataFrame) -> pd.Series:
    """Media del valor absoluto de la contribución de cada variable, de mayor a menor."""
    return por_variable.abs().mean().sort_values(ascending=False).rename("importancia")


def permutacion(
    modelo: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    puntuacion: str,
    repeticiones: int = REPETICIONES_PERMUTACION,
    semilla: int = SEMILLA,
) -> pd.DataFrame:
    """Caída media de la puntuación al barajar cada columna de entrada del `Pipeline`, de mayor a menor."""
    resultado = permutation_importance(modelo, X, y, scoring=puntuacion, n_repeats=repeticiones, random_state=semilla)
    tabla = pd.DataFrame({"importancia": resultado.importances_mean, "desviacion": resultado.importances_std}, index=X.columns)
    return tabla.sort_values("importancia", ascending=False)


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Correlación de Spearman entre dos importancias de las mismas variables."""
    return float(stats.spearmanr(a, b.reindex(a.index)).statistic)


def media_por_categoria(contribucion: pd.Series, categoria: pd.Series, minimo: int = MINIMO_POR_GRUPO) -> pd.DataFrame:
    """Contribución media por categoría, solo para las categorías con `minimo` o más filas."""
    tabla = contribucion.groupby(categoria.to_numpy()).agg(n="size", media="mean")
    return tabla.loc[tabla["n"] >= minimo]


def errores_regresion(y: pd.Series, prediccion: np.ndarray, grupo: pd.Series, minimo: int = MINIMO_POR_GRUPO) -> pd.DataFrame:
    """Residuo medio (observado menos predicho), RMSE y MAE por grupo."""
    residuo = pd.Series(np.asarray(y, dtype=float) - np.asarray(prediccion, dtype=float))
    tabla = residuo.groupby(grupo.to_numpy()).agg(n="size", residuo_medio="mean", rmse=lambda r: float(np.sqrt(np.mean(r**2))), mae=lambda r: float(np.mean(np.abs(r))))
    return tabla.loc[tabla["n"] >= minimo]


def errores_clasificacion(y: pd.Series, prediccion: np.ndarray, grupo: pd.Series, minimo: int = MINIMO_POR_GRUPO) -> pd.DataFrame:
    """Tasa de error y, dentro de cada clase real, la proporción mal clasificada, por grupo.

    `falsos_negativos` es la proporción de "NO OFICIAL" predichos como "OFICIAL" y
    `falsos_positivos`, la de "OFICIAL" predichos como "NO OFICIAL"; NaN si el grupo no
    tiene esa clase.
    """
    positiva = np.asarray(y) == v.CLASE_POSITIVA
    predicha = np.asarray(prediccion) == v.CLASE_POSITIVA
    datos = pd.DataFrame({"error": positiva != predicha, "positiva": positiva, "fn": positiva & ~predicha, "fp": ~positiva & predicha})
    grupos = datos.groupby(grupo.to_numpy())
    tabla = pd.DataFrame({
        "n": grupos.size(),
        "error": grupos["error"].mean(),
        "falsos_negativos": grupos["fn"].sum() / grupos["positiva"].sum().replace(0, np.nan),
        "falsos_positivos": grupos["fp"].sum() / (grupos.size() - grupos["positiva"].sum()).replace(0, np.nan),
    })
    return tabla.loc[tabla["n"] >= minimo]


def mas_importantes(importancia: pd.Series, k: int) -> tuple[str, ...]:
    """Las `k` variables de mayor importancia, de mayor a menor."""
    return tuple(importancia.sort_values(ascending=False).index[:k])


def r2_con_las_mas_importantes(
    ajustar: Callable[[tuple[str, ...]], BaseEstimator],
    importancia: pd.Series,
    k: int,
    validacion: tuple[pd.DataFrame, pd.Series],
) -> float:
    """R² de validación del modelo que `ajustar` entrena solo con las `k` variables más importantes.

    `ajustar` recibe los predictores y devuelve el modelo ya entrenado, listo para predecir
    con las columnas de entrada completas: así quien llama decide dónde entrena y dónde
    predice (en la GPU, XGBoost entrena allí y predice en la CPU).
    """
    modelo = ajustar(mas_importantes(importancia, k))
    return float(r2_score(validacion[1], modelo.predict(validacion[0])))
