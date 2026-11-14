"""Comparación estadística de modelos: McNemar y t corregido de Nadeau y Bengio.

McNemar compara dos clasificadores sobre las mismas filas de prueba con sus aciertos
discordantes, con la corrección de continuidad que recomienda Dietterich (1998). El t de
Nadeau y Bengio (2003) compara dos modelos con validación cruzada repetida: como los
conjuntos de entrenamiento se solapan, la varianza de las diferencias se infla por
(1/J + n_prueba/n_entrenamiento) en lugar de 1/J.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import RepeatedKFold
from statsmodels.stats.contingency_tables import mcnemar as mcnemar_statsmodels

from saber11.hipotesis import SEMILLA, Contraste

REPETICIONES = 5
PLIEGUES = 5


def tabla_aciertos(y: np.ndarray, prediccion_a: np.ndarray, prediccion_b: np.ndarray) -> np.ndarray:
    """Tabla 2 × 2: filas, A acierta o falla; columnas, B acierta o falla."""
    y = np.asarray(y)
    acierta_a, acierta_b = np.asarray(prediccion_a) == y, np.asarray(prediccion_b) == y
    return np.array([
        [np.sum(acierta_a & acierta_b), np.sum(acierta_a & ~acierta_b)],
        [np.sum(~acierta_a & acierta_b), np.sum(~acierta_a & ~acierta_b)],
    ])


def mcnemar(y: np.ndarray, prediccion_a: np.ndarray, prediccion_b: np.ndarray) -> Contraste:
    """McNemar con corrección de continuidad: (|b - c| - 1)² / (b + c) con 1 grado de libertad."""
    tabla = tabla_aciertos(y, prediccion_a, prediccion_b)
    resultado = mcnemar_statsmodels(tabla, exact=False, correction=True)
    return Contraste(
        prueba="McNemar con corrección de continuidad",
        estadistico=float(resultado.statistic),
        gl=(1.0,),
        p=float(resultado.pvalue),
        n=int(tabla.sum()),
        detalle={"solo_acierta_a": int(tabla[0, 1]), "solo_acierta_b": int(tabla[1, 0])},
    )


def nadeau_bengio(diferencias: np.ndarray, n_entrenamiento: float, n_prueba: float) -> Contraste:
    """t corregido de Nadeau y Bengio sobre las J diferencias por pliegue, con J - 1 grados de libertad."""
    diferencias = np.asarray(diferencias, dtype=float)
    j = len(diferencias)
    error = np.sqrt((1 / j + n_prueba / n_entrenamiento) * diferencias.var(ddof=1))
    t = diferencias.mean() / error
    return Contraste(
        prueba="t corregido de Nadeau y Bengio",
        estadistico=float(t),
        gl=(float(j - 1),),
        p=float(2 * stats.t.sf(abs(t), j - 1)),
        n=j,
        detalle={"diferencia_media": float(diferencias.mean()), "error_estandar": float(error)},
    )


def diferencias_por_pliegue(
    modelo_a: BaseEstimator,
    modelo_b: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    metrica: Callable[[pd.Series, np.ndarray], float] = root_mean_squared_error,
    repeticiones: int = REPETICIONES,
    pliegues: int = PLIEGUES,
    semilla: int = SEMILLA,
) -> pd.DataFrame:
    """Métrica de A y de B en cada pliegue de una validación cruzada repetida, y su diferencia (A menos B)."""
    divisor = RepeatedKFold(n_splits=pliegues, n_repeats=repeticiones, random_state=semilla)
    filas = []
    for i, (entrenamiento, prueba) in enumerate(divisor.split(X)):
        valores = {}
        for nombre, modelo in (("a", modelo_a), ("b", modelo_b)):
            ajustado = clone(modelo).fit(X.iloc[entrenamiento], y.iloc[entrenamiento])
            valores[nombre] = float(metrica(y.iloc[prueba], ajustado.predict(X.iloc[prueba])))
        filas.append({
            "repeticion": i // pliegues + 1,
            "pliegue": i % pliegues + 1,
            "metrica_a": valores["a"],
            "metrica_b": valores["b"],
            "diferencia": valores["a"] - valores["b"],
            "n_entrenamiento": len(entrenamiento),
            "n_prueba": len(prueba),
        })
    return pd.DataFrame(filas)
