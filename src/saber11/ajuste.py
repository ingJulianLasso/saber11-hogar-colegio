"""Búsqueda de hiperparámetros de la fase 3: espacios, presupuesto y caché de cómputo.

Presupuesto: con la primera medición del tiempo de un ajuste, el cuaderno fija dos
constantes que no dependen del tiempo de cada corrida. `N_ITER` guarda, por modelo, el
mayor entero hasta 30 cuya búsqueda estimada (5 pliegues por el tiempo de un ajuste con la
configuración central del espacio, `configuracion_central`) cabe en 15 minutos
(`n_iter_que_cabe`).
`REAJUSTE_RANDOM_FOREST` dice si el reajuste de Random Forest con todo el entrenamiento cabe
en 15 minutos (`reajuste_cabe`); si no cabe, queda el modelo ajustado en la muestra.

Caché: `en_cache` envuelve una función de `saber11` con `joblib.Memory` en
`data/intermedio/cache/` (ignorada por git). La clave lleva los argumentos, que joblib
resume con un hash, y la versión del código: el SHA-256 del módulo de la función, de los
módulos de `saber11` que importa, directa o indirectamente, y de los de `saber11.modelos`,
con los que se arman los `Pipeline` que reciben las funciones guardadas (joblib resume el
objeto, no el código de sus clases). Si cambia el código, la función se vuelve a calcular
en una carpeta nueva.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from joblib import Memory
from scipy import stats
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold

from saber11.datos import DESTINO

SEMILLA = 42
PLIEGUES = 5
PRESUPUESTO_S = 15 * 60
MAXIMO_N_ITER = 30
CARPETA_CACHE = DESTINO / "intermedio" / "cache"
NUCLEO = "saber11.modelos"
PUNTUACION = {"regresion": "neg_root_mean_squared_error", "clasificacion": "roc_auc"}

# Fijadas con la primera medición del cuaderno de la fase 3, la de regresión el 10-nov-2026 (§10)
# y la de clasificación el 14-nov-2026 (§12), ya con solo las variables del hogar y del
# estudiante: un ajuste con la configuración central tardó 1,0 s (bosque de regresión, muestra),
# 5,6 s (XGBoost de regresión, entrenamiento), 0,6 s (bosque de clasificación, muestra), 2,6 s
# (XGBoost de clasificación, entrenamiento) y 8,5 s (SVM, muestra); los reajustes de los dos
# bosques elegidos se proyectaron en 0,4 y 0,2 minutos. Una comprobación del cuaderno las ata a
# la regla.
N_ITER: dict[str, int] = {
    "randomForestRegresion": 30,
    "xgboostRegresion": 30,
    "randomForestClasificacion": 30,
    "xgboostClasificacion": 30,
    "svmClasificacion": 21,
}
REAJUSTE_RANDOM_FOREST: bool | None = True

def _estable[D](distribucion: D) -> D:
    """La distribución con su propio generador fijo, para que la clave de la caché no cambie.

    scipy congela cada distribución con el generador global de numpy, y joblib lo incluye al
    resumir el espacio: sin esto, la clave de `buscar` cambiaría en cada ejecución. El
    muestreo no cambia, porque `RandomizedSearchCV` le pasa su propio generador a `rvs`.
    """
    distribucion.random_state = SEMILLA
    return distribucion


_BOSQUE = {
    "modelo__n_estimators": _estable(stats.randint(100, 501)),
    "modelo__max_depth": [None, 10, 20, 30],
    "modelo__min_samples_leaf": _estable(stats.randint(1, 21)),
    "modelo__max_features": ["sqrt", 0.33, 0.5, 1.0],
}
_XGBOOST = {
    "modelo__n_estimators": _estable(stats.randint(200, 1001)),
    "modelo__learning_rate": _estable(stats.loguniform(0.01, 0.3)),
    "modelo__max_depth": _estable(stats.randint(3, 11)),
    "modelo__min_child_weight": _estable(stats.loguniform(1, 20)),
    "modelo__subsample": _estable(stats.uniform(0.6, 0.4)),
    "modelo__colsample_bytree": _estable(stats.uniform(0.6, 0.4)),
    "modelo__reg_lambda": _estable(stats.loguniform(0.1, 10)),
}
# Espacios de `RandomizedSearchCV` sobre el paso "modelo" del `Pipeline` de `saber11.modelos`.
ESPACIOS = {
    "randomForestRegresion": _BOSQUE,
    "xgboostRegresion": _XGBOOST,
    "randomForestClasificacion": {**_BOSQUE, "modelo__criterion": ["gini", "entropy"]},
    "xgboostClasificacion": _XGBOOST,
    "svmClasificacion": {"modelo__C": _estable(stats.loguniform(0.1, 100)), "modelo__gamma": _estable(stats.loguniform(1e-4, 1))},
}


def configuracion_central(espacio: dict) -> dict:
    """La configuración con que se mide un ajuste: la mediana de cada distribución y el elemento del medio de cada lista.

    En las listas con un número par de elementos toma el segundo de los dos centrales. En
    las distribuciones discretas devuelve un entero.
    """
    central = {}
    for parametro, valores in espacio.items():
        if isinstance(valores, list):
            central[parametro] = valores[len(valores) // 2]
        else:
            mediana = valores.median()
            central[parametro] = int(mediana) if isinstance(valores.dist, stats.rv_discrete) else float(mediana)
    return central


def n_iter_que_cabe(segundos_por_ajuste: float, pliegues: int = PLIEGUES, presupuesto_s: float = PRESUPUESTO_S, maximo: int = MAXIMO_N_ITER) -> int:
    """Mayor número de configuraciones, hasta `maximo`, cuya búsqueda estimada cabe en el presupuesto.

    Devuelve 0 si ni una configuración cabe.
    """
    return min(maximo, math.floor(presupuesto_s / (pliegues * segundos_por_ajuste)))


def proyectar_segundos(segundos: float, filas_medidas: int, filas_objetivo: int) -> float:
    """Tiempo esperado con `filas_objetivo`, escalando el medido como n log n (el de construir un árbol)."""
    return segundos * (filas_objetivo * math.log(filas_objetivo)) / (filas_medidas * math.log(filas_medidas))


def reajuste_cabe(segundos_estimados: float, presupuesto_s: float = PRESUPUESTO_S) -> bool:
    return segundos_estimados <= presupuesto_s


class Ajustado(NamedTuple):
    """Un modelo ajustado y los segundos que tardó su ajuste."""

    modelo: BaseEstimator
    segundos: float


def ajustado(modelo: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> Ajustado:
    """Una copia sin entrenar del modelo, ajustada con `X` e `y`, con su tiempo.

    Con `en_cache` guarda el modelo ajustado y el tiempo de la primera vez que se ajustó.
    """
    copia = clone(modelo)
    inicio = time.perf_counter()
    copia.fit(X, y)
    return Ajustado(copia, time.perf_counter() - inicio)


def segundos_de_ajuste(modelo: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> float:
    """Segundos de un ajuste de una copia sin entrenar del modelo."""
    return ajustado(modelo, X, y).segundos


@dataclass(frozen=True)
class Busqueda:
    """Lo que deja una búsqueda: la mejor configuración, su puntuación media y la tabla completa."""

    mejores: dict
    puntuacion: float
    resultados: pd.DataFrame
    segundos: float


def buscar(
    modelo: BaseEstimator,
    espacio: dict,
    n_iter: int,
    X: pd.DataFrame,
    y: pd.Series,
    puntuacion: str,
    pliegues: int = PLIEGUES,
    semilla: int = SEMILLA,
    n_jobs: int | None = None,
) -> Busqueda:
    """`RandomizedSearchCV` con pliegues barajados (estratificados si es clasificador) y sin reajuste.

    No reajusta para que la caché guarde una tabla y no un modelo; el cuaderno ajusta la
    mejor configuración con `modelo.set_params(**busqueda.mejores)`.
    """
    divisor = (StratifiedKFold if is_classifier(modelo) else KFold)(n_splits=pliegues, shuffle=True, random_state=semilla)
    busqueda = RandomizedSearchCV(
        modelo, espacio, n_iter=n_iter, scoring=puntuacion, cv=divisor, refit=False, random_state=semilla, n_jobs=n_jobs
    )
    inicio = time.perf_counter()
    busqueda.fit(X, y)
    return Busqueda(dict(busqueda.best_params_), float(busqueda.best_score_), pd.DataFrame(busqueda.cv_results_), time.perf_counter() - inicio)


def _origen(modulo: str) -> Path:
    especificacion = importlib.util.find_spec(modulo)
    if especificacion is None or especificacion.origin is None:
        raise ValueError(f"No se encontró el módulo {modulo!r}")
    return Path(especificacion.origin)


def _importados(modulo: str) -> set[str]:
    """Módulos de `saber11` que importa `modulo`, leídos de su código."""
    importados = set()
    for nodo in ast.walk(ast.parse(_origen(modulo).read_text(encoding="utf-8"))):
        if isinstance(nodo, ast.ImportFrom) and nodo.module == "saber11":
            importados |= {f"saber11.{alias.name}" for alias in nodo.names}
        elif isinstance(nodo, ast.ImportFrom) and nodo.module and nodo.module.startswith("saber11."):
            importados.add(nodo.module)
        elif isinstance(nodo, ast.Import):
            importados |= {alias.name for alias in nodo.names if alias.name.startswith("saber11.")}
    return importados


def dependencias(modulo: str) -> tuple[str, ...]:
    """`modulo` y los módulos de `saber11` de los que depende, directa o indirectamente."""
    vistos: set[str] = set()
    pendientes = [modulo]
    while pendientes:
        actual = pendientes.pop()
        if actual not in vistos:
            vistos.add(actual)
            pendientes += _importados(actual) - vistos
    return tuple(sorted(vistos))


def version(*modulos: str) -> str:
    """SHA-256 (12 caracteres) del código de los `modulos` y de sus dependencias de `saber11`."""
    resumen = hashlib.sha256()
    for nombre in sorted(set().union(*(dependencias(m) for m in modulos))):
        resumen.update(nombre.encode() + b"\0" + _origen(nombre).read_bytes())
    return resumen.hexdigest()[:12]


def en_cache[F: Callable](funcion: F, carpeta: Path = CARPETA_CACHE) -> F:
    """La misma función, guardada en disco por argumentos y por versión del código de su módulo."""
    modulo = funcion.__module__
    if not modulo.startswith("saber11."):
        raise ValueError(f"Solo se guardan en caché funciones de saber11, no de {modulo!r}")
    return Memory(carpeta / f"{modulo.removeprefix('saber11.')}-{version(modulo, NUCLEO)}", verbose=0).cache(funcion)
