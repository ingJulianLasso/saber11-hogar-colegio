"""Pipelines, modelos base, métricas y registro de la evaluación única en prueba.

Las tres constantes de decisión (`IMPUTACION`, `ESPECIFICACION` y `ATIPICOS`) las fija el
cuaderno de la fase 2 comparando alternativas en validación; una comprobación del cuaderno
falla si dejan de coincidir con la ganadora. `saber11.intermedios` las lee para escribir lo
que replica R, y la fase 3 las hereda.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from saber11 import particion
from saber11 import variables as v

IMPUTACION = "moda"
ESPECIFICACION = "componentes"
ATIPICOS = {v.EDAD: "winsorizar", v.HACINAMIENTO: "conservar"}

TERMINO = re.compile(r"C\((\w+), Treatment\('(.*)'\)\)\[T\.(.*)\]")


def preparacion(imputacion: str = IMPUTACION, atipicos: dict[str, str] | None = None) -> Pipeline:
    """Imputación, derivadas y atípicos, sin codificar ni escalar.

    Con `imputacion="lista"` no imputa: se usa sobre filas completas (`filas_completas`).
    """
    pasos = [] if imputacion == "lista" else [("imputacion", v.Imputador(imputacion))]
    pasos += [("derivadas", v.Derivadas()), ("atipicos", v.Atipicos(ATIPICOS if atipicos is None else atipicos))]
    return Pipeline(pasos)


def codificador(predictores: tuple[str, ...], departamento: str = "onehot") -> ColumnTransformer:
    """One-hot con referencias explícitas, `StandardScaler` en las numéricas e indicadores sin tocar.

    Con `departamento="objetivo"`, el departamento va con `TargetEncoder`, que ajusta con
    validación cruzada interna para no usar el objetivo de la misma fila que codifica.
    """
    numericas = [c for c in predictores if c in v.NUMERICAS]
    categoricas = [c for c in predictores if c in v.REFERENCIAS and not (departamento == "objetivo" and c == v.DEPARTAMENTO)]
    transformadores = [
        ("numericas", StandardScaler(), numericas),
        (
            "categoricas",
            OneHotEncoder(drop=[v.REFERENCIAS[c] for c in categoricas], handle_unknown="ignore", sparse_output=False),
            categoricas,
        ),
    ]
    if departamento == "objetivo":
        objetivo = TargetEncoder(target_type="continuous", cv=KFold(n_splits=5, shuffle=True, random_state=v.SEMILLA))
        transformadores.append(("departamento", make_pipeline(objetivo, StandardScaler()), [v.DEPARTAMENTO]))
    transformador = ColumnTransformer(transformadores, remainder="passthrough", verbose_feature_names_out=False)
    return transformador.set_output(transform="pandas")


def pipeline(
    predictores: tuple[str, ...],
    modelo: BaseEstimator,
    imputacion: str = IMPUTACION,
    atipicos: dict[str, str] | None = None,
    departamento: str = "onehot",
) -> Pipeline:
    """Preparación, selección de predictores, codificación y escalado, y el modelo."""
    return Pipeline([
        *preparacion(imputacion, atipicos).steps,
        ("seleccion", v.Seleccion(predictores)),
        ("codificacion", codificador(predictores, departamento)),
        ("modelo", modelo),
    ])


def lineal_simple(predictor: str, imputacion: str = IMPUTACION, atipicos: dict[str, str] | None = None) -> Pipeline:
    """Regresión lineal con un solo predictor numérico, sin indicadores ni escalado."""
    return Pipeline([
        *preparacion(imputacion, atipicos).steps,
        ("seleccion", v.Seleccion((predictor,), indicadores=False)),
        ("modelo", LinearRegression()),
    ])


def regresion_multiple(especificacion: str = ESPECIFICACION, **opciones: object) -> Pipeline:
    return pipeline(v.PREDICTORES_REGRESION[especificacion], LinearRegression(), **opciones)


def logistica_base(**opciones: object) -> Pipeline:
    """Logística con pesos de clase inversos a su frecuencia (77/23), sin remuestreo."""
    return pipeline(v.PREDICTORES_CLASIFICACION, LogisticRegression(class_weight="balanced", max_iter=2000), **opciones)


def dummy_regresion() -> DummyRegressor:
    return DummyRegressor(strategy="mean")


def dummy_clasificacion() -> DummyClassifier:
    return DummyClassifier(strategy="prior")


def ajustar(modelo: BaseEstimator, X: pd.DataFrame, y: pd.Series, conjuntos: pd.Series) -> BaseEstimator:
    """Ajusta el modelo (o el Pipeline completo) solo con las filas de entrenamiento."""
    entrenamiento = conjuntos.eq(particion.ENTRENAMIENTO).to_numpy()
    return modelo.fit(X.loc[entrenamiento], y.loc[entrenamiento])


def filas_completas(X: pd.DataFrame) -> pd.Series:
    """Máscara de filas sin faltantes, para la eliminación por lista."""
    return X.notna().all(axis=1)


def predictor_simple(entrenamiento: pd.DataFrame, objetivo: pd.Series) -> tuple[str, float]:
    """Numérica de mayor |Spearman| con el objetivo, entre las definidas para todo estudiante."""
    correlaciones = {c: stats.spearmanr(entrenamiento[c], objetivo).statistic for c in v.NUMERICAS}
    ganadora = max(correlaciones, key=lambda c: abs(correlaciones[c]))
    return ganadora, float(correlaciones[ganadora])


def metricas_regresion(y: pd.Series, prediccion: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y, prediccion)),
        "rmse": float(root_mean_squared_error(y, prediccion)),
        "mae": float(mean_absolute_error(y, prediccion)),
    }


def metricas_clasificacion(y: pd.Series, prediccion: np.ndarray, probabilidad: np.ndarray) -> dict[str, float]:
    """Exactitud, precisión, exhaustividad, F1 y AUC-ROC de la clase "NO OFICIAL"."""
    positiva = v.CLASE_POSITIVA
    return {
        "accuracy": float(accuracy_score(y, prediccion)),
        "precision": float(precision_score(y, prediccion, pos_label=positiva, zero_division=0)),
        "recall": float(recall_score(y, prediccion, pos_label=positiva)),
        "f1": float(f1_score(y, prediccion, pos_label=positiva)),
        "auc": float(roc_auc_score(y.eq(positiva), probabilidad)),
    }


def iniciar_registro_prueba(ruta: Path) -> None:
    """Deja vacío el registro de evaluaciones en prueba al comienzo de cada ejecución."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text("[]\n", encoding="utf-8")


def evaluar_en_prueba(ruta: Path, modelo: str, metricas: dict[str, float]) -> None:
    """Agrega la evaluación de un modelo en prueba; falla si ese modelo ya se evaluó."""
    registro = json.loads(ruta.read_text(encoding="utf-8"))
    if any(fila["modelo"] == modelo for fila in registro):
        raise ValueError(f"El modelo {modelo!r} ya se evaluó en prueba: la prueba se usa una sola vez")
    registro.append({"modelo": modelo, "metricas": metricas})
    ruta.write_text(json.dumps(registro, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bloques_de_columnas(columnas: list[str], predictores: tuple[str, ...]) -> dict[str, list[str]]:
    """Agrupa las columnas codificadas por variable de origen; los indicadores de faltante van en un bloque."""
    bloques: dict[str, list[str]] = {}
    for columna in columnas:
        if v.es_indicador(columna):
            origen = "Indicadores de faltante"
        else:
            origen = next(p for p in sorted(predictores, key=len, reverse=True) if columna == p or columna.startswith(f"{p}_"))
        bloques.setdefault(origen, []).append(columna)
    return bloques


def vif_generalizado(X: pd.DataFrame, predictores: tuple[str, ...]) -> pd.DataFrame:
    """VIF generalizado de Fox y Monette (1992) por variable, con GVIF^(1/(2gl)) comparable al VIF.

    GVIF = det(R11) det(R22) / det(R), donde R es la matriz de correlación de las columnas
    codificadas, R11 la del bloque de la variable y R22 la del resto. Con un solo grado de
    libertad coincide con el VIF clásico.
    """
    correlacion = np.corrcoef(X.to_numpy(dtype=float), rowvar=False)
    _, log_total = np.linalg.slogdet(correlacion)
    posicion = {columna: i for i, columna in enumerate(X.columns)}
    filas = []
    for variable, columnas in bloques_de_columnas(list(X.columns), predictores).items():
        dentro = [posicion[c] for c in columnas]
        fuera = [i for i in range(len(X.columns)) if i not in set(dentro)]
        log_gvif = np.linalg.slogdet(correlacion[np.ix_(dentro, dentro)])[1] + np.linalg.slogdet(correlacion[np.ix_(fuera, fuera)])[1] - log_total
        gvif = float(np.exp(log_gvif))
        filas.append({"variable": variable, "gl": len(columnas), "gvif": gvif, "gvif_ajustado": gvif ** (1 / (2 * len(columnas)))})
    return pd.DataFrame(filas).set_index("variable").sort_values("gvif_ajustado", ascending=False)


def formula(objetivo: str, predictores: tuple[str, ...], indicadores: list[str]) -> str:
    """Fórmula de statsmodels con las mismas referencias que el one-hot y que R."""
    terminos = [f"C({c}, Treatment('{v.REFERENCIAS[c]}'))" if c in v.REFERENCIAS else c for c in predictores]
    return f"{objetivo} ~ " + " + ".join([*terminos, *indicadores])


def clave_termino(nombre: str) -> str:
    """Nombre común de un término: `VARIABLE=nivel` para las categóricas y el nombre para el resto.

    Entiende los nombres de statsmodels (`C(X, Treatment('a'))[T.b]`) y los del one-hot
    (`X_b`).
    """
    coincidencia = TERMINO.fullmatch(nombre)
    if coincidencia:
        return f"{coincidencia.group(1)}={coincidencia.group(3)}"
    for variable in sorted(v.REFERENCIAS, key=len, reverse=True):
        if nombre.startswith(f"{variable}_"):
            return f"{variable}={nombre[len(variable) + 1:]}"
    return nombre
