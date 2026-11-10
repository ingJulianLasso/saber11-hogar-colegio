"""Variables del modelado: predictores por tarea, niveles, imputación, derivadas y atípicos.

Los tres transformadores (`Imputador`, `Derivadas` y `Atipicos`) reciben y devuelven
`DataFrame` y se ajustan solo con lo que les pase el `Pipeline`, que en el proyecto es
siempre el conjunto de entrenamiento.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (habilita IterativeImputer)
from sklearn.impute import IterativeImputer, SimpleImputer

from saber11.limpieza import CUARTOS, EDAD, PERSONAS

SEMILLA = 42
# Converge en 11 rondas sobre el entrenamiento del 20224; el margen evita cortarla antes.
MAXIMO_ITERACIONES = 25
# Un indicador con menos faltantes daría un coeficiente estimado con un puñado de estudiantes
# (con uno solo, el modelo reproduce su puntaje); es el mismo mínimo del análisis del mecanismo.
MINIMO_FALTANTES_INDICADOR = 30
OBJETIVO_REGRESION = "PUNT_GLOBAL"
OBJETIVO_CLASIFICACION = "COLE_NATURALEZA"
CLASE_POSITIVA = "NO OFICIAL"
# PUNT_GLOBAL = round(5 × (3 × (lectura + matemáticas + naturales + sociales) + inglés) / 13):
# los puntajes por área, el de inglés y su nivel de desempeño reconstruyen el objetivo.
PROHIBIDOS_REGRESION = frozenset({
    "PUNT_LECTURA_CRITICA",
    "PUNT_MATEMATICAS",
    "PUNT_C_NATURALES",
    "PUNT_SOCIALES_CIUDADANAS",
    "PUNT_INGLES",
    "DESEMP_INGLES",
})

ESTRATO = "FAMI_ESTRATOVIVIENDA"
EDUCACION_MADRE, EDUCACION_PADRE = "FAMI_EDUCACIONMADRE", "FAMI_EDUCACIONPADRE"
BIENES = ("FAMI_TIENEINTERNET", "FAMI_TIENECOMPUTADOR", "FAMI_TIENEAUTOMOVIL", "FAMI_TIENELAVADORA")
CUARTOS_HOGAR, PERSONAS_HOGAR = "FAMI_CUARTOSHOGAR", "FAMI_PERSONASHOGAR"
DEPARTAMENTO = "COLE_DEPTO_UBICACION"
INDICE_BIENES = "INDICE_BIENES"
EDUCACION_MAXIMA = "EDUCACION_MAXIMA_PADRES"
HACINAMIENTO = "HACINAMIENTO"
PREFIJO_FALTA = "FALTA_"

ESTRATOS = tuple(f"Estrato {n}" for n in range(1, 7))
EDUCACION = (
    "Ninguno",
    "Primaria incompleta",
    "Primaria completa",
    "Secundaria (Bachillerato) incompleta",
    "Secundaria (Bachillerato) completa",
    "Técnica o tecnológica incompleta",
    "Técnica o tecnológica completa",
    "Educación profesional incompleta",
    "Educación profesional completa",
    "Postgrado",
)
EDUCACION_DESCONOCIDA = "No sabe o no aplica"
# Escalas ordinales que la imputación iterativa trata como números. "Sin Estrato", "No sabe"
# y "No Aplica" son respuestas válidas fuera de la escala: no se imputan ni se reordenan.
ESCALAS: dict[str, tuple[str, ...]] = {
    ESTRATO: ESTRATOS,
    EDUCACION_MADRE: EDUCACION,
    EDUCACION_PADRE: EDUCACION,
    **{bien: ("No", "Si") for bien in BIENES},
    "COLE_BILINGUE": ("N", "S"),
}
# Valores posibles de las numéricas discretas, para devolver la imputación a un valor real.
VALORES_POSIBLES: dict[str, tuple[float, ...]] = {
    CUARTOS_HOGAR: tuple(sorted(CUARTOS.values())),
    PERSONAS_HOGAR: tuple(sorted(PERSONAS.values())),
}
NUMERICAS = (CUARTOS_HOGAR, PERSONAS_HOGAR, EDAD, INDICE_BIENES, HACINAMIENTO)

HOGAR = (ESTRATO, EDUCACION_MADRE, EDUCACION_PADRE, *BIENES, CUARTOS_HOGAR, PERSONAS_HOGAR)
COLEGIO = (
    "COLE_NATURALEZA",
    "COLE_AREA_UBICACION",
    "COLE_JORNADA",
    "COLE_CALENDARIO",
    "COLE_CARACTER",
    "COLE_BILINGUE",
    "COLE_GENERO",
    DEPARTAMENTO,
)
ESTUDIANTE = ("ESTU_GENERO", EDAD)
# Columnas que entran al Pipeline, antes de imputar y derivar.
ENTRADA_REGRESION = (*HOGAR, *COLEGIO, *ESTUDIANTE)
ENTRADA_CLASIFICACION = (*HOGAR, *ESTUDIANTE)
DERIVADAS = (INDICE_BIENES, EDUCACION_MAXIMA, HACINAMIENTO)
# Predictores del modelo por especificación. El índice de bienes es la suma exacta de los
# cuatro bienes, así que las dos versiones no pueden ir juntas.
PREDICTORES_REGRESION = {
    "componentes": ENTRADA_REGRESION,
    "derivadas": (ESTRATO, INDICE_BIENES, EDUCACION_MAXIMA, HACINAMIENTO, *COLEGIO, *ESTUDIANTE),
}
PREDICTORES_CLASIFICACION = ENTRADA_CLASIFICACION

# Niveles de referencia: el más bajo en las ordenadas, el más frecuente en las nominales y
# Bogotá en el departamento. El cuaderno comprueba que las nominales sigan siendo las más
# frecuentes en entrenamiento.
REFERENCIAS = {
    ESTRATO: "Estrato 1",
    EDUCACION_MADRE: "Ninguno",
    EDUCACION_PADRE: "Ninguno",
    EDUCACION_MAXIMA: "Ninguno",
    **{bien: "No" for bien in BIENES},
    "COLE_NATURALEZA": "OFICIAL",
    "COLE_AREA_UBICACION": "URBANO",
    "COLE_JORNADA": "MAÑANA",
    "COLE_CALENDARIO": "A",
    "COLE_CARACTER": "ACADÉMICO",
    "COLE_BILINGUE": "N",
    "COLE_GENERO": "MIXTO",
    DEPARTAMENTO: "BOGOTÁ",
    "ESTU_GENERO": "F",
}
REFERENCIAS_ORDENADAS = (ESTRATO, EDUCACION_MADRE, EDUCACION_PADRE, EDUCACION_MAXIMA, *BIENES)


def codificar_escala(serie: pd.Series, escala: tuple[str, ...]) -> pd.Series:
    """Posición en la escala ordinal; NaN para faltantes y para respuestas fuera de la escala."""
    return serie.map({nivel: float(i) for i, nivel in enumerate(escala)}).astype(float)


def _valor_posible_mas_cercano(valores: np.ndarray, posibles: tuple[float, ...]) -> np.ndarray:
    rejilla = np.asarray(posibles)
    return rejilla[np.abs(valores[:, None] - rejilla[None, :]).argmin(axis=1)]


class Imputador(TransformerMixin, BaseEstimator):
    """Imputación por moda o iterativa, con un indicador `FALTA_<columna>` por columna con faltantes.

    - `moda`: `SimpleImputer(strategy="most_frequent")` sobre todas las columnas.
    - `iterativa`: `IterativeImputer` sobre las escalas ordinales y las numéricas codificadas
      como números; el resultado se redondea al valor posible más cercano dentro del rango
      visto en entrenamiento. Las nominales restantes se imputan con la moda.

    Los indicadores salen de los faltantes reales de la entrada (los de `add_indicator=True`
    marcarían también las respuestas fuera de escala que la iterativa trata como ausentes), y
    solo para columnas con al menos `MINIMO_FALTANTES_INDICADOR` faltantes en entrenamiento.
    """

    def __init__(self, estrategia: str = "moda", semilla: int = SEMILLA) -> None:
        self.estrategia = estrategia
        self.semilla = semilla

    def fit(self, X: pd.DataFrame, y: object = None) -> Imputador:
        if self.estrategia not in {"moda", "iterativa"}:
            raise ValueError(f"Estrategia de imputación desconocida: {self.estrategia}")
        self.columnas_ = list(X.columns)
        self.con_faltantes_ = [c for c in X.columns if X[c].isna().any()]
        self.indicadores_ = [c for c in self.con_faltantes_ if X[c].isna().sum() >= MINIMO_FALTANTES_INDICADOR]
        self.moda_ = SimpleImputer(strategy="most_frequent").fit(X.astype(object))
        if self.estrategia == "iterativa":
            codigos = self._codigos(X)
            self.rangos_ = {c: (codigos[c].min(), codigos[c].max()) for c in codigos.columns}
            self.iterativo_ = IterativeImputer(random_state=self.semilla, max_iter=MAXIMO_ITERACIONES).fit(codigos)
        return self

    def _codigos(self, X: pd.DataFrame) -> pd.DataFrame:
        columnas = {c: codificar_escala(X[c], ESCALAS[c]) for c in X.columns if c in ESCALAS}
        columnas |= {c: X[c].astype(float) for c in X.columns if c in (CUARTOS_HOGAR, PERSONAS_HOGAR, EDAD)}
        return pd.DataFrame(columnas, index=X.index)

    def _redondear(self, columna: str, valores: np.ndarray) -> np.ndarray:
        minimo, maximo = self.rangos_[columna]
        recortados = np.clip(valores, minimo, maximo)
        if columna in VALORES_POSIBLES:
            posibles = tuple(v for v in VALORES_POSIBLES[columna] if minimo <= v <= maximo)
            return _valor_posible_mas_cercano(recortados, posibles)
        return np.round(recortados)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        faltantes = X[self.con_faltantes_].isna()
        moda = pd.DataFrame(self.moda_.transform(X.astype(object)), columns=self.columnas_, index=X.index)
        salida = X.copy()
        for columna in self.con_faltantes_:
            salida[columna] = X[columna].where(~faltantes[columna], moda[columna]).astype(X[columna].dtype)
        if self.estrategia == "iterativa":
            codigos = self._codigos(X)
            estimados = pd.DataFrame(self.iterativo_.transform(codigos), columns=codigos.columns, index=X.index)
            for columna in codigos.columns:
                mascara = X[columna].isna().to_numpy()
                if not mascara.any():
                    continue
                valores = self._redondear(columna, estimados.loc[mascara, columna].to_numpy())
                if columna in ESCALAS:
                    salida.loc[mascara, columna] = [ESCALAS[columna][int(v)] for v in valores]
                else:
                    salida.loc[mascara, columna] = valores
        indicadores = faltantes[self.indicadores_].astype(np.int8).add_prefix(PREFIJO_FALTA)
        return pd.concat([salida, indicadores], axis=1)


def agregar_derivadas(X: pd.DataFrame) -> pd.DataFrame:
    """Índice de bienes (0 a 4), máxima educación de los padres y hacinamiento.

    Espera datos ya imputados. La máxima educación toma el nivel más alto de la escala entre
    los dos padres; si ninguno tiene un nivel en la escala, vale "No sabe o no aplica". El
    hacinamiento son personas por cuarto, el indicador que el DANE usa para el hacinamiento
    crítico (más de tres personas por cuarto).
    """
    codigos = pd.concat([codificar_escala(X[c], EDUCACION) for c in (EDUCACION_MADRE, EDUCACION_PADRE)], axis=1)
    maximo = codigos.max(axis=1)
    educacion_maxima = maximo.map(lambda codigo: EDUCACION_DESCONOCIDA if np.isnan(codigo) else EDUCACION[int(codigo)])
    return X.assign(**{
        INDICE_BIENES: sum(X[bien].eq("Si").astype(float) for bien in BIENES),
        EDUCACION_MAXIMA: educacion_maxima.astype(X[EDUCACION_MADRE].dtype),
        HACINAMIENTO: X[PERSONAS_HOGAR] / X[CUARTOS_HOGAR],
    })


class Derivadas(TransformerMixin, BaseEstimator):
    """Agrega las tres derivadas del hogar; no aprende nada de los datos."""

    def fit(self, X: pd.DataFrame, y: object = None) -> Derivadas:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return agregar_derivadas(X)


class Atipicos(TransformerMixin, BaseEstimator):
    """Tratamiento por variable: `conservar`, `winsorizar` (tope en el percentil de entrenamiento) o `log`."""

    def __init__(self, tratamiento: Mapping[str, str] | None = None, percentil: float = 99) -> None:
        self.tratamiento = tratamiento
        self.percentil = percentil

    def fit(self, X: pd.DataFrame, y: object = None) -> Atipicos:
        tratamiento = dict(self.tratamiento or {})
        desconocidos = set(tratamiento.values()) - {"conservar", "winsorizar", "log"}
        if desconocidos:
            raise ValueError(f"Tratamientos de atípicos desconocidos: {sorted(desconocidos)}")
        self.topes_ = {c: float(np.nanpercentile(X[c], self.percentil)) for c, t in tratamiento.items() if t == "winsorizar"}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        salida = X.copy()
        for columna, tratamiento in (self.tratamiento or {}).items():
            if tratamiento == "winsorizar":
                salida[columna] = salida[columna].clip(upper=self.topes_[columna])
            elif tratamiento == "log":
                salida[columna] = np.log(salida[columna])
        return salida


def es_indicador(columna: str) -> bool:
    return columna.startswith(PREFIJO_FALTA)


class Seleccion(TransformerMixin, BaseEstimator):
    """Deja los predictores de la especificación y, si se pide, los indicadores de faltantes."""

    def __init__(self, predictores: tuple[str, ...] = (), indicadores: bool = True) -> None:
        self.predictores = predictores
        self.indicadores = indicadores

    def fit(self, X: pd.DataFrame, y: object = None) -> Seleccion:
        indicadores = [c for c in X.columns if es_indicador(c)] if self.indicadores else []
        self.columnas_ = [*self.predictores, *indicadores]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.columnas_]

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.asarray(self.columnas_, dtype=object)

