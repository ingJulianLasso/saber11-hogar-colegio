"""Datos sintéticos con la forma del recorte ya limpio, para las pruebas de preparación y modelos."""

import numpy as np
import pandas as pd
import pytest

from saber11 import variables as v

COLUMNAS_CON_FALTANTES = (v.ESTRATO, v.EDUCACION_MADRE, "FAMI_TIENEINTERNET", v.CUARTOS_HOGAR, v.EDAD, "COLE_BILINGUE")


def marco_limpio(filas: int = 4000, semilla: int = 0) -> pd.DataFrame:
    """Marco limpio sintético: todas las columnas de entrada, el puntaje y un 5 % de faltantes."""
    rng = np.random.default_rng(semilla)

    def elegir(valores: list, p: list[float] | None = None) -> pd.Series:
        return pd.Series(rng.choice(valores, size=filas, p=p), dtype="str")

    marco = pd.DataFrame({
        v.ESTRATO: elegir([*v.ESTRATOS, "Sin Estrato"]),
        v.EDUCACION_MADRE: elegir([*v.EDUCACION, "No sabe", "No Aplica"]),
        v.EDUCACION_PADRE: elegir([*v.EDUCACION, "No sabe", "No Aplica"]),
        **{bien: elegir(["Si", "No"]) for bien in v.BIENES},
        v.CUARTOS_HOGAR: rng.choice([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], size=filas),
        v.PERSONAS_HOGAR: rng.choice([1.5, 3.5, 5.5, 7.5, 9.0], size=filas),
        "COLE_NATURALEZA": elegir(["OFICIAL", "NO OFICIAL"], p=[0.77, 0.23]),
        "COLE_AREA_UBICACION": elegir(["URBANO", "RURAL"]),
        "COLE_JORNADA": elegir(["MAÑANA", "TARDE", "UNICA"]),
        "COLE_CALENDARIO": elegir(["A", "B"]),
        "COLE_CARACTER": elegir(["ACADÉMICO", "TÉCNICO"]),
        "COLE_BILINGUE": elegir(["N", "S"]),
        "COLE_GENERO": elegir(["MIXTO", "FEMENINO"]),
        v.DEPARTAMENTO: elegir(["BOGOTÁ", "ANTIOQUIA", "VALLE"]),
        "ESTU_GENERO": elegir(["F", "M"]),
        v.EDAD: rng.integers(15, 21, size=filas).astype(float),
    })
    estrato = v.codificar_escala(marco[v.ESTRATO], v.ESTRATOS).fillna(0)
    marco[v.OBJETIVO_REGRESION] = (200 + 8 * estrato + 15 * marco["FAMI_TIENEINTERNET"].eq("Si") + rng.normal(0, 30, filas)).round()
    for columna in COLUMNAS_CON_FALTANTES:
        marco.loc[rng.random(filas) < 0.05, columna] = np.nan
    return marco


@pytest.fixture
def limpio() -> pd.DataFrame:
    return marco_limpio()
