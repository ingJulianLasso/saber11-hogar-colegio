"""Reglas fijas de limpieza del recorte de modelado: exclusiones, armonización y edad.

Ninguna regla aprende de los datos (no usa medias, modas ni cuantiles), así que se aplican
antes de partir sin filtrar información de validación ni de prueba.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

FRASE_RURAL = "Vive en una zona rural donde no hay estratificación socioeconómica"
SIN_ESTRATO = "Sin Estrato"
# FAMI_CUARTOSHOGAR viene en palabras y FAMI_PERSONASHOGAR en rangos en el 20224. Los rangos
# pasan a su punto medio, la lectura neutral de un intervalo; el abierto "9 o más", a 9.
CUARTOS = {"Uno": 1.0, "Dos": 2.0, "Tres": 3.0, "Cuatro": 4.0, "Cinco": 5.0, "Seis o mas": 6.0}
PERSONAS = {"1 a 2": 1.5, "3 a 4": 3.5, "5 a 6": 5.5, "7 a 8": 7.5, "9 o más": 9.0}
# La guía de orientación 2022-2 no trae la fecha de aplicación, así que la edad es la
# cumplida al cierre del año del examen.
FECHA_REFERENCIA = date(2022, 12, 31)
ANIO_MINIMO, ANIO_MAXIMO = 1950, 2012
EDAD = "ESTU_EDAD"
COLEGIO_BASICO = ("COLE_NATURALEZA", "COLE_AREA_UBICACION", "COLE_JORNADA", "COLE_CALENDARIO", "COLE_GENERO")


@dataclass(frozen=True)
class Exclusion:
    motivo: str
    filas: int


def excluir(df: pd.DataFrame) -> tuple[pd.DataFrame, list[Exclusion]]:
    """Quita las filas sin datos del colegio y las de puntaje global en cero, con su conteo.

    El índice del resultado es la posición en el recorte ya sin exclusiones, que es la que
    usa la partición.
    """
    sin_colegio = df[list(COLEGIO_BASICO)].isna().all(axis=1)
    global_cero = df["PUNT_GLOBAL"].eq(0) & ~sin_colegio
    exclusiones = [
        Exclusion("Sin datos del colegio (naturaleza, zona, jornada, calendario ni género)", int(sin_colegio.sum())),
        Exclusion("Puntaje global en cero, indistinguible de una prueba en blanco", int(global_cero.sum())),
    ]
    return df.loc[~(sin_colegio | global_cero)].reset_index(drop=True), exclusiones


def fecha_nacimiento(texto: pd.Series) -> pd.Series:
    """Fecha `dd/mm/aaaa`; las ilegibles y las de año fuera de 1950 a 2012 quedan como NaT."""
    fechas = pd.to_datetime(texto, format="%d/%m/%Y", errors="coerce")
    return fechas.where(fechas.dt.year.between(ANIO_MINIMO, ANIO_MAXIMO))


def edad_cumplida(fechas: pd.Series, referencia: date = FECHA_REFERENCIA) -> pd.Series:
    """Años cumplidos a la fecha de referencia; NaN donde la fecha es NaT."""
    ya_cumplio = (fechas.dt.month < referencia.month) | ((fechas.dt.month == referencia.month) & (fechas.dt.day <= referencia.day))
    edad = referencia.year - fechas.dt.year - np.where(ya_cumplio, 0, 1)
    return edad.where(fechas.notna()).astype(float)


def _a_numero(serie: pd.Series, equivalencias: dict[str, float]) -> pd.Series:
    desconocidos = set(serie.dropna().unique()) - set(equivalencias)
    if desconocidos:
        raise ValueError(f"{serie.name}: valores sin equivalencia {sorted(desconocidos)}")
    return serie.map(equivalencias).astype(float)


def armonizar(df: pd.DataFrame) -> pd.DataFrame:
    """Cuartos y personas a número, la frase rural del estrato a "Sin Estrato" y la edad.

    La fecha de nacimiento se reemplaza por `ESTU_EDAD`.
    """
    armonizado = df.drop(columns="ESTU_FECHANACIMIENTO").assign(
        FAMI_ESTRATOVIVIENDA=df["FAMI_ESTRATOVIVIENDA"].replace(FRASE_RURAL, SIN_ESTRATO),
        FAMI_CUARTOSHOGAR=_a_numero(df["FAMI_CUARTOSHOGAR"], CUARTOS),
        FAMI_PERSONASHOGAR=_a_numero(df["FAMI_PERSONASHOGAR"], PERSONAS),
    )
    armonizado[EDAD] = edad_cumplida(fecha_nacimiento(df["ESTU_FECHANACIMIENTO"]))
    return armonizado


def limpiar(df: pd.DataFrame) -> tuple[pd.DataFrame, list[Exclusion]]:
    """Exclusiones y armonización, en ese orden."""
    sin_excluidas, exclusiones = excluir(df)
    return armonizar(sin_excluidas), exclusiones
