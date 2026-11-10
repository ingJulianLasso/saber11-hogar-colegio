"""Partición 70/15/15 estratificada por posición, su huella y la muestra para la fase 3.

Los recortes no traen un identificador del estudiante, así que la partición se fija por
posición en el recorte ya sin exclusiones (`saber11.limpieza.excluir`). La huella es el
SHA-256 de la secuencia de conjuntos: si otra máquina, con los mismos datos, obtiene la
misma huella, tiene exactamente la misma partición.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from saber11.datos import DESTINO

SEMILLA = 42
ENTRENAMIENTO, VALIDACION, PRUEBA = "entrenamiento", "validacion", "prueba"
CODIGO = {ENTRENAMIENTO: "E", VALIDACION: "V", PRUEBA: "P"}
PROPORCION_FUERA_DE_ENTRENAMIENTO = 0.30
RUTA_PARTICION = DESTINO / "intermedio" / "particion-20224.parquet"


def particionar(estrato: pd.Series, semilla: int = SEMILLA) -> pd.Series:
    """Conjunto de cada posición: 70 % entrenamiento, 15 % validación y 15 % prueba.

    Dos `train_test_split` estratificados por `estrato`: 70/30 y luego 50/50 del 30 %.
    """
    posiciones = np.arange(len(estrato))
    etiquetas = estrato.to_numpy()
    entrenamiento, resto = train_test_split(
        posiciones, test_size=PROPORCION_FUERA_DE_ENTRENAMIENTO, stratify=etiquetas, random_state=semilla
    )
    validacion, prueba = train_test_split(resto, test_size=0.5, stratify=etiquetas[resto], random_state=semilla)
    conjuntos = np.empty(len(estrato), dtype=object)
    conjuntos[entrenamiento], conjuntos[validacion], conjuntos[prueba] = ENTRENAMIENTO, VALIDACION, PRUEBA
    return pd.Series(conjuntos, index=estrato.index, name="conjunto")


def huella(etiquetas: pd.Series) -> str:
    """SHA-256 de la secuencia, por posición, de las etiquetas (conjuntos o pertenencia a la muestra)."""
    texto = "".join(CODIGO.get(e, str(e)) for e in etiquetas.astype(object))
    return hashlib.sha256(texto.encode("ascii")).hexdigest()


def muestra_estratificada(conjuntos: pd.Series, estrato: pd.Series, tamano: int, semilla: int = SEMILLA) -> pd.Series:
    """Marca una muestra de `tamano` filas de entrenamiento, estratificada por `estrato`."""
    posiciones = np.flatnonzero(conjuntos.to_numpy() == ENTRENAMIENTO)
    elegidas, _ = train_test_split(posiciones, train_size=tamano, stratify=estrato.to_numpy()[posiciones], random_state=semilla)
    en_muestra = np.zeros(len(conjuntos), dtype=bool)
    en_muestra[elegidas] = True
    return pd.Series(en_muestra, index=conjuntos.index, name="muestra")


def escribir(conjuntos: pd.Series, muestra: pd.Series, ruta: Path = RUTA_PARTICION) -> Path:
    """Guarda posición, conjunto y muestra; la carpeta `data/` no se versiona."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"posicion": np.arange(len(conjuntos)), "conjunto": conjuntos.to_numpy(), "muestra": muestra.to_numpy()}).to_parquet(ruta, index=False)
    return ruta
