"""Sin fuga de datos: nada de validación ni de prueba entra al ajuste, y ningún predictor reconstruye el objetivo.

La estrategia es poner valores centinela extremos en validación y prueba (edad de 1000
años, puntaje de 9999, un departamento inventado) y comprobar que el Pipeline ajustado con
`modelos.ajustar` queda idéntico al ajustado sin esos centinelas.
"""

import numpy as np
import pandas as pd
import pytest

from saber11 import modelos, particion
from saber11 import variables as v


def con_centinelas(limpio: pd.DataFrame, conjuntos: pd.Series) -> pd.DataFrame:
    fuera = conjuntos.ne(particion.ENTRENAMIENTO)
    alterado = limpio.copy()
    alterado.loc[fuera, v.EDAD] = 1000.0
    alterado.loc[fuera, v.CUARTOS_HOGAR] = np.nan
    alterado.loc[fuera, v.OBJETIVO_REGRESION] = 9999.0
    alterado.loc[fuera, v.DEPARTAMENTO] = "CENTINELA"
    alterado.loc[fuera, v.ESTRATO] = "Estrato 6"
    return alterado


def ajustado(marco: pd.DataFrame, conjuntos: pd.Series, **opciones: object):
    pipeline = modelos.regresion_multiple("componentes", **opciones)
    return modelos.ajustar(pipeline, marco[list(v.ENTRADA_REGRESION)], marco[v.OBJETIVO_REGRESION], conjuntos)


@pytest.mark.parametrize("opciones", [{"imputacion": "moda"}, {"imputacion": "iterativa", "departamento": "objetivo"}])
def test_el_ajuste_no_ve_validacion_ni_prueba(limpio, opciones):
    conjuntos = particion.particionar(limpio["COLE_NATURALEZA"])
    limpio_ajustado = ajustado(limpio, conjuntos, **opciones)
    centinela_ajustado = ajustado(con_centinelas(limpio, conjuntos), conjuntos, **opciones)

    entrenamiento = limpio.loc[conjuntos.eq(particion.ENTRENAMIENTO), list(v.ENTRADA_REGRESION)]
    sin_modelo = slice(None, -1)
    pd.testing.assert_frame_equal(limpio_ajustado[sin_modelo].transform(entrenamiento), centinela_ajustado[sin_modelo].transform(entrenamiento))
    escalador = "numericas"
    np.testing.assert_array_equal(
        limpio_ajustado["codificacion"].named_transformers_[escalador].mean_,
        centinela_ajustado["codificacion"].named_transformers_[escalador].mean_,
    )
    np.testing.assert_array_equal(limpio_ajustado["modelo"].coef_, centinela_ajustado["modelo"].coef_)
    if opciones.get("departamento") == "objetivo":
        codificador = limpio_ajustado["codificacion"].named_transformers_["departamento"][0]
        assert "CENTINELA" not in codificador.categories_[0]


def test_los_centinelas_cambiarian_el_ajuste_si_se_filtraran(limpio):
    conjuntos = particion.particionar(limpio["COLE_NATURALEZA"])
    todo = pd.Series(particion.ENTRENAMIENTO, index=limpio.index)
    limpio_ajustado = ajustado(limpio, conjuntos, imputacion="moda")
    filtrado = ajustado(con_centinelas(limpio, conjuntos), todo, imputacion="moda")
    medias = [p["codificacion"].named_transformers_["numericas"].mean_ for p in (limpio_ajustado, filtrado)]
    assert not np.allclose(*medias)


@pytest.mark.filterwarnings("ignore:Found unknown categories")
def test_transformar_validacion_no_reajusta(limpio):
    conjuntos = particion.particionar(limpio["COLE_NATURALEZA"])
    pipeline = ajustado(limpio, conjuntos, imputacion="moda")
    antes = pipeline["codificacion"].named_transformers_["numericas"].mean_.copy()
    validacion = con_centinelas(limpio, conjuntos).loc[conjuntos.eq(particion.VALIDACION), list(v.ENTRADA_REGRESION)]
    pipeline.predict(validacion)
    np.testing.assert_array_equal(antes, pipeline["codificacion"].named_transformers_["numericas"].mean_)


@pytest.mark.parametrize("especificacion", sorted(v.PREDICTORES_REGRESION))
def test_ningun_predictor_de_regresion_reconstruye_el_puntaje_global(especificacion):
    predictores = set(v.PREDICTORES_REGRESION[especificacion])
    assert not predictores & v.PROHIBIDOS_REGRESION
    assert v.OBJETIVO_REGRESION not in predictores
    assert not any(p.startswith("PUNT_") for p in predictores)


def test_la_clasificacion_solo_usa_hogar_y_estudiante():
    assert set(v.PREDICTORES_CLASIFICACION) <= set(v.HOGAR) | set(v.ESTUDIANTE)
    assert not set(v.PREDICTORES_CLASIFICACION) & set(v.COLEGIO)


def test_las_columnas_finales_de_regresion_no_traen_puntajes(limpio):
    conjuntos = particion.particionar(limpio["COLE_NATURALEZA"])
    pipeline = ajustado(limpio, conjuntos, imputacion="moda")
    columnas = pipeline["codificacion"].get_feature_names_out()
    assert not any(c.startswith("PUNT_") or c in v.PROHIBIDOS_REGRESION for c in columnas)
