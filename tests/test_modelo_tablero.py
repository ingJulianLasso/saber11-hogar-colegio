"""Pruebas del modelo de predicción que publica el tablero (`dashboard/modelo/`).

Las del modelo exportado leen lo que deja `notebooks/fase-4-sintesis.ipynb`; corren sin GPU
(`CUDA_VISIBLE_DEVICES=`), como en Streamlit Community Cloud. Las de `saber11.tablero` usan
datos sintéticos.
"""

import json
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest

from saber11 import datos, tablero
from saber11 import variables as v

CARPETA_MODELO = datos.RAIZ_PROYECTO / "dashboard" / "modelo"
RUTA_MODELO = CARPETA_MODELO / "regresion.joblib"
RUTA_FICHA = CARPETA_MODELO / "ficha.json"


@pytest.fixture(scope="module")
def modelo():
    return joblib.load(RUTA_MODELO)


@pytest.fixture(scope="module")
def ficha():
    return json.loads(RUTA_FICHA.read_text(encoding="utf-8"))


def test_el_modelo_pesa_menos_de_cincuenta_megabytes():
    assert RUTA_MODELO.stat().st_size / 1e6 < tablero.PESO_MAXIMO_MB


def test_el_modelo_no_tiene_vecinos_ni_vectores_de_soporte(modelo):
    assert tablero.clases_con_observaciones(modelo) == []


def test_el_modelo_no_guarda_matrices_de_mil_filas_o_mas(modelo):
    assert tablero.matrices_grandes(modelo) == []


def test_toda_hoja_cubre_diez_o_mas_estudiantes(modelo):
    assert tablero.cobertura_minima_de_hojas(modelo.named_steps["modelo"]) >= tablero.MINIMO_POR_HOJA


def test_el_modelo_predice_en_la_cpu(modelo):
    assert modelo.named_steps["modelo"].get_params()["device"] == "cpu"


def test_las_versiones_de_la_ficha_son_las_instaladas(ficha):
    assert ficha["versiones"] == tablero.versiones()


def test_la_ficha_tiene_las_entradas_del_pipeline_y_un_margen_positivo(modelo, ficha):
    assert [entrada["columna"] for entrada in ficha["entrada"]] == list(v.ENTRADA_REGRESION)
    assert {e["columna"] for e in ficha["entrada"] if e["sin_dato"]} == set(modelo.named_steps["imputacion"].con_faltantes_)
    assert ficha["margen"]["puntos"] > 0
    assert ficha["margen"]["cobertura_validacion"] >= ficha["margen"]["nivel"]


def test_un_perfil_fijo_da_una_estimacion_entre_cero_y_quinientos(modelo, ficha):
    completo = tablero.perfil(ficha, tablero.PERFIL_DE_EJEMPLO)
    sin_dato = tablero.perfil(ficha, tablero.PERFIL_DE_EJEMPLO | {e["columna"]: None for e in ficha["entrada"] if e["sin_dato"]})
    perfiles = pd.concat([completo, sin_dato], ignore_index=True)
    with warnings.catch_warnings():
        # Un nivel desconocido para el one-hot sería un «Sin dato» que no se imputó.
        warnings.filterwarnings("error", message="Found unknown categories")
        estimaciones = modelo.predict(perfiles)
    assert np.all((estimaciones >= 0) & (estimaciones <= 500))
    assert not modelo.named_steps["imputacion"].transform(perfiles).isna().any().any()


def test_perfil_rechaza_un_nivel_que_no_admite_la_ficha(ficha):
    with pytest.raises(ValueError):
        tablero.perfil(ficha, tablero.PERFIL_DE_EJEMPLO | {v.ESTRATO: "Estrato 9"})


def test_perfil_rechaza_sin_dato_donde_el_pipeline_no_imputa(ficha):
    sin_imputacion = next(e["columna"] for e in ficha["entrada"] if not e["sin_dato"])
    with pytest.raises(ValueError):
        tablero.perfil(ficha, tablero.PERFIL_DE_EJEMPLO | {sin_imputacion: None})


def test_margen_conformal_toma_el_residuo_en_la_posicion_corregida():
    residuos = np.arange(1, 20, dtype=float)
    assert tablero.margen_conformal(residuos, 0.9) == 18.0
    with pytest.raises(ValueError):
        tablero.margen_conformal(residuos[:5], 0.9)


def test_matrices_grandes_detecta_observaciones_guardadas():
    class Guardon:
        __module__ = "saber11.prueba"

        def __init__(self) -> None:
            self.chica = np.zeros((999, 3))
            self.vector = np.zeros(5000)
            self.anidada = {"X": [np.zeros((1000, 2))]}

    assert tablero.matrices_grandes(Guardon()) == ["modelo.anidada['X'][0]"]


def test_entradas_ofrecen_solo_niveles_con_diez_o_mas(limpio):
    X = limpio[list(v.ENTRADA_REGRESION)].copy()
    X.loc[X.index[:3], "COLE_JORNADA"] = "NOCHE"
    formulario = {e["columna"]: e for e in tablero.entradas(X, ["COLE_BILINGUE"])}
    assert "NOCHE" not in formulario["COLE_JORNADA"]["niveles"]
    assert formulario[v.ESTRATO]["niveles"][:6] == list(v.ESTRATOS)
    assert formulario[v.PERSONAS_HOGAR]["respuestas"][0] == {"valor": 1.5, "respuesta": "1 a 2"}
    assert formulario["COLE_BILINGUE"]["sin_dato"] and not formulario[v.ESTRATO]["sin_dato"]
