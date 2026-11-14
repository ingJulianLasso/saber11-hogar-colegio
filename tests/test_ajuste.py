"""Pruebas de saber11.ajuste: reglas del presupuesto, espacios de búsqueda y caché con versión."""

import joblib
import numpy as np
import pytest
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import ParameterSampler
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor

from conftest import marco_limpio
from saber11 import ajuste, hipotesis, modelos
from saber11 import variables as v


@pytest.mark.parametrize(("segundos", "esperado"), [(1.0, 30), (6.0, 30), (10.0, 18), (179.0, 1), (200.0, 0)])
def test_n_iter_es_el_mayor_que_cabe_en_quince_minutos_hasta_treinta(segundos, esperado):
    assert ajuste.n_iter_que_cabe(segundos) == esperado


def test_reajuste_y_proyeccion():
    assert ajuste.reajuste_cabe(900.0)
    assert not ajuste.reajuste_cabe(900.1)
    assert ajuste.proyectar_segundos(10.0, 1000, 1000) == pytest.approx(10.0)
    assert ajuste.proyectar_segundos(10.0, 1000, 10_000) > 100.0


MODELOS = {
    "randomForestRegresion": (RandomForestRegressor(), v.PREDICTORES_REGRESION[modelos.ESPECIFICACION]),
    "xgboostRegresion": (XGBRegressor(), v.PREDICTORES_REGRESION[modelos.ESPECIFICACION]),
    "randomForestClasificacion": (RandomForestClassifier(), v.PREDICTORES_CLASIFICACION),
    "xgboostClasificacion": (XGBClassifier(), v.PREDICTORES_CLASIFICACION),
    "svmClasificacion": (SVC(), v.PREDICTORES_CLASIFICACION),
}


@pytest.mark.parametrize("nombre", sorted(MODELOS))
def test_los_espacios_se_aplican_al_pipeline(nombre):
    estimador, predictores = MODELOS[nombre]
    tuberia = modelos.pipeline(predictores, estimador)
    for configuracion in ParameterSampler(ajuste.ESPACIOS[nombre], n_iter=3, random_state=ajuste.SEMILLA):
        tuberia.set_params(**configuracion)


def test_buscar_es_reproducible_y_no_reajusta():
    limpio = marco_limpio(600)
    tuberia = modelos.pipeline(v.PREDICTORES_REGRESION[modelos.ESPECIFICACION], Ridge())
    espacio = {"modelo__alpha": [0.1, 1.0, 10.0, 100.0]}
    X, y = limpio[list(v.ENTRADA_REGRESION)], limpio[v.OBJETIVO_REGRESION]
    primera = ajuste.buscar(tuberia, espacio, 3, X, y, ajuste.PUNTUACION["regresion"])
    segunda = ajuste.buscar(tuberia, espacio, 3, X, y, ajuste.PUNTUACION["regresion"])
    assert primera.mejores == segunda.mejores
    assert primera.puntuacion == pytest.approx(segunda.puntuacion)
    assert len(primera.resultados) == 3
    assert not hasattr(tuberia.named_steps["modelo"], "coef_")


def test_la_version_cubre_las_dependencias_de_saber11():
    assert {"saber11.interpretabilidad", "saber11.modelos", "saber11.variables", "saber11.limpieza"} <= set(ajuste.dependencias("saber11.interpretabilidad"))
    assert ajuste.version("saber11.hipotesis") != ajuste.version("saber11.hipotesis", ajuste.NUCLEO)


def test_en_cache_guarda_por_argumentos_y_solo_acepta_funciones_de_saber11(tmp_path):
    guardada = ajuste.en_cache(hipotesis.d_cohen, carpeta=tmp_path)
    a, b = [1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 6.0]
    assert not guardada.check_call_in_cache(a, b)
    assert guardada(a, b) == pytest.approx(hipotesis.d_cohen(a, b))
    assert guardada.check_call_in_cache(a, b)
    assert not guardada.check_call_in_cache(a, [2.0, 3.0, 4.0, 7.0])
    assert any(p.name.startswith("hipotesis-") for p in tmp_path.iterdir())
    with pytest.raises(ValueError, match="saber11"):
        ajuste.en_cache(sorted, carpeta=tmp_path)


def test_los_espacios_tienen_una_clave_de_cache_estable():
    claves = {nombre: joblib.hash(espacio) for nombre, espacio in ajuste.ESPACIOS.items()}
    np.random.rand(3)
    list(ParameterSampler(ajuste.ESPACIOS["xgboostRegresion"], n_iter=3, random_state=ajuste.SEMILLA))
    assert claves == {nombre: joblib.hash(espacio) for nombre, espacio in ajuste.ESPACIOS.items()}


def test_configuracion_central_toma_medianas_y_el_medio_de_las_listas():
    espacio = {"a": stats.randint(1, 11), "b": stats.loguniform(0.01, 1), "c": [None, 10, 20, 30], "d": ["x", "y", "z"]}
    central = ajuste.configuracion_central(espacio)
    assert central == {"a": 5, "b": pytest.approx(0.1), "c": 20, "d": "y"}
    assert isinstance(central["a"], int)
    modelos.pipeline(v.PREDICTORES_CLASIFICACION, SVC()).set_params(**ajuste.configuracion_central(ajuste.ESPACIOS["svmClasificacion"]))


def test_ajustado_devuelve_una_copia_ajustada_sin_tocar_el_original():
    limpio = marco_limpio(300)
    tuberia = modelos.pipeline(v.PREDICTORES_REGRESION[modelos.ESPECIFICACION], Ridge())
    ajustada, segundos = ajuste.ajustado(tuberia, limpio[list(v.ENTRADA_REGRESION)], limpio[v.OBJETIVO_REGRESION])
    assert hasattr(ajustada.named_steps["modelo"], "coef_")
    assert segundos > 0
    assert not hasattr(tuberia.named_steps["modelo"], "coef_")
