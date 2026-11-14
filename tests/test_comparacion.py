"""Pruebas de saber11.comparacion: McNemar con una tabla conocida y el t de Nadeau y Bengio."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from saber11 import comparacion


def predicciones_con_tabla(ambos: int, solo_a: int, solo_b: int, ninguno: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Etiquetas reales y dos predicciones cuyos aciertos dan la tabla pedida."""
    y = np.array(["NO OFICIAL"] * (ambos + solo_a + solo_b + ninguno))
    fallo = "OFICIAL"
    a = np.array(["NO OFICIAL"] * ambos + ["NO OFICIAL"] * solo_a + [fallo] * solo_b + [fallo] * ninguno)
    b = np.array(["NO OFICIAL"] * ambos + [fallo] * solo_a + ["NO OFICIAL"] * solo_b + [fallo] * ninguno)
    return y, a, b


def test_tabla_de_aciertos_cuenta_cada_caso():
    y, a, b = predicciones_con_tabla(50, 10, 20, 40)
    assert comparacion.tabla_aciertos(y, a, b).tolist() == [[50, 10], [20, 40]]


def test_mcnemar_con_una_tabla_conocida():
    y, a, b = predicciones_con_tabla(50, 10, 20, 40)
    c = comparacion.mcnemar(y, a, b)
    assert c.estadistico == pytest.approx((abs(10 - 20) - 1) ** 2 / (10 + 20))
    assert c.p == pytest.approx(stats.chi2.sf(2.7, 1))
    assert c.detalle == {"solo_acierta_a": 10, "solo_acierta_b": 20}
    assert c.n == 120


def test_nadeau_bengio_contra_su_formula():
    diferencias = np.array([0.8, 1.1, 0.4, 1.6, 0.9, 1.3, 0.2, 1.0, 0.7, 1.2])
    j, n_entrenamiento, n_prueba = len(diferencias), 800, 200
    t = diferencias.mean() / np.sqrt((1 / j + n_prueba / n_entrenamiento) * diferencias.var(ddof=1))
    c = comparacion.nadeau_bengio(diferencias, n_entrenamiento, n_prueba)
    assert c.estadistico == pytest.approx(t)
    assert c.gl == (9.0,)
    assert c.p == pytest.approx(2 * stats.t.sf(abs(t), 9))
    sin_corregir = diferencias.mean() / np.sqrt(diferencias.var(ddof=1) / j)
    assert abs(c.estadistico) < abs(sin_corregir)


def test_diferencias_por_pliegue_recorren_cinco_por_cinco_y_restan_a_menos_b():
    rng = np.random.default_rng(3)
    X = pd.DataFrame({"x": rng.normal(size=500)})
    y = pd.Series(3 * X["x"] + rng.normal(scale=0.5, size=500))
    tabla = comparacion.diferencias_por_pliegue(LinearRegression(), DummyRegressor(), X, y)
    assert len(tabla) == 25
    assert set(tabla["repeticion"]) == set(tabla["pliegue"]) == {1, 2, 3, 4, 5}
    assert (tabla["n_entrenamiento"] + tabla["n_prueba"] == 500).all()
    assert tabla["diferencia"].to_numpy() == pytest.approx((tabla["metrica_a"] - tabla["metrica_b"]).to_numpy())
    assert (tabla["diferencia"] < 0).all()
    otra = comparacion.diferencias_por_pliegue(LinearRegression(), DummyRegressor(), X, y)
    pd.testing.assert_frame_equal(tabla, otra)
