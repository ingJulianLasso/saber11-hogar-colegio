"""Pruebas de saber11.interpretabilidad: agregación por variable original, SHAP y errores por grupo."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

from saber11 import interpretabilidad
from saber11 import variables as v

PREDICTORES = (v.ESTRATO, "FAMI_TIENEINTERNET", v.EDAD)
CODIFICADAS = [
    "FAMI_ESTRATOVIVIENDA_Estrato 2",
    "FAMI_ESTRATOVIVIENDA_Estrato 3",
    "FAMI_TIENEINTERNET_Si",
    "ESTU_EDAD",
    "FALTA_FAMI_TIENEINTERNET",
]


@pytest.fixture
def codificado() -> pd.DataFrame:
    rng = np.random.default_rng(9)
    estrato = rng.choice(3, size=800)
    return pd.DataFrame({
        "FAMI_ESTRATOVIVIENDA_Estrato 2": (estrato == 1).astype(float),
        "FAMI_ESTRATOVIVIENDA_Estrato 3": (estrato == 2).astype(float),
        "FAMI_TIENEINTERNET_Si": rng.choice([0.0, 1.0], size=800),
        "ESTU_EDAD": rng.normal(size=800),
        "FALTA_FAMI_TIENEINTERNET": rng.choice([0.0, 1.0], size=800, p=[0.9, 0.1]),
    })


def test_las_contribuciones_por_variable_suman_lo_mismo_que_las_codificadas():
    rng = np.random.default_rng(1)
    contribuciones = pd.DataFrame(rng.normal(size=(50, len(CODIFICADAS))), columns=CODIFICADAS)
    agregadas = interpretabilidad.por_variable(contribuciones, PREDICTORES)
    assert list(agregadas.columns) == [v.ESTRATO, "FAMI_TIENEINTERNET", v.EDAD]
    assert agregadas.sum(axis=1).to_numpy() == pytest.approx(contribuciones.sum(axis=1).to_numpy())
    esperada = contribuciones["FAMI_TIENEINTERNET_Si"] + contribuciones["FALTA_FAMI_TIENEINTERNET"]
    assert agregadas["FAMI_TIENEINTERNET"].to_numpy() == pytest.approx(esperada.to_numpy())


def test_shap_de_un_xgboost_es_aditivo_antes_y_despues_de_agregar(codificado):
    y = 250 + 15 * codificado["FAMI_ESTRATOVIVIENDA_Estrato 3"] + 10 * codificado["FAMI_TIENEINTERNET_Si"] + 5 * codificado["ESTU_EDAD"]
    modelo = XGBRegressor(n_estimators=30, max_depth=3, random_state=42).fit(codificado, y)
    contribuciones = interpretabilidad.contribuciones_shap(modelo, codificado)
    base = modelo.predict(codificado) - contribuciones.sum(axis=1).to_numpy()
    assert np.ptp(base) == pytest.approx(0, abs=1e-3)
    agregadas = interpretabilidad.por_variable(contribuciones, PREDICTORES)
    assert agregadas.sum(axis=1).to_numpy() == pytest.approx(contribuciones.sum(axis=1).to_numpy(), abs=1e-4)
    importancia = interpretabilidad.importancia_media(agregadas)
    assert importancia.index[0] == v.ESTRATO
    assert interpretabilidad.spearman(importancia, importancia.iloc[::-1]) == pytest.approx(1.0)


def test_errores_por_grupo_dejan_solo_grupos_de_diez_o_mas():
    grupo = pd.Series(["URBANO"] * 30 + ["RURAL"] * 12 + ["OTRO"] * 5)
    y = pd.Series(np.r_[np.full(30, 250.0), np.full(12, 200.0), np.full(5, 300.0)])
    prediccion = np.r_[np.full(30, 240.0), np.full(12, 210.0), np.full(5, 300.0)]
    tabla = interpretabilidad.errores_regresion(y, prediccion, grupo)
    assert set(tabla.index) == {"URBANO", "RURAL"}
    assert tabla.loc["URBANO", "residuo_medio"] == pytest.approx(10.0)
    assert tabla.loc["RURAL", "rmse"] == pytest.approx(10.0)


def test_errores_de_clasificacion_por_grupo():
    grupo = pd.Series(["A"] * 20)
    y = pd.Series(["NO OFICIAL"] * 10 + ["OFICIAL"] * 10)
    prediccion = np.array(["NO OFICIAL"] * 6 + ["OFICIAL"] * 4 + ["OFICIAL"] * 8 + ["NO OFICIAL"] * 2)
    fila = interpretabilidad.errores_clasificacion(y, prediccion, grupo).loc["A"]
    assert fila["error"] == pytest.approx(6 / 20)
    assert fila["falsos_negativos"] == pytest.approx(0.4)
    assert fila["falsos_positivos"] == pytest.approx(0.2)


def test_r2_con_las_mas_importantes_entrena_solo_con_las_k_primeras():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(400, 3)), columns=["a", "b", "c"])
    y = 3 * X["a"] + X["b"]
    importancia = pd.Series({"b": 0.5, "a": 2.0, "c": 0.1})
    recibidos = []

    def ajustar(predictores):
        recibidos.append(predictores)
        return make_pipeline(ColumnTransformer([("elegidas", "passthrough", list(predictores))]), LinearRegression()).fit(X, y)

    assert interpretabilidad.mas_importantes(importancia, 2) == ("a", "b")
    assert interpretabilidad.r2_con_las_mas_importantes(ajustar, importancia, 2, (X, y)) == pytest.approx(1.0)
    assert interpretabilidad.r2_con_las_mas_importantes(ajustar, importancia, 1, (X, y)) == pytest.approx(0.9, abs=0.05)
    assert recibidos == [("a", "b"), ("a",)]


def test_media_por_categoria_omite_las_categorias_pequenas():
    contribucion = pd.Series(np.r_[np.ones(15), np.full(4, 9.0)])
    categoria = pd.Series(["Si"] * 15 + ["No"] * 4)
    tabla = interpretabilidad.media_por_categoria(contribucion, categoria)
    assert list(tabla.index) == ["Si"]
    assert tabla.loc["Si", "media"] == pytest.approx(1.0)
