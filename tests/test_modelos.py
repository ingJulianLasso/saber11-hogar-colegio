"""Pruebas de saber11.modelos: evaluación única en prueba, fórmula y nombres de términos."""

import json

import numpy as np
import pandas as pd
import pytest

from saber11 import modelos
from saber11 import variables as v


def test_evaluar_dos_veces_el_mismo_modelo_en_prueba_falla(tmp_path):
    ruta = tmp_path / "evaluaciones-prueba.json"
    modelos.iniciar_registro_prueba(ruta)
    modelos.evaluar_en_prueba(ruta, "linealSimple", {"rmse": 40.0})
    modelos.evaluar_en_prueba(ruta, "dummyRegresion", {"rmse": 50.0})
    with pytest.raises(ValueError, match="una sola vez"):
        modelos.evaluar_en_prueba(ruta, "linealSimple", {"rmse": 39.0})
    assert [fila["modelo"] for fila in json.loads(ruta.read_text())] == ["linealSimple", "dummyRegresion"]


def test_iniciar_registro_lo_deja_vacio(tmp_path):
    ruta = tmp_path / "evaluaciones-prueba.json"
    ruta.write_text('[{"modelo": "viejo", "metricas": {}}]')
    modelos.iniciar_registro_prueba(ruta)
    assert json.loads(ruta.read_text()) == []


def test_formula_usa_las_referencias_explicitas():
    texto = modelos.formula("PUNT_GLOBAL", (v.ESTRATO, v.EDAD), ["FALTA_FAMI_ESTRATOVIVIENDA"])
    assert texto == "PUNT_GLOBAL ~ C(FAMI_ESTRATOVIVIENDA, Treatment('Estrato 1')) + ESTU_EDAD + FALTA_FAMI_ESTRATOVIVIENDA"


@pytest.mark.parametrize(
    ("nombre", "clave"),
    [
        ("C(FAMI_ESTRATOVIVIENDA, Treatment('Estrato 1'))[T.Estrato 2]", "FAMI_ESTRATOVIVIENDA=Estrato 2"),
        ("FAMI_ESTRATOVIVIENDA_Estrato 2", "FAMI_ESTRATOVIVIENDA=Estrato 2"),
        ("FAMI_EDUCACIONMADRE_Secundaria (Bachillerato) completa", "FAMI_EDUCACIONMADRE=Secundaria (Bachillerato) completa"),
        ("ESTU_EDAD", "ESTU_EDAD"),
        ("FALTA_FAMI_TIENEINTERNET", "FALTA_FAMI_TIENEINTERNET"),
    ],
)
def test_clave_termino_une_los_nombres_de_statsmodels_y_del_one_hot(nombre, clave):
    assert modelos.clave_termino(nombre) == clave


def test_metricas_de_clasificacion_toman_no_oficial_como_positiva():
    y = pd.Series(["OFICIAL", "NO OFICIAL", "NO OFICIAL", "OFICIAL"])
    prediccion = np.array(["OFICIAL", "NO OFICIAL", "OFICIAL", "OFICIAL"])
    metricas = modelos.metricas_clasificacion(y, prediccion, np.array([0.1, 0.9, 0.4, 0.2]))
    assert metricas["recall"] == pytest.approx(0.5)
    assert metricas["precision"] == pytest.approx(1.0)
    assert metricas["auc"] == pytest.approx(1.0)


def test_vif_generalizado_coincide_con_el_vif_clasico_en_un_grado_de_libertad():
    rng = np.random.default_rng(3)
    a = rng.normal(size=2000)
    X = pd.DataFrame({"a": a, "b": a + rng.normal(scale=0.5, size=2000), "c_x": rng.normal(size=2000), "c_y": rng.normal(size=2000)})
    tabla = modelos.vif_generalizado(X, ("a", "b", "c"))
    clasico = np.diag(np.linalg.inv(np.corrcoef(X.to_numpy(), rowvar=False)))
    assert tabla.loc["a", "gvif"] == pytest.approx(clasico[0])
    assert tabla.loc["c", "gl"] == 2
    assert tabla.loc["c", "gvif_ajustado"] == pytest.approx(1.0, abs=0.01)
