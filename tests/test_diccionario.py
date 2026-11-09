"""Pruebas de saber11.diccionario: cobertura exacta de las 51 columnas del dataset."""

from saber11 import datos
from saber11.diccionario import COLUMNAS_EXCLUIDAS, CONTINUA, DISCRETA, NOMINAL, ORDINAL, VARIABLES

COLUMNAS_DEL_DATASET = 51
GRUPOS = {"periodo": ("PERIODO",), "puntaje": datos.PUNTAJES, "hogar": datos.HOGAR, "colegio": datos.COLEGIO, "estudiante": datos.ESTUDIANTE}


def excluidas() -> list[str]:
    return [columna for columnas in COLUMNAS_EXCLUIDAS.values() for columna in columnas]


def test_variables_cubre_exactamente_las_columnas_de_los_recortes():
    assert tuple(VARIABLES) == datos.COLUMNAS


def test_excluidas_no_se_repiten_ni_se_cruzan_con_las_variables():
    assert len(excluidas()) == len(set(excluidas()))
    assert not set(excluidas()) & set(VARIABLES)


def test_variables_y_excluidas_suman_las_51_columnas_del_dataset():
    assert len(VARIABLES) + len(excluidas()) == COLUMNAS_DEL_DATASET


def test_las_columnas_de_control_de_datos_quedan_entre_las_excluidas():
    assert set(datos.CONTROL) <= set(excluidas())


def test_cada_variable_tiene_tipo_valido_descripcion_y_el_grupo_de_datos():
    for columna, variable in VARIABLES.items():
        assert variable.tipo in {CONTINUA, DISCRETA, NOMINAL, ORDINAL}
        assert variable.descripcion
        assert columna in GRUPOS[variable.grupo]
