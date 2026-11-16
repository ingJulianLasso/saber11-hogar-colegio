"""Pruebas de saber11.cifras."""

import json
import re

import pytest

from saber11.cifras import Cifras, comprobar_volcado, en_espanol, nombre_metrica


@pytest.mark.parametrize(
    ("valor", "decimales", "esperado"),
    [
        (1234, 0, "1234"),
        (12345, 0, "12.345"),
        (532566, 0, "532.566"),
        (0.5, 2, "0,50"),
        (-12345.6, 1, "-12.345,6"),
        (9999.95, 1, "10000,0"),
    ],
)
def test_en_espanol(valor, decimales, esperado):
    assert en_espanol(valor, decimales) == esperado


def test_cifra_rechaza_claves_que_latex_no_admite():
    with pytest.raises(ValueError):
        Cifras().cifra("r2_test", 0.5, 2)


def test_volcar_escribe_json_y_tex_que_coinciden(tmp_path):
    cifras = Cifras()
    assert cifras.cifra("filasModelado", 532566) == "532.566"
    cifras.cifra("rDosPrueba", 0.4321, 3)
    cifras.cifra("versionTorch", "2.11.0+cu130_x")
    ruta_json, ruta_tex = cifras.volcar(tmp_path / "resultados", "cifras-fase-1")

    datos = json.loads(ruta_json.read_text(encoding="utf-8"))
    assert datos["rDosPrueba"] == {"valor": 0.4321, "texto": "0,432"}
    tex = ruta_tex.read_text(encoding="utf-8")
    assert r"\newcommand{\filasModelado}{532.566}" in tex
    assert r"\newcommand{\versionTorch}{2.11.0+cu130\_x}" in tex
    assert len(cifras) == 3


def test_comprobar_volcado_detecta_un_tex_editado_a_mano(tmp_path):
    cifras = Cifras()
    cifras.cifra("filasModelado", 532566)
    ruta_json, ruta_tex = cifras.volcar(tmp_path, "cifras-fase-1")
    ruta_tex.write_text(ruta_tex.read_text(encoding="utf-8").replace("532.566", "532.567"), encoding="utf-8")
    with pytest.raises(ValueError):
        comprobar_volcado(ruta_json, ruta_tex)


def escribir_fase(carpeta, numero, cifras):
    ruta = carpeta / f"cifras-fase-{numero}.json"
    ruta.write_text(json.dumps({clave: {"valor": valor, "texto": texto} for clave, (valor, texto) in cifras.items()}), encoding="utf-8")
    return ruta


def test_consolidar_deja_una_vez_las_claves_iguales_y_sufija_las_que_chocan(tmp_path):
    rutas = [
        escribir_fase(tmp_path, 1, {"filasRecorte": (532566, "532.566"), "variablesConFaltantes": (21, "21")}),
        escribir_fase(tmp_path, 2, {"filasRecorte": (532566, "532.566"), "variablesConFaltantes": (12, "12"), "huella": ("ab12", "ab12")}),
        escribir_fase(tmp_path, 3, {"variablesConFaltantes": (12, "12")}),
    ]
    cifras = Cifras.consolidar(rutas)
    assert len(cifras) == 5
    assert cifras.texto("filasRecorte") == "532.566"
    assert "variablesConFaltantes" not in cifras
    assert [cifras.valor(f"variablesConFaltantes{s}") for s in ("FaseUno", "FaseDos", "FaseTres")] == [21, 12, 12]
    assert cifras.valor("huella") == "ab12"


def test_consolidar_rechaza_claves_con_digitos(tmp_path):
    rutas = [escribir_fase(tmp_path, 1, {"r2Prueba": (0.5, "0,500")})]
    with pytest.raises(ValueError):
        Cifras.consolidar(rutas)


def test_consolidar_falla_si_un_sufijo_repite_una_clave(tmp_path):
    rutas = [
        escribir_fase(tmp_path, 1, {"media": (1, "1"), "mediaFaseUno": (2, "2")}),
        escribir_fase(tmp_path, 2, {"media": (3, "3")}),
    ]
    with pytest.raises(ValueError):
        Cifras.consolidar(rutas)


def test_el_volcado_consolidado_no_repite_newcommand(tmp_path):
    rutas = [
        escribir_fase(tmp_path, 1, {"filasRecorte": (532566, "532.566"), "media": (1.5, "1,5")}),
        escribir_fase(tmp_path, 2, {"filasRecorte": (532566, "532.566"), "media": (2.5, "2,5")}),
    ]
    cifras = Cifras.consolidar(rutas)
    cifras.cifra("pesoModeloTablero", 12.3, 1)
    _, ruta_tex = cifras.volcar(tmp_path / "resultados", "cifras-fase-4")
    macros = re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", ruta_tex.read_text(encoding="utf-8"))
    assert len(macros) == len(set(macros)) == 4


@pytest.mark.parametrize(
    ("metrica", "esperado"),
    [("r2", "rCuadrado"), ("f1", "fUno"), ("f1Macro", "fUnoMacro"), ("roc_auc", "rocAuc"), ("aucPr", "aucPr"), ("rmse", "rmse")],
)
def test_nombre_metrica(metrica, esperado):
    assert nombre_metrica(metrica) == esperado


def test_metricas_de_prueba_reusa_la_macro_de_la_fase_y_crea_solo_las_que_faltan(tmp_path):
    cifras = Cifras.consolidar([escribir_fase(tmp_path, 2, {"rmsePruebaMultiple": (42.78, "42,8"), "rmseValidacionMultiple": (40.0, "40,0")})])
    registro = tmp_path / "evaluaciones-prueba.json"
    registro.write_text(json.dumps([
        {"modelo": "regresionMultiple", "metricas": {"rmse": 42.78, "r2": 0.32}},
        {"modelo": "xgboost2", "metricas": {"f1": 0.5}},
    ]), encoding="utf-8")
    nombres = cifras.metricas_de_prueba(registro)
    assert nombres == {
        ("regresionMultiple", "rmse"): "rmsePruebaMultiple",
        ("regresionMultiple", "r2"): "rCuadradoPruebaRegresionMultiple",
        ("xgboost2", "f1"): "fUnoPruebaXgboostDos",
    }
    assert cifras.texto("rCuadradoPruebaRegresionMultiple") == "0,320"
    assert len(cifras) == 4


def test_metricas_de_prueba_falla_si_dos_macros_tienen_el_mismo_valor(tmp_path):
    cifras = Cifras.consolidar([escribir_fase(tmp_path, 3, {"aucPruebaA": (0.7, "0,700"), "aucPruebaB": (0.7, "0,700")})])
    registro = tmp_path / "evaluaciones-prueba.json"
    registro.write_text(json.dumps([{"modelo": "a", "metricas": {"auc": 0.7}}]), encoding="utf-8")
    with pytest.raises(ValueError):
        cifras.metricas_de_prueba(registro)
