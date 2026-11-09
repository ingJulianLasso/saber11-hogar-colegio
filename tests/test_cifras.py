"""Pruebas de saber11.cifras."""

import json

import pytest

from saber11.cifras import Cifras, comprobar_volcado, en_espanol


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
