"""La prueba se toca una sola vez: el registro, los dos archivos de evaluaciones y el cuaderno de la fase 3.

(a) `evaluar_en_prueba` y `predecir_en_prueba` fallan si el modelo ya está en el registro, y
la segunda ni siquiera predice; (b) si existen, los registros de las fases 2 y 3 no repiten
modelos, ni entre sí ni dentro, y el de la fase 3, cuando el cuaderno ya llegó a §22, tiene
exactamente `FINALES_FASE_3`; (c) en el cuaderno de la fase 3, `X_prueba` e `y_prueba` solo
aparecen en §1, donde se recuperan, y en §22, donde se evalúa.
"""

import json
import re

import pytest

from saber11 import datos, modelos

RESULTADOS = datos.RAIZ_PROYECTO / "resultados"
REGISTRO_FASE_2 = RESULTADOS / "evaluaciones-prueba.json"
REGISTRO_FASE_3 = RESULTADOS / "evaluaciones-prueba-fase-3.json"
CUADERNO = datos.RAIZ_PROYECTO / "notebooks" / "fase-3-modelado.py"
SECCION = re.compile(r"^# ## (\d+)\.")
PRUEBA = re.compile(r"\b[Xy]_prueba\b")
SECCIONES_CON_PRUEBA = {1, 22}


def nombres(ruta):
    return [fila["modelo"] for fila in json.loads(ruta.read_text(encoding="utf-8"))]


def test_evaluar_en_prueba_falla_si_el_modelo_ya_esta_en_el_registro(tmp_path):
    ruta = tmp_path / "evaluaciones-prueba-fase-3.json"
    modelos.iniciar_registro_prueba(ruta)
    modelos.evaluar_en_prueba(ruta, "ridge", {"rmse": 40.0})
    with pytest.raises(ValueError, match="una sola vez"):
        modelos.evaluar_en_prueba(ruta, "ridge", {"rmse": 39.0})


def test_predecir_en_prueba_registra_devuelve_y_no_vuelve_a_predecir(tmp_path):
    ruta = tmp_path / "evaluaciones-prueba-fase-3.json"
    modelos.iniciar_registro_prueba(ruta)
    llamadas = []

    def predecir():
        llamadas.append(1)
        return [250.0, 260.0]

    predicciones = modelos.predecir_en_prueba(ruta, "lasso", predecir, lambda p: {"media": sum(p) / len(p)})
    assert predicciones == [250.0, 260.0]
    assert json.loads(ruta.read_text()) == [{"modelo": "lasso", "metricas": {"media": 255.0}}]
    with pytest.raises(ValueError, match="una sola vez"):
        modelos.predecir_en_prueba(ruta, "lasso", predecir, lambda p: {})
    assert len(llamadas) == 1


@pytest.mark.skipif(not REGISTRO_FASE_2.exists(), reason="sin el registro de la fase 2")
def test_el_registro_de_la_fase_2_no_repite_modelos():
    fase_2 = nombres(REGISTRO_FASE_2)
    assert len(fase_2) == len(set(fase_2))


@pytest.mark.skipif(not REGISTRO_FASE_3.exists(), reason="sin el registro de la fase 3")
def test_el_registro_de_la_fase_3_no_repite_ni_choca_con_la_fase_2_y_tiene_los_finales():
    fase_3 = nombres(REGISTRO_FASE_3)
    assert len(fase_3) == len(set(fase_3))
    if REGISTRO_FASE_2.exists():
        assert not set(fase_3) & set(nombres(REGISTRO_FASE_2))
    if fase_3:
        assert sorted(fase_3) == sorted(modelos.FINALES_FASE_3)


def test_los_finales_de_la_fase_3_no_repiten_ni_usan_nombres_de_la_fase_2():
    assert len(modelos.FINALES_FASE_3) == len(set(modelos.FINALES_FASE_3))
    if REGISTRO_FASE_2.exists():
        assert not set(modelos.FINALES_FASE_3) & set(nombres(REGISTRO_FASE_2))


@pytest.mark.skipif(not CUADERNO.exists(), reason="el cuaderno de la fase 3 aún no existe")
def test_el_cuaderno_solo_toca_la_prueba_en_las_secciones_1_y_22():
    seccion, fuera = 0, []
    for numero, linea in enumerate(CUADERNO.read_text(encoding="utf-8").splitlines(), start=1):
        encabezado = SECCION.match(linea)
        if encabezado:
            seccion = int(encabezado.group(1))
        elif PRUEBA.search(linea) and seccion not in SECCIONES_CON_PRUEBA:
            fuera.append(f"línea {numero} (§{seccion}): {linea.strip()}")
    assert not fuera, "\n".join(fuera)
