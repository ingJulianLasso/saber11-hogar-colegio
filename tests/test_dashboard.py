"""El dashboard arranca, responde a cada filtro con el agregado que le toca y avisa cuando una
combinación no está publicada."""

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from saber11 import datos

APP = datos.RAIZ_PROYECTO / "dashboard" / "app.py"
AGREGADOS = datos.DESTINO / "agregados"
CUBO = AGREGADOS / "estrato-naturaleza-zona-20224.csv"
DEPARTAMENTO_ZONA = AGREGADOS / "departamento-zona-20224.csv"

pytestmark = pytest.mark.skipif(not CUBO.is_file(), reason="faltan los agregados de la fase 2")


def iniciar() -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=60)
    app.run()
    assert not app.exception
    return app


def numero(valor: float) -> str:
    texto = f"{valor:,.0f}" if valor >= 10_000 else f"{valor:.0f}"
    return texto.replace(",", ".")


def estudiantes(app: AppTest) -> str:
    return next(m.value for m in app.metric if m.label == "Estudiantes")


def test_arranca_con_las_cuatro_tarjetas_y_las_cuatro_pestanas():
    app = iniciar()
    cubo = pd.read_csv(CUBO)
    total = cubo.query("estrato == 'Todos' and naturaleza == 'Todos' and zona == 'Todos'")["n"].item()
    assert estudiantes(app) == numero(total)
    assert {"Estudiantes", "Media del puntaje global", "Mediana", "Brecha no oficial frente a oficial"} <= {m.label for m in app.metric}
    assert [t.label for t in app.tabs] == ["Panorama", "Serie", "Territorio", "Regresión"]
    assert not app.warning


@pytest.mark.parametrize(
    ("clave", "valor"),
    [("departamento", "ANTIOQUIA"), ("estrato", "Estrato 3"), ("naturaleza", "NO OFICIAL"), ("zona", "RURAL")],
)
def test_cada_filtro_responde_sin_errores(clave, valor):
    app = iniciar()
    app.selectbox(key=clave).set_value(valor).run()
    assert not app.exception
    assert not app.warning


def test_estrato_naturaleza_y_zona_responden_con_su_celda_del_cubo():
    app = iniciar()
    app.selectbox(key="estrato").set_value("Estrato 2")
    app.selectbox(key="naturaleza").set_value("OFICIAL")
    app.selectbox(key="zona").set_value("URBANO").run()
    cubo = pd.read_csv(CUBO)
    celda = cubo.query("estrato == 'Estrato 2' and naturaleza == 'OFICIAL' and zona == 'URBANO'")
    assert estudiantes(app) == numero(celda["n"].item())


def test_departamento_y_zona_responden_con_su_archivo_y_sin_brecha():
    app = iniciar()
    app.selectbox(key="departamento").set_value("BOGOTÁ")
    app.selectbox(key="zona").set_value("URBANO").run()
    departamentos = pd.read_csv(DEPARTAMENTO_ZONA)
    celda = departamentos.query("departamento == 'BOGOTÁ' and zona == 'URBANO'")
    assert estudiantes(app) == numero(celda["n"].item())
    assert "Brecha no oficial frente a oficial" not in {m.label for m in app.metric}


def test_rango_de_puntaje_y_periodos():
    app = iniciar()
    app.slider(key="puntaje").set_value((200, 300))
    app.select_slider(key="periodos").set_value((20172, 20224)).run()
    assert not app.exception
    assert any(m.label == "Estudiantes con puntaje entre 200 y 300" for m in app.metric)


@pytest.mark.parametrize("otro", ["estrato", "naturaleza"])
def test_departamento_con_estrato_o_naturaleza_no_esta_publicado(otro):
    app = iniciar()
    app.selectbox(key="departamento").set_value("ANTIOQUIA")
    app.selectbox(key=otro).set_value("Estrato 1" if otro == "estrato" else "NO OFICIAL").run()
    assert not app.exception
    assert len(app.warning) == 1
    assert "no está publicada" in app.warning[0].value
    assert "Estudiantes" not in {m.label for m in app.metric}


def test_regresion_sin_bloques_avisa_en_lugar_de_fallar():
    app = iniciar()
    app.multiselect(key="bloques").set_value([]).run()
    assert not app.exception
    assert any("al menos un bloque" in i.value for i in app.info)
