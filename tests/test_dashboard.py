"""El dashboard arranca, responde a cada filtro con el agregado que le toca y avisa cuando una
combinación o un cruce no están publicados."""

import shutil

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from saber11 import datos

APP = datos.RAIZ_PROYECTO / "dashboard" / "app.py"
AGREGADOS = datos.DESTINO / "agregados"
CUBO = AGREGADOS / "estrato-naturaleza-zona-20224.csv"
DEPARTAMENTO_ZONA = AGREGADOS / "departamento-zona-20224.csv"
PERFILES = AGREGADOS / "cluster-perfiles-20224.csv"
PESTANAS = ["Panorama", "Serie", "Territorio", "Regresión", "Pruebas de hipótesis", "Segmentación"]

pytestmark = pytest.mark.skipif(not CUBO.is_file(), reason="faltan los agregados de la fase 2")
fase_3 = pytest.mark.skipif(not PERFILES.is_file(), reason="faltan los agregados de la fase 3")


def iniciar(app_py=APP) -> AppTest:
    app = AppTest.from_file(str(app_py), default_timeout=60)
    app.run()
    assert not app.exception
    return app


def numero(valor: float, decimales: int = 0) -> str:
    texto = f"{valor:,.{decimales}f}" if valor >= 10_000 else f"{valor:.{decimales}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


def estudiantes(app: AppTest) -> str:
    return next(m.value for m in app.metric if m.label == "Estudiantes")


def test_arranca_con_las_cuatro_tarjetas_y_las_seis_pestanas():
    app = iniciar()
    cubo = pd.read_csv(CUBO)
    total = cubo.query("estrato == 'Todos' and naturaleza == 'Todos' and zona == 'Todos'")["n"].item()
    assert estudiantes(app) == numero(total)
    assert {"Estudiantes", "Media del puntaje global", "Mediana", "Brecha no oficial frente a oficial"} <= {m.label for m in app.metric}
    assert [t.label for t in app.tabs] == PESTANAS
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


@fase_3
def test_pruebas_de_hipotesis_responden_a_sus_controles():
    app = iniciar()
    assert not app.info, [i.value for i in app.info]
    app.radio(key="hipotesis_medias").set_value("H2")
    app.multiselect(key="metodos_tukey").set_value(["Games-Howell"])
    app.checkbox(key="consecutivos").uncheck()
    app.radio(key="tabla_h4").set_value("internet_zona").run()
    assert not app.exception
    assert not app.warning


@fase_3
def test_tukey_sin_metodo_avisa_en_lugar_de_fallar():
    app = iniciar()
    app.multiselect(key="metodos_tukey").set_value([]).run()
    assert not app.exception
    assert any("al menos un método" in i.value for i in app.info)


@fase_3
def test_segmentacion_da_la_parte_de_los_estudiantes_en_cada_cluster():
    app = iniciar()
    perfiles = pd.read_csv(PERFILES)
    tarjetas = {m.label: m.value for m in app.metric}
    for _, perfil in perfiles.iterrows():
        assert tarjetas[f"Clúster {int(perfil['cluster'])}"] == f"{numero(perfil['porcentaje'], 1)} %"


@fase_3
@pytest.mark.parametrize("dimension", ["Estrato", "Zona", "Naturaleza del colegio"])
def test_segmentacion_responde_a_sus_controles(dimension):
    app = iniciar()
    app.radio(key="escala_perfiles").set_value("Estandarizado")
    app.radio(key="reparto").set_value(dimension).run()
    assert not app.exception
    assert not app.info, [i.value for i in app.info]


@fase_3
def test_un_cruce_no_publicado_se_avisa_en_lugar_de_fallar(tmp_path):
    """Con una copia de la app y de los agregados sin el cruce de clúster por estrato, la pestaña lo dice."""
    copia = tmp_path / "dashboard" / "app.py"
    copia.parent.mkdir()
    shutil.copy(APP, copia)
    shutil.copytree(AGREGADOS, tmp_path / "data" / "agregados", ignore=shutil.ignore_patterns("cluster-estrato-20224.csv"))
    app = iniciar(copia)
    app.radio(key="reparto").set_value("Estrato").run()
    assert not app.exception
    assert [i.value for i in app.info if "no está publicado" in i.value] == [
        "El cruce de clúster por estrato no está publicado: solo se publican los cruces con 10 o más estudiantes en cada celda, para que nadie pueda ser identificado."
    ]
