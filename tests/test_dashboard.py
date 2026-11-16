"""El dashboard arranca, responde a cada filtro con el agregado que le toca y avisa cuando una
combinación o un cruce no están publicados; sus siete pestañas conservan las seis de las fases 2
y 3, y la de Predicción estima con el modelo del tablero y rechaza lo que la ficha no admite.

No importa `saber11.datos` (polars), para correr también en el entorno liviano del dashboard."""

import json
import shutil
from pathlib import Path

import joblib
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from saber11 import tablero

RAIZ = Path(__file__).resolve().parent.parent
APP = RAIZ / "dashboard" / "app.py"
AGREGADOS = RAIZ / "data" / "agregados"
MODELO = RAIZ / "dashboard" / "modelo"
CUBO = AGREGADOS / "estrato-naturaleza-zona-20224.csv"
DEPARTAMENTO_ZONA = AGREGADOS / "departamento-zona-20224.csv"
PERFILES = AGREGADOS / "cluster-perfiles-20224.csv"
JORNADA = AGREGADOS / "jornada-estrato-naturaleza-20224.csv"
METRICAS = AGREGADOS / "metricas-modelos.csv"
SERIE = AGREGADOS / "serie-estrato-naturaleza.csv"
PESTANAS = ["Resumen", "Exploración", "Pruebas de hipótesis", "Modelos", "Predicción", "Segmentación", "Método y fuente"]
FASES_2_Y_3 = ["Panorama", "Serie", "Territorio", "Regresión", "Pruebas de hipótesis", "Segmentación"]
AVISO_PERIODO = "Periodo 2022-2: el filtro de periodos no aplica."

pytestmark = pytest.mark.skipif(not CUBO.is_file(), reason="faltan los agregados de la fase 2")
fase_3 = pytest.mark.skipif(not PERFILES.is_file(), reason="faltan los agregados de la fase 3")
fase_4 = pytest.mark.skipif(not (METRICAS.is_file() and (MODELO / "ficha.json").is_file()), reason="faltan el agregado de métricas o el modelo del tablero")
jornada = pytest.mark.skipif(not JORNADA.is_file(), reason="el agregado de jornada no está publicado")


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


def metrica(app: AppTest, etiqueta: str) -> str:
    return next(m.value for m in app.metric if m.label == etiqueta)


def pestanas_principales(app: AppTest) -> list[str]:
    contenedor = next(n for n in app.main.children.values() if getattr(n, "type", None) == "tab_container")
    return [pestana.label for pestana in contenedor.children.values()]


def lecturas(app: AppTest) -> list[str]:
    return [m.value for m in app.markdown if m.value.startswith("**Lectura para")]


def test_arranca_con_las_cuatro_tarjetas_y_las_siete_pestanas():
    app = iniciar()
    cubo = pd.read_csv(CUBO)
    total = cubo.query("estrato == 'Todos' and naturaleza == 'Todos' and zona == 'Todos'")["n"].item()
    assert estudiantes(app) == numero(total)
    assert {"Estudiantes", "Media del puntaje global", "Mediana", "Brecha no oficial frente a oficial"} <= {m.label for m in app.metric}
    assert pestanas_principales(app) == PESTANAS
    assert not app.warning


def test_las_seis_pestanas_de_las_fases_2_y_3_siguen_con_su_nombre():
    app = iniciar()
    etiquetas = [t.label for t in app.tabs]
    assert all(etiquetas.count(nombre) == 1 for nombre in FASES_2_Y_3)
    assert etiquetas[etiquetas.index("Exploración") + 1:etiquetas.index("Exploración") + 4] == ["Panorama", "Serie", "Territorio"]
    assert etiquetas[etiquetas.index("Modelos") + 1] == "Regresión"


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


def test_serie_escribe_sus_huecos_y_las_pestanas_de_modelos_avisan_del_periodo():
    app = iniciar()
    anios = set(pd.read_csv(SERIE)["periodo"] // 10)
    faltantes = [f"{a}-2" for a in range(min(anios), max(anios) + 1) if a not in anios]
    assert faltantes
    assert any(all(f in c.value for f in faltantes) for c in app.caption)  # Serie
    assert any(all(f in m.value for f in faltantes) for m in app.markdown)  # Método y fuente
    assert [c.value for c in app.caption].count(AVISO_PERIODO) == 3  # Modelos, Predicción y Segmentación


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
    (copia.parent / "modelo").symlink_to(MODELO)
    for carpeta in ("src", "figuras"):
        (tmp_path / carpeta).symlink_to(RAIZ / carpeta)
    shutil.copytree(AGREGADOS, tmp_path / "data" / "agregados", ignore=shutil.ignore_patterns("cluster-estrato-20224.csv"))
    app = iniciar(copia)
    app.radio(key="reparto").set_value("Estrato").run()
    assert not app.exception
    assert [i.value for i in app.info if "no está publicado" in i.value] == [
        "El cruce de clúster por estrato no está publicado: solo se publican los cruces con 10 o más estudiantes en cada celda, para que nadie pueda ser identificado."
    ]


@fase_4
def test_resumen_muestra_las_metricas_de_prueba_de_la_fase_3():
    app = iniciar()
    metricas = pd.read_csv(METRICAS).set_index("modelo")
    assert metrica(app, "R² de XGBoost") == numero(metricas.loc["xgboostRegresion", "r2"], 3)
    assert metrica(app, "RMSE de XGBoost") == f"{numero(metricas.loc['xgboostRegresion', 'rmse'], 1)} puntos"
    assert metrica(app, "AUC-ROC de XGBoost (naturaleza)") == numero(metricas.loc["xgboostClasificacion", "auc"], 3)


@jornada
@pytest.mark.parametrize("valor", ["MAÑANA", "UNICA"])
def test_jornada_responde_a_su_filtro(valor):
    app = iniciar()
    app.selectbox(key="jornada").set_value(valor)
    app.selectbox(key="naturaleza").set_value("OFICIAL").run()
    assert not app.exception
    celda = pd.read_csv(JORNADA).query("jornada == @valor and estrato == 'Todos' and naturaleza == 'OFICIAL'")
    assert metrica(app, "Estudiantes en la jornada") == numero(celda["n"].item())


@fase_4
def test_leer_como_cambia_los_textos_de_lectura():
    app = iniciar()
    antes = lecturas(app)
    assert len(antes) == 6 and all("la secretaría de educación" in texto for texto in antes)
    app.selectbox(key="audiencia").set_value("Familia").run()
    despues = lecturas(app)
    assert not app.exception
    assert len(despues) == 6 and all("las familias" in texto for texto in despues)
    assert not set(antes) & set(despues)


@fase_4
def test_prediccion_por_el_formulario_coincide_con_el_modelo():
    app = iniciar()
    ficha = json.loads((MODELO / "ficha.json").read_text(encoding="utf-8"))
    modelo = joblib.load(MODELO / ficha["archivo"])
    perfil = tablero.PERFIL_DE_EJEMPLO | {"FAMI_ESTRATOVIVIENDA": "Estrato 4", "FAMI_TIENEINTERNET": None}
    app.selectbox(key="prediccion_FAMI_ESTRATOVIVIENDA").set_value("Estrato 4")
    app.selectbox(key="prediccion_FAMI_TIENEINTERNET").set_value("Sin dato")
    app.button(key="estimar").click().run()
    assert not app.exception and not app.error
    esperado = float(modelo.predict(tablero.perfil(ficha, perfil))[0])
    assert 0 <= esperado <= 500
    assert metrica(app, "Puntaje global esperado del perfil") == numero(esperado, 1)


@fase_4
def test_prediccion_rechaza_una_edad_que_la_ficha_no_admite():
    ficha = json.loads((MODELO / "ficha.json").read_text(encoding="utf-8"))
    edad = next(e for e in ficha["entrada"] if e["columna"] == "ESTU_EDAD")
    fuera = [x for x in range(int(edad["minimo"]), int(edad["maximo"]) + 1) if float(x) not in edad["valores"]]
    if not fuera:
        pytest.skip("la ficha admite todas las edades de su rango")
    app = iniciar()
    app.number_input(key="prediccion_ESTU_EDAD").set_value(fuera[0])
    app.button(key="estimar").click().run()
    assert not app.exception
    assert len(app.error) == 1 and "no es un valor admitido" in app.error[0].value
    assert "Puntaje global esperado del perfil" not in {m.label for m in app.metric}
    app.checkbox(key="prediccion_ESTU_EDAD_sin_dato").check()
    app.button(key="estimar").click().run()
    assert not app.error
    assert "Puntaje global esperado del perfil" in {m.label for m in app.metric}


@fase_4
def test_metodo_y_fuente_enlaza_el_repositorio_y_la_licencia():
    app = iniciar()
    texto = " ".join(m.value for m in app.markdown)
    assert "github.com/ingJulianLasso/saber11-hogar-colegio" in texto
    assert "CC BY-SA 4.0" in texto and "Ley 1581 de 2012" in texto
