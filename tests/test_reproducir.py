"""Pruebas de saber11.reproducir: orden de los pasos, ejecución con un corredor falso y recortes."""

import importlib.util

import polars as pl
import pytest

from saber11 import datos, reproducir


def objetivos(lista: list[reproducir.Paso]) -> list[str]:
    """Último argumento de cada comando: el archivo o el módulo sobre el que actúa."""
    return [comando[-1] for paso in lista for comando in paso.comandos]


def test_cada_fase_corre_sus_intermedios_y_su_r_antes_que_su_cuaderno():
    orden = objetivos(reproducir.pasos())
    antes = [
        ("--recortes", "src/r/fase-1.R"),
        ("src/r/fase-1.R", "notebooks/fase-1-eda.ipynb"),
        ("saber11.intermedios", "src/r/fase-2.R"),
        ("src/r/fase-2.R", "notebooks/fase-2-preparacion.ipynb"),
        ("notebooks/fase-2-preparacion.ipynb", "3"),
        ("3", "src/r/fase-3-hipotesis.R"),
        ("src/r/fase-3-multivariante.R", "notebooks/fase-3-modelado.ipynb"),
        ("notebooks/fase-3-modelado.ipynb", "notebooks/fase-4-sintesis.ipynb"),
    ]
    for primero, despues in antes:
        assert orden.index(primero) < orden.index(despues), (primero, despues)
    assert reproducir.pasos()[-1].comandos == (("python", "-m", "pytest", "-q"),)


def test_cada_cuaderno_se_convierte_antes_de_ejecutarse():
    for fase, nombre in reproducir.CUADERNOS.items():
        convertir, ejecutar = reproducir.cuaderno(fase).comandos
        assert convertir[-1] == f"notebooks/{nombre}.py"
        assert ejecutar[-1] == f"notebooks/{nombre}.ipynb"
        assert "--execute" in ejecutar


def test_cada_paso_apunta_a_un_archivo_o_modulo_que_existe():
    for paso in reproducir.pasos(api=True):
        for comando in paso.comandos:
            # La carpeta del portal la crea la descarga.
            rutas = [a for a in comando[1:] if "/" in a and not a.startswith("-") and a != reproducir.CARPETA_PORTAL]
            for ruta in rutas:
                if ruta.endswith(".ipynb"):
                    ruta = ruta.removesuffix(".ipynb") + ".py"
                assert (datos.RAIZ_PROYECTO / ruta).is_file(), ruta
            if comando[:2] == ("python", "-m"):
                assert importlib.util.find_spec(comando[2]) is not None, comando[2]


def test_con_api_la_descarga_va_primero_y_lo_demas_no_cambia():
    sin_api, con_api = reproducir.pasos(), reproducir.pasos(api=True)
    assert con_api[1:] == sin_api
    assert con_api[0].comandos == (
        ("python", "-m", "saber11.datos", "--fuente", "api", "--verificar", "--materializar", "--destino", "data/intermedio/portal"),
        ("python", "-m", "saber11.reproducir", "--portal", "data/intermedio/portal"),
    )


def test_con_api_la_descarga_no_reescribe_los_recortes_de_data():
    # El orden de las filas fija la partición de la fase 2, y el portal las entrega en otro.
    descarga, comparacion = reproducir.pasos(api=True)[0].comandos
    destino = descarga[descarga.index("--destino") + 1]
    assert (datos.RAIZ_PROYECTO / destino).resolve() != datos.DESTINO.resolve()
    assert destino.startswith("data/intermedio/")
    assert comparacion[-2:] == ("--portal", destino)


def test_ejecutar_se_detiene_en_el_primer_paso_que_falla():
    lista = [reproducir.Paso(f"paso {i}", ((f"comando-{i}",),)) for i in range(4)]
    salidas = {"comando-0": 0, "comando-1": 3, "comando-2": 0}
    corridos = []

    def correr(comando):
        corridos.append(comando[0])
        return salidas[comando[0]]

    resultados = reproducir.ejecutar(lista, correr)
    assert corridos == ["comando-0", "comando-1"]
    assert [r.salida for r in resultados] == [0, 3]


def test_un_paso_con_varios_comandos_para_en_el_que_falla():
    paso = reproducir.Paso("cuaderno", (("convertir",), ("ejecutar",)))
    corridos = []

    def correr(comando):
        corridos.append(comando[0])
        return 1

    assert reproducir.correr_paso(paso, correr) == 1
    assert corridos == ["convertir"]


def test_ejecutar_mide_cada_paso_y_el_resumen_suma_el_total():
    lista = [reproducir.Paso("rapido", (("a",),)), reproducir.Paso("lento", (("b",),))]
    marcas = iter([0.0, 5.0, 5.0, 3_730.0])
    resultados = reproducir.ejecutar(lista, lambda comando: 0, reloj=lambda: next(marcas))
    assert [r.segundos for r in resultados] == [5.0, 3_725.0]
    texto = reproducir.resumen(resultados, len(lista))
    assert "1 h 02 min 05 s" in texto
    assert texto.endswith("Total: 1 h 02 min 10 s. 2 de 2 pasos con salida 0.")


@pytest.mark.parametrize(("segundos", "texto"), [(0.4, "0 min 00 s"), (65, "1 min 05 s"), (3_600, "1 h 00 min 00 s")])
def test_duracion(segundos, texto):
    assert reproducir.duracion(segundos) == texto


def test_main_devuelve_la_salida_del_paso_que_falla_y_lista_no_ejecuta_nada(capsys):
    assert reproducir.main([], correr=lambda comando: 2) == 2
    assert "0 de 12 pasos con salida 0" in capsys.readouterr().out

    def no_debe_correr(comando):
        raise AssertionError("--lista no ejecuta nada")

    assert reproducir.main(["--lista"], correr=no_debe_correr) == 0
    assert "12. Pruebas" in capsys.readouterr().out


def test_correr_avisa_si_falta_el_ejecutable(tmp_path, capsys):
    assert reproducir.correr_en(tmp_path)(("ejecutable-que-no-existe",)) == reproducir.SIN_EJECUTABLE
    assert "ejecutable-que-no-existe" in capsys.readouterr().err


def recorte(ruta, periodos: list[int], **extra: list) -> None:
    """Recorte sintético: las columnas del proyecto, todas vacías salvo el periodo."""
    columnas = {c: [None] * len(periodos) for c in datos.COLUMNAS}
    columnas.update(PERIODO=periodos, **extra)
    pl.DataFrame(columnas, schema_overrides={"PERIODO": pl.Int32}).write_parquet(ruta)


@pytest.fixture
def conteos_pequenos(monkeypatch):
    monkeypatch.setattr(datos, "FILAS_UNICAS", {20194: 3, 20224: 3})
    monkeypatch.setattr(datos, "EXCLUIDAS", {20194: 1, 20224: 0})


def test_recortes_que_cuadran_no_dan_problemas(tmp_path, conteos_pequenos):
    recorte(tmp_path / "serie.parquet", [20194, 20194, 20224, 20224, 20224])
    recorte(tmp_path / "modelado.parquet", [20224, 20224, 20224])
    assert reproducir.problemas_de_los_recortes(tmp_path / "serie.parquet", tmp_path / "modelado.parquet") == []


def test_recortes_con_una_fila_de_menos_o_una_columna_de_control_dan_problemas(tmp_path, conteos_pequenos):
    recorte(tmp_path / "serie.parquet", [20194, 20224, 20224, 20224])
    recorte(tmp_path / "modelado.parquet", [20224, 20224, 20224], ESTU_CONSECUTIVO=["X0001", "X0002", "X0003"])
    problemas = reproducir.problemas_de_los_recortes(tmp_path / "serie.parquet", tmp_path / "modelado.parquet")
    assert len(problemas) == 2
    assert problemas[0].startswith("serie.parquet: se esperaban")
    assert problemas[1].startswith("modelado.parquet: las columnas")


@pytest.mark.skipif(not datos.RECORTE_SERIE.exists(), reason="sin los recortes de data/")
def test_los_recortes_del_repositorio_cuadran_con_los_conteos_del_portal():
    assert reproducir.problemas_de_los_recortes() == []


def test_mismas_filas_no_mira_el_orden_pero_si_los_valores_las_repeticiones_y_el_esquema():
    a = pl.DataFrame({"x": [1, 1, 2, None], "y": ["a", "a", "b", "c"]})
    assert reproducir.mismas_filas(a, a.reverse())
    assert not reproducir.mismas_filas(a, a.with_columns(pl.col("x").fill_null(3)))
    assert not reproducir.mismas_filas(a, pl.DataFrame({"x": [1, 2, 2, None], "y": ["a", "b", "b", "c"]}))
    assert not reproducir.mismas_filas(a, a.head(3))
    assert not reproducir.mismas_filas(a, a.with_columns(pl.col("x").cast(pl.Float64)))


@pytest.fixture
def data_y_portal(tmp_path):
    """Recortes sintéticos de data/ y del portal: las mismas filas, en otro orden."""
    data, portal = tmp_path / "data", tmp_path / "portal"
    data.mkdir()
    portal.mkdir()
    recorte(data / "modelado.parquet", [20224] * 3, PUNT_GLOBAL=[200.0, 250.0, 300.0])
    recorte(portal / "modelado.parquet", [20224] * 3, PUNT_GLOBAL=[300.0, 200.0, 250.0])
    recorte(data / "serie.parquet", [20194, 20224], PUNT_GLOBAL=[210.0, 220.0])
    recorte(portal / "serie.parquet", [20224, 20194], PUNT_GLOBAL=[220.0, 210.0])
    return portal, (data / "modelado.parquet", data / "serie.parquet")


def test_el_portal_con_las_mismas_filas_en_otro_orden_no_da_problemas(data_y_portal):
    assert reproducir.problemas_con_el_portal(*data_y_portal) == []


def test_el_portal_con_un_valor_distinto_o_sin_un_recorte_da_problemas(data_y_portal):
    portal, recortes = data_y_portal
    recorte(portal / "serie.parquet", [20194, 20224], PUNT_GLOBAL=[210.0, 221.0])
    assert reproducir.problemas_con_el_portal(portal, recortes) == [f"{portal / 'serie.parquet'}: no trae las mismas filas que serie.parquet de data/"]
    (portal / "modelado.parquet").unlink()
    assert reproducir.problemas_con_el_portal(portal, recortes)[0] == f"{portal / 'modelado.parquet'}: no existe"


def test_main_con_portal_solo_compara_y_sale_con_1_si_hay_problemas(monkeypatch, capsys):
    def no_debe_correr(comando):
        raise AssertionError("--portal no ejecuta pasos")

    monkeypatch.setattr(reproducir, "problemas_con_el_portal", lambda portal: [f"{portal}: no existe"])
    assert reproducir.main(["--portal", "otra"], correr=no_debe_correr) == 1
    assert "ERROR: otra: no existe" in capsys.readouterr().err
    monkeypatch.setattr(reproducir, "problemas_con_el_portal", lambda portal: [])
    assert reproducir.main(["--portal", "otra"], correr=no_debe_correr) == 0
    assert "mismas filas" in capsys.readouterr().out


def test_mismos_recortes_detecta_un_periodo_de_mas(data_y_portal):
    portal, (_, serie) = data_y_portal
    assert reproducir.mismos_recortes(serie, portal / "serie.parquet")
    recorte(portal / "serie.parquet", [20194, 20224, 20142], PUNT_GLOBAL=[210.0, 220.0, 230.0])
    assert not reproducir.mismos_recortes(serie, portal / "serie.parquet")
