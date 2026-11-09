"""Pruebas de saber11.datos con datos sintéticos y, si existe, con el Parquet local."""

import polars as pl
import pytest

from saber11 import datos


def fila(consecutivo: str, estado: str = "PUBLICAR", periodo: str = "20224", **valores: str) -> dict[str, str]:
    """Fila cruda, toda en texto, como la entrega la API."""
    base = {c: "" for c in (*datos.COLUMNAS, *datos.CONTROL)}
    base.update(PERIODO=periodo, ESTU_CONSECUTIVO=consecutivo, ESTU_ESTADOINVESTIGACION=estado, PUNT_GLOBAL="250")
    base.update(valores)
    return base


def marco(*filas: dict[str, str]) -> pl.DataFrame:
    return datos.tipar_texto_crudo(pl.DataFrame(list(filas)))


def csv_api(*filas: dict[str, str]) -> bytes:
    """Página CSV con los encabezados en minúscula, como la sirve datos.gov.co."""
    return pl.DataFrame(list(filas)).rename(str.lower).write_csv().encode()


def test_deduplicar_deja_una_fila_por_examinando_y_periodo():
    df = marco(fila("X0001"), fila("X0001"), fila("X0002"), fila("X0001", periodo="20194"))
    unico = datos.deduplicar(df)
    assert unico.height == 3
    assert sorted(unico["ESTU_CONSECUTIVO"].to_list()) == ["X0001", "X0001", "X0002"]


def test_publicados_descarta_los_estados_que_no_son_publicar():
    df = marco(fila("X0001"), fila("X0002", estado="VALIDEZ OFICINA JURÍDICA"), fila("X0003"))
    assert datos.publicados(df)["ESTU_CONSECUTIVO"].to_list() == ["X0001", "X0003"]


def test_tipar_texto_crudo_corrige_coma_decimal_vacios_e_ingles_negativo():
    df = marco(fila("X0001", PUNT_MATEMATICAS="36,44", PUNT_INGLES="-1", FAMI_ESTRATOVIVIENDA="  "))
    assert df["PERIODO"].dtype == pl.Int32
    assert df["PUNT_MATEMATICAS"].item() == pytest.approx(36.44)
    assert df["PUNT_INGLES"].item() is None
    assert df["FAMI_ESTRATOVIVIENDA"].item() is None


def test_conteos_y_verificar_detectan_un_desfase():
    df = marco(fila("X0001"), fila("X0002", estado="NO SE COMPROBO IDENTIDAD DEL EXAMINADO"))
    conteo = datos.conteos(datos.deduplicar(df))
    assert conteo == {20224: (2, 1)}
    assert datos.verificar(conteo, [20224]) != []


def test_descargar_periodo_pagina_hasta_una_pagina_incompleta(monkeypatch):
    monkeypatch.setattr(datos, "LIMITE_PAGINA", 2)
    paginas = [csv_api(fila("X0001"), fila("X0002")), csv_api(fila("X0003"))]
    pedidas = []

    def obtener(url: str) -> bytes:
        pedidas.append(url)
        return paginas[len(pedidas) - 1]

    df = datos.descargar_periodo(20224, obtener)
    assert df["ESTU_CONSECUTIVO"].to_list() == ["X0001", "X0002", "X0003"]
    assert len(pedidas) == 2
    assert "%24offset=2" in pedidas[1]
    assert df.columns == [*datos.COLUMNAS, *datos.CONTROL]


def test_materializar_quita_las_columnas_de_control(tmp_path):
    df = marco(fila("X0001"), fila("X0002"))
    escritos = datos.materializar(datos.publicados(df), tmp_path)
    assert [r.name for r in escritos] == [datos.RECORTE_MODELADO.name]
    recorte = pl.read_parquet(escritos[0])
    assert recorte.columns == list(datos.COLUMNAS)
    assert recorte.height == 2


@pytest.mark.skipif(not datos.PARQUET_LOCAL.exists(), reason="sin el Parquet local del dataset")
def test_conteos_del_parquet_local_coinciden_con_los_esperados():
    periodos = sorted(datos.FILAS_UNICAS)
    conteo = datos.conteos(datos.deduplicar(datos.leer_local(periodos)))
    assert datos.verificar(conteo, periodos) == []
