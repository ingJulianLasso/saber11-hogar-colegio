"""Pruebas de saber11.limpieza con filas sintéticas en el formato del recorte."""

import math

import pandas as pd
import pytest

from saber11 import datos, limpieza


def fila(consecutivo: str, **valores: object) -> dict[str, object]:
    """Fila completa y válida del recorte; `consecutivo` solo nombra la fila en la prueba."""
    base: dict[str, object] = {columna: "x" for columna in datos.COLUMNAS}
    base.update(
        PERIODO=20224,
        PUNT_GLOBAL=250.0,
        FAMI_ESTRATOVIVIENDA="Estrato 2",
        FAMI_CUARTOSHOGAR="Tres",
        FAMI_PERSONASHOGAR="3 a 4",
        COLE_NATURALEZA="OFICIAL",
        COLE_AREA_UBICACION="URBANO",
        COLE_JORNADA="MAÑANA",
        COLE_CALENDARIO="A",
        COLE_GENERO="MIXTO",
        ESTU_FECHANACIMIENTO="14/02/2004",
        ESTU_CONSECUTIVO=consecutivo,
    )
    base.update(valores)
    return base


def marco(*filas: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(list(filas)).set_index("ESTU_CONSECUTIVO")


def test_armonizar_pasa_cuartos_y_personas_a_numero_y_unifica_la_frase_rural():
    df = marco(
        fila("X0001", FAMI_CUARTOSHOGAR="Seis o mas", FAMI_PERSONASHOGAR="9 o más"),
        fila("X0002", FAMI_CUARTOSHOGAR="Uno", FAMI_PERSONASHOGAR="1 a 2", FAMI_ESTRATOVIVIENDA=limpieza.FRASE_RURAL),
        fila("X0003", FAMI_CUARTOSHOGAR=None, FAMI_PERSONASHOGAR=None, FAMI_ESTRATOVIVIENDA=None),
    )
    armonizado = limpieza.armonizar(df)
    assert armonizado["FAMI_CUARTOSHOGAR"].tolist()[:2] == [6.0, 1.0]
    assert armonizado["FAMI_PERSONASHOGAR"].tolist()[:2] == [9.0, 1.5]
    assert armonizado["FAMI_ESTRATOVIVIENDA"].tolist()[1] == limpieza.SIN_ESTRATO
    assert armonizado.loc["X0003", ["FAMI_CUARTOSHOGAR", "FAMI_PERSONASHOGAR"]].isna().all()
    assert "ESTU_FECHANACIMIENTO" not in armonizado.columns


def test_armonizar_falla_ante_un_valor_sin_equivalencia():
    with pytest.raises(ValueError, match="FAMI_CUARTOSHOGAR"):
        limpieza.armonizar(marco(fila("X0001", FAMI_CUARTOSHOGAR="Siete")))


@pytest.mark.parametrize(
    ("fecha", "edad"),
    [
        ("14/02/2004", 18),
        ("31/12/2004", 18),
        ("01/01/2005", 17),
        ("15/06/1950", 72),
        ("01/01/2012", 10),
    ],
)
def test_edad_cumplida_al_cierre_de_2022(fecha, edad):
    assert limpieza.edad_cumplida(limpieza.fecha_nacimiento(pd.Series([fecha]))).item() == edad


@pytest.mark.parametrize("fecha", ["31/12/1949", "01/01/2013", "02/03/2055", "2004-02-14", "31/02/2004", None])
def test_fechas_imposibles_o_ilegibles_quedan_sin_edad(fecha):
    assert math.isnan(limpieza.edad_cumplida(limpieza.fecha_nacimiento(pd.Series([fecha], dtype="str"))).item())


def test_excluir_cuenta_cada_motivo_y_reinicia_las_posiciones():
    sin_colegio = {c: None for c in limpieza.COLEGIO_BASICO}
    df = marco(fila("X0001"), fila("X0002", **sin_colegio), fila("X0003", PUNT_GLOBAL=0.0), fila("X0004"))
    conservado, exclusiones = limpieza.excluir(df)
    assert [e.filas for e in exclusiones] == [1, 1]
    assert conservado.index.tolist() == [0, 1]
    assert conservado["PUNT_GLOBAL"].min() > 0


def test_limpiar_excluye_antes_de_armonizar():
    df = marco(fila("X0001", PUNT_GLOBAL=0.0, FAMI_CUARTOSHOGAR="Siete"), fila("X0002"))
    limpio, exclusiones = limpieza.limpiar(df)
    assert len(limpio) == 1
    assert sum(e.filas for e in exclusiones) == 1
