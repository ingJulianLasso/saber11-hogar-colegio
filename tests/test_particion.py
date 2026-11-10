"""Pruebas de saber11.particion: tamaños, estratificación, huella y muestra."""

import json

import numpy as np
import pandas as pd
import pytest

from saber11 import datos, intermedios, particion
from saber11 import variables as v

RUTA_CIFRAS = datos.RAIZ_PROYECTO / "resultados" / "cifras-fase-2.json"


@pytest.fixture
def estrato() -> pd.Series:
    rng = np.random.default_rng(1)
    return pd.Series(rng.choice(["OFICIAL", "NO OFICIAL"], size=10_000, p=[0.77, 0.23]))


def test_tamanos_70_15_15_con_una_fila_de_tolerancia(estrato):
    conteo = particion.particionar(estrato).value_counts()
    for conjunto, proporcion in ((particion.ENTRENAMIENTO, 0.70), (particion.VALIDACION, 0.15), (particion.PRUEBA, 0.15)):
        assert abs(conteo[conjunto] - proporcion * len(estrato)) <= 1


def test_la_proporcion_de_no_oficiales_difiere_menos_de_una_decima_de_punto(estrato):
    conjuntos = particion.particionar(estrato)
    proporciones = estrato.eq("NO OFICIAL").groupby(conjuntos).mean() * 100
    assert proporciones.max() - proporciones.min() < 0.1


def test_los_conjuntos_son_disjuntos_y_cubren_todo(estrato):
    conjuntos = particion.particionar(estrato)
    assert conjuntos.notna().all()
    assert set(conjuntos) == {particion.ENTRENAMIENTO, particion.VALIDACION, particion.PRUEBA}
    assert conjuntos.value_counts().sum() == len(estrato)


def test_la_misma_semilla_da_la_misma_huella_y_otra_semilla_otra(estrato):
    huella = particion.huella(particion.particionar(estrato))
    assert huella == particion.huella(particion.particionar(estrato))
    assert huella != particion.huella(particion.particionar(estrato, semilla=7))
    assert len(huella) == 64


def test_la_muestra_sale_solo_de_entrenamiento_y_conserva_la_proporcion(estrato):
    conjuntos = particion.particionar(estrato)
    muestra = particion.muestra_estratificada(conjuntos, estrato, 2000)
    assert muestra.sum() == 2000
    assert conjuntos[muestra].eq(particion.ENTRENAMIENTO).all()
    assert abs(estrato[muestra].eq("NO OFICIAL").mean() - estrato[conjuntos.eq(particion.ENTRENAMIENTO)].eq("NO OFICIAL").mean()) < 0.001


def test_escribir_guarda_posicion_conjunto_y_muestra(estrato, tmp_path):
    conjuntos = particion.particionar(estrato)
    muestra = particion.muestra_estratificada(conjuntos, estrato, 500)
    leido = pd.read_parquet(particion.escribir(conjuntos, muestra, tmp_path / "particion.parquet"))
    assert leido.columns.tolist() == ["posicion", "conjunto", "muestra"]
    assert leido["posicion"].tolist() == list(range(len(estrato)))
    assert particion.huella(leido["conjunto"]) == particion.huella(conjuntos)


def _huella_publicada() -> str | None:
    if not RUTA_CIFRAS.is_file():
        return None
    return json.loads(RUTA_CIFRAS.read_text(encoding="utf-8")).get("huellaParticion", {}).get("valor")


@pytest.mark.skipif(not datos.RECORTE_MODELADO.is_file() or _huella_publicada() is None, reason="sin el recorte o sin las cifras de la fase 2")
def test_la_huella_del_recorte_real_coincide_con_la_publicada():
    limpio, _ = intermedios.cargar_limpio()
    assert particion.huella(particion.particionar(limpio[v.OBJETIVO_CLASIFICACION])) == _huella_publicada()
