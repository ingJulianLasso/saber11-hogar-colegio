"""Pruebas de saber11.segmentacion: celdas del plano, asignación de Ward, reglas y reproducibilidad."""

import numpy as np
import pandas as pd
import pytest

from saber11 import segmentacion


@pytest.fixture
def areas() -> pd.DataFrame:
    """Tres grupos de estudiantes con niveles distintos en las cinco áreas."""
    rng = np.random.default_rng(4)
    centros = np.array([[45, 40, 42, 41, 40], [55, 55, 54, 53, 58], [68, 72, 66, 67, 75]], dtype=float)
    etiquetas = rng.choice(3, size=1200, p=[0.4, 0.4, 0.2])
    return pd.DataFrame(centros[etiquetas] + rng.normal(0, 4, (1200, 5)), columns=list(segmentacion.AREAS))


@pytest.fixture
def plano() -> np.ndarray:
    rng = np.random.default_rng(8)
    return np.column_stack([rng.normal(0, 2, 50_000), rng.normal(0, 1, 50_000)])


def test_celdas_del_plano_tienen_diez_o_mas_y_suman_todo(plano):
    etiquetas = (plano[:, 0] > 0).astype(int) + 1
    celdas = segmentacion.celdas_plano(plano, etiquetas)
    assert (celdas["n"] >= 10).all()
    assert celdas["n"].sum() == len(plano)
    assert celdas[["celda_x", "celda_y"]].duplicated().sum() == 0
    assert ((celdas["proporcion"] > 0) & (celdas["proporcion"] <= 1)).all()
    assert (celdas["x_inferior"] < celdas["x_superior"]).all()


def test_celda_lleva_su_cluster_mayoritario(plano):
    etiquetas = np.where(plano[:, 1] > 0.3, 2, 1)
    celdas = segmentacion.celdas_plano(plano, etiquetas)
    bordes_x, bordes_y = segmentacion.rejilla(plano)
    indice_x, indice_y = np.digitize(plano[:, 0], bordes_x[1:-1]), np.digitize(plano[:, 1], bordes_y[1:-1])
    fila = celdas.sort_values("n", ascending=False).iloc[0]
    dentro = (indice_x == fila["celda_x"]) & (indice_y == fila["celda_y"])
    conteo = pd.Series(etiquetas[dentro]).value_counts()
    assert fila["n"] == dentro.sum()
    assert fila["cluster"] == conteo.idxmax()
    assert fila["proporcion"] == pytest.approx(conteo.max() / dentro.sum())


def test_la_rejilla_es_la_mas_fina_que_deja_todas_las_celdas_con_diez_o_mas(plano):
    bordes_x, _ = segmentacion.rejilla(plano)
    elegida = len(bordes_x) - 1
    assert elegida in segmentacion.CELDAS_POR_EJE
    for por_eje in [c for c in segmentacion.CELDAS_POR_EJE if c > elegida]:
        with pytest.raises(ValueError):
            segmentacion.rejilla(plano, celdas=[por_eje])


def test_la_rejilla_va_de_los_percentiles_dos_a_noventa_y_ocho(plano):
    assert segmentacion.PERCENTILES_REJILLA == (2, 98)
    bordes_x, bordes_y = segmentacion.rejilla(plano)
    for bordes, eje in ((bordes_x, 0), (bordes_y, 1)):
        assert bordes[0] == pytest.approx(np.percentile(plano[:, eje], 2))
        assert bordes[-1] == pytest.approx(np.percentile(plano[:, eje], 98))
        assert np.allclose(np.diff(bordes), np.diff(bordes)[0])


def test_las_colas_largas_piden_los_percentiles_dos_y_noventa_y_ocho():
    """Con colas como las del plano real, los percentiles 0,5 y 99,5 dejan celdas con menos de 10."""
    rng = np.random.default_rng(15)
    colas = np.column_stack([rng.standard_t(3, 60_000), rng.standard_t(3, 60_000)])
    with pytest.raises(ValueError):
        segmentacion.rejilla(colas, percentiles=(0.5, 99.5))
    bordes_x, bordes_y = segmentacion.rejilla(colas)
    assert segmentacion.menor_celda(colas, bordes_x, bordes_y) >= 10


def test_menor_celda_cuenta_la_celda_no_vacia_mas_pequena():
    plano = np.array([[0.1, 0.1]] * 3 + [[0.9, 0.9]] * 12 + [[0.1, 0.9]] * 5 + [[0.0, 0.0], [1.0, 1.0]])
    bordes = np.array([0.0, 0.5, 1.0])
    assert segmentacion.menor_celda(plano, bordes, bordes) == 4


def test_perfil_relativo_quita_el_nivel_de_cada_fila(areas):
    z = segmentacion.estandarizar(areas)
    relativo = segmentacion.perfil_relativo(z)
    assert np.allclose(relativo.sum(axis=1), 0)
    assert np.allclose(segmentacion.perfil_relativo(z + 3.0), relativo)


def test_los_bordes_exteriores_estan_abiertos(plano):
    extremo = np.vstack([plano, [[1e6, 0.0]]])
    celdas = segmentacion.celdas_plano(extremo, np.ones(len(extremo), dtype=int))
    assert celdas["n"].sum() == len(extremo)


def test_ward_asigna_cada_fila_al_centroide_mas_cercano(areas):
    z = segmentacion.estandarizar(areas)
    muestra = z.iloc[:300]
    etiquetas = segmentacion.cortar(segmentacion.ward(muestra), 3)
    asignadas = segmentacion.asignar_por_centroides(z, muestra, etiquetas)
    centroides = muestra.groupby(etiquetas).mean()
    distancias = ((z.to_numpy()[:, None, :] - centroides.to_numpy()[None]) ** 2).sum(axis=2)
    assert (asignadas == centroides.index.to_numpy()[distancias.argmin(axis=1)]).all()
    assert (asignadas[:300] == etiquetas).mean() > 0.95


def test_segmentar_es_reproducible_y_numera_por_nivel(areas):
    z = segmentacion.estandarizar(areas)
    en_muestra = np.arange(len(z)) % 3 == 0
    primera, segunda = segmentacion.segmentar(z, en_muestra, 3), segmentacion.segmentar(z, en_muestra, 3)
    pd.testing.assert_frame_equal(primera, segunda)
    nivel = z.mean(axis=1)
    for columna in ("kmeans", "ward"):
        medias = nivel.groupby(primera[columna]).mean()
        assert list(medias.index) == [1, 2, 3]
        assert medias.is_monotonic_increasing


def test_barrido_es_reproducible_y_la_silueta_elige_tres(areas):
    z = segmentacion.estandarizar(areas)
    en_muestra = np.arange(len(z)) % 2 == 0
    barrido = segmentacion.barrido_kmeans(z, en_muestra, range(2, 6))
    pd.testing.assert_frame_equal(barrido, segmentacion.barrido_kmeans(z, en_muestra, range(2, 6)))
    assert barrido["inercia"].is_monotonic_decreasing
    assert segmentacion.k_por_silueta(barrido) in (2, 3)


def test_empate_de_silueta_elige_el_menor_k():
    barrido = pd.DataFrame({"k": [2, 3, 4, 5], "inercia": [9.0, 5.0, 4.0, 3.5], "silueta": [0.40, 0.495, 0.50, 0.30]})
    assert segmentacion.k_por_silueta(barrido) == 3
    assert segmentacion.k_por_silueta(barrido, tolerancia=0.001) == 4


def test_codo_y_eps_toman_el_punto_mas_lejano_de_la_recta():
    barrido = pd.DataFrame({"k": range(2, 11), "inercia": [100, 60, 35, 30, 27, 25, 24, 23, 22]})
    assert segmentacion.codo(barrido) == 4
    escalada = barrido.assign(inercia=barrido["inercia"] * 1e6)
    assert segmentacion.codo(escalada) == 4
    distancias = np.array([0.1, 0.11, 0.12, 0.13, 0.15, 0.2, 0.9, 2.0])
    assert segmentacion.eps_por_k_distancias(distancias) == pytest.approx(0.2)


def test_hojas_del_dendrograma_tienen_el_minimo(areas):
    enlace = segmentacion.ward(segmentacion.estandarizar(areas).iloc[:400])
    hojas = segmentacion.hojas_truncadas(enlace)
    from scipy.cluster.hierarchy import fcluster

    assert np.bincount(fcluster(enlace, hojas, criterion="maxclust"))[1:].min() >= 10
    assert 2 <= hojas <= segmentacion.HOJAS_MAXIMAS


def test_pca_orienta_las_componentes_y_ari_compara_pares(areas):
    z = segmentacion.estandarizar(areas)
    modelo = segmentacion.pca(z)
    assert (modelo.components_.sum(axis=1) >= 0).all()
    assert np.allclose(modelo.transform(z)[:, 0], (z.to_numpy() - modelo.mean_) @ modelo.components_[0])
    etiquetas = segmentacion.segmentar(z, np.ones(len(z), dtype=bool), 3)
    tabla = segmentacion.ari({"kmeans": etiquetas["kmeans"], "ward": etiquetas["ward"], "igual": etiquetas["kmeans"]})
    assert len(tabla) == 3
    assert tabla.loc[tabla["b"].eq("igual") & tabla["a"].eq("kmeans"), "ari"].item() == pytest.approx(1.0)
