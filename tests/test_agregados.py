"""Los agregados que se versionan en data/agregados/ no tienen celdas de menos de 10 estudiantes."""

import polars as pl
import pytest

from saber11 import datos

CARPETA = datos.DESTINO / "agregados"
MINIMO_POR_CELDA = 10
ARCHIVOS = sorted(p for p in CARPETA.glob("*") if p.suffix in {".csv", ".parquet"})


def leer(ruta) -> pl.DataFrame:
    return pl.read_csv(ruta) if ruta.suffix == ".csv" else pl.read_parquet(ruta)


def test_hay_al_menos_un_agregado():
    assert ARCHIVOS


@pytest.mark.parametrize("ruta", ARCHIVOS, ids=lambda ruta: ruta.name)
def test_cada_celda_tiene_al_menos_diez_estudiantes(ruta):
    agregado = leer(ruta)
    assert "n" in agregado.columns
    assert agregado["n"].min() >= MINIMO_POR_CELDA


def test_metricas_modelos_tiene_un_modelo_por_fila_y_los_dos_enfoques_por_tarea():
    metricas = pl.read_csv(CARPETA / "metricas-modelos.csv")
    assert metricas["modelo"].is_unique().all()
    enfoques = metricas.group_by("tarea").agg(pl.col("enfoque").n_unique())
    assert set(enfoques["tarea"]) == {"Regresión", "Clasificación"}
    assert (enfoques["enfoque"] == 2).all()
    assert metricas["n"].n_unique() == 1


@pytest.mark.parametrize("ruta", sorted(CARPETA.glob("jornada-*.csv")), ids=lambda ruta: ruta.name)
def test_el_agregado_de_jornada_es_un_cruce_completo(ruta):
    agregado = pl.read_csv(ruta)
    dimensiones = [columna for columna in agregado.columns if columna in {"jornada", "estrato", "naturaleza", "zona"}]
    assert "jornada" in dimensiones
    celdas = 1
    for dimension in dimensiones:
        celdas *= agregado[dimension].n_unique()
    assert agregado.height == celdas
    assert not agregado.select(dimensiones).is_duplicated().any()
