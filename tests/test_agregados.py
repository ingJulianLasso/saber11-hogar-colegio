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
