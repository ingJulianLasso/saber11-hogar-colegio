"""Carga reproducible del Saber 11 (ICFES) para el proyecto integrador.

Fuente: datos.gov.co, recurso ``kgxf-xxbe``, licencia CC BY-SA 4.0. Usa el Parquet
local del dataset si existe; si no, descarga por periodo desde la API. En los dos
casos deduplica por periodo y consecutivo, verifica los conteos contra los
esperados y aplica la regla de ``PUBLICAR`` antes de escribir los recortes.

Uso:
    uv run python -m saber11.datos --verificar
    uv run python -m saber11.datos --materializar
    uv run python -m saber11.datos --fuente api --periodos 20224 --verificar --materializar --destino <carpeta>
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

import polars as pl

from saber11.cifras import en_espanol

RAIZ_PROYECTO = Path(__file__).resolve().parents[2]
PARQUET_LOCAL = RAIZ_PROYECTO.parent / "dataset" / "saber11" / "saber11.parquet"
DESTINO = RAIZ_PROYECTO / "data"
RECORTE_MODELADO = DESTINO / "saber11-20224.parquet"
RECORTE_SERIE = DESTINO / "saber11-serie.parquet"

API = "https://www.datos.gov.co/resource/kgxf-xxbe.csv"
LIMITE_PAGINA = 50_000

PERIODO_MODELADO = 20224
ESTADO_VALIDO = "PUBLICAR"

# Filas publicadas en el portal por periodo, duplicados exactos incluidos. Salen de la
# exportación completa del 3-oct-2026 (CSV con su SHA-256 verificado, tabla "Conteo por
# periodo" de dataset/saber11/README.md); la API dio las mismas 1.065.888 del 20224 el
# 8-nov-2026. Fijarlas aquí deja el diagnóstico de duplicados igual en un clon limpio.
FILAS_PORTAL = {20142: 546_727, 20152: 544_491, 20162: 550_275, 20172: 548_269, 20194: 1_096_524, 20224: 1_065_888}
# Filas únicas por periodo (misma tabla) y filas fuera de PUBLICAR (verificadas en la
# Actividad 1).
FILAS_UNICAS = {20142: 544_535, 20152: 542_450, 20162: 548_206, 20172: 546_261, 20194: 546_211, 20224: 532_792}
EXCLUIDAS = {20142: 171, 20152: 925, 20162: 1, 20172: 557, 20194: 457, 20224: 226}

PUNTAJES = (
    "PUNT_GLOBAL",
    "PUNT_LECTURA_CRITICA",
    "PUNT_MATEMATICAS",
    "PUNT_C_NATURALES",
    "PUNT_SOCIALES_CIUDADANAS",
    "PUNT_INGLES",
)
HOGAR = (
    "FAMI_ESTRATOVIVIENDA",
    "FAMI_EDUCACIONMADRE",
    "FAMI_EDUCACIONPADRE",
    "FAMI_TIENEINTERNET",
    "FAMI_TIENECOMPUTADOR",
    "FAMI_TIENEAUTOMOVIL",
    "FAMI_TIENELAVADORA",
    "FAMI_CUARTOSHOGAR",
    "FAMI_PERSONASHOGAR",
)
COLEGIO = (
    "COLE_NATURALEZA",
    "COLE_AREA_UBICACION",
    "COLE_JORNADA",
    "COLE_CALENDARIO",
    "COLE_CARACTER",
    "COLE_BILINGUE",
    "COLE_GENERO",
    "COLE_COD_DEPTO_UBICACION",
    "COLE_DEPTO_UBICACION",
)
ESTUDIANTE = ("ESTU_GENERO", "ESTU_FECHANACIMIENTO")
COLUMNAS = ("PERIODO", *PUNTAJES, *HOGAR, *COLEGIO, *ESTUDIANTE)
# Solo sirven para deduplicar y filtrar: no salen en los recortes.
CONTROL = ("ESTU_CONSECUTIVO", "ESTU_ESTADOINVESTIGACION")
TEXTO = tuple(c for c in (*COLUMNAS, *CONTROL) if c != "PERIODO" and c not in PUNTAJES)

Obtener = Callable[[str], bytes]


def obtener_http(url: str) -> bytes:
    """Descarga una URL con un reintento, porque la API a veces corta peticiones largas."""
    peticion = urllib.request.Request(url, headers={"User-Agent": "saber11-proyecto/0.1"})
    for intento in (1, 2):
        try:
            with urllib.request.urlopen(peticion, timeout=300) as respuesta:
                return respuesta.read()
        except OSError:
            if intento == 2:
                raise
            time.sleep(10)
    raise AssertionError("inalcanzable")


def tipar_texto_crudo(crudo: pl.DataFrame) -> pl.DataFrame:
    """Tipa un marco leído como texto con el mismo criterio de dataset/saber11/preparar_saber11.py."""
    vacio_a_nulo = [pl.col(c).str.strip_chars().replace("", None) for c in (*PUNTAJES, *TEXTO)]
    return crudo.with_columns(vacio_a_nulo).with_columns(
        pl.col("PERIODO").cast(pl.Int32),
        *[pl.col(c).str.replace(",", ".", literal=True).cast(pl.Float32, strict=True) for c in PUNTAJES],
    ).with_columns(
        pl.when(pl.col("PUNT_INGLES") < 0).then(None).otherwise(pl.col("PUNT_INGLES")).alias("PUNT_INGLES"),
    )


def descargar_periodo(periodo: int, obtener: Obtener = obtener_http) -> pl.DataFrame:
    """Baja un periodo de la API, paginando en un orden estable, y lo devuelve tipado."""
    columnas = ",".join(c.lower() for c in (*COLUMNAS, *CONTROL))
    paginas = []
    desplazamiento = 0
    while True:
        consulta = urllib.parse.urlencode({
            "$select": columnas,
            "$where": f"periodo='{periodo}'",
            "$order": ":id",
            "$limit": LIMITE_PAGINA,
            "$offset": desplazamiento,
        })
        pagina = pl.read_csv(io.BytesIO(obtener(f"{API}?{consulta}")), infer_schema=False)
        paginas.append(pagina)
        desplazamiento += pagina.height
        print(f"{periodo}: {en_espanol(desplazamiento)} filas descargadas", file=sys.stderr)
        if pagina.height < LIMITE_PAGINA:
            break
    crudo = pl.concat(paginas).rename(str.upper).select(*COLUMNAS, *CONTROL)
    return tipar_texto_crudo(crudo)


def leer_local(periodos: Sequence[int], parquet: Path = PARQUET_LOCAL) -> pl.DataFrame:
    """Lee los periodos del Parquet del dataset con el mismo esquema que la descarga."""
    return (
        pl.scan_parquet(parquet)
        .filter(pl.col("PERIODO").is_in(list(periodos)))
        .select(*COLUMNAS, *CONTROL)
        .with_columns(pl.col(c).cast(pl.String) for c in TEXTO)
        .collect()
    )


def cargar(periodos: Sequence[int], fuente: str = "auto", parquet: Path = PARQUET_LOCAL) -> pl.DataFrame:
    """Devuelve las filas crudas de los periodos desde el Parquet local o desde la API."""
    if fuente == "local" or (fuente == "auto" and parquet.exists()):
        return leer_local(periodos, parquet)
    return pl.concat([descargar_periodo(p) for p in periodos])


def deduplicar(df: pl.DataFrame) -> pl.DataFrame:
    """Deja una fila por examinando y periodo: el portal repite filas completas."""
    return df.unique(subset=["PERIODO", "ESTU_CONSECUTIVO"], keep="first", maintain_order=True)


def publicados(df: pl.DataFrame) -> pl.DataFrame:
    """Conserva solo los resultados en firme (estado PUBLICAR)."""
    return df.filter(pl.col("ESTU_ESTADOINVESTIGACION") == ESTADO_VALIDO)


def conteos(unico: pl.DataFrame) -> dict[int, tuple[int, int]]:
    """Filas únicas y filas fuera de PUBLICAR por periodo."""
    resumen = unico.group_by("PERIODO").agg(
        pl.len().alias("unicas"),
        (pl.col("ESTU_ESTADOINVESTIGACION") != ESTADO_VALIDO).sum().alias("excluidas"),
    )
    return {p: (u, e) for p, u, e in resumen.sort("PERIODO").iter_rows()}


def verificar(conteo: dict[int, tuple[int, int]], periodos: Sequence[int]) -> list[str]:
    """Compara los conteos con los esperados; una lista vacía significa que todo cuadra."""
    problemas = []
    for periodo in periodos:
        esperado = (FILAS_UNICAS[periodo], EXCLUIDAS[periodo])
        obtenido = conteo.get(periodo, (0, 0))
        if obtenido != esperado:
            problemas.append(f"{periodo}: se esperaban {esperado} (únicas, excluidas) y hay {obtenido}")
    return problemas


def materializar(publicado: pl.DataFrame, destino: Path = DESTINO) -> list[Path]:
    """Escribe el recorte de modelado y, si están los seis periodos, el de la serie."""
    destino.mkdir(parents=True, exist_ok=True)
    limpio = publicado.select(COLUMNAS).sort("PERIODO", maintain_order=True)
    escritos = []
    if PERIODO_MODELADO in limpio["PERIODO"]:
        ruta = destino / RECORTE_MODELADO.name
        limpio.filter(pl.col("PERIODO") == PERIODO_MODELADO).write_parquet(ruta, compression="zstd")
        escritos.append(ruta)
    if set(limpio["PERIODO"].unique()) == set(FILAS_UNICAS):
        ruta = destino / RECORTE_SERIE.name
        limpio.write_parquet(ruta, compression="zstd")
        escritos.append(ruta)
    return escritos


def imprimir(conteo: dict[int, tuple[int, int]]) -> None:
    print(f"{'Periodo':<8} {'Filas únicas':>13} {'Fuera de PUBLICAR':>18} {'Publicadas':>11}")
    for periodo, (unicas, excluidas) in conteo.items():
        print(f"{periodo:<8} {en_espanol(unicas):>13} {en_espanol(excluidas):>18} {en_espanol(unicas - excluidas):>11}")


def argumentos(argv: Sequence[str] | None) -> argparse.Namespace:
    analizador = argparse.ArgumentParser(prog="python -m saber11.datos", description="Carga reproducible del Saber 11.")
    analizador.add_argument("--verificar", action="store_true", help="verifica los conteos por periodo")
    analizador.add_argument("--materializar", action="store_true", help="escribe los recortes tras verificar")
    analizador.add_argument("--fuente", choices=("auto", "local", "api"), default="auto")
    analizador.add_argument("--periodos", type=int, nargs="+", choices=sorted(FILAS_UNICAS), default=sorted(FILAS_UNICAS))
    analizador.add_argument("--parquet", type=Path, default=PARQUET_LOCAL, help="Parquet local del dataset")
    analizador.add_argument("--destino", type=Path, default=DESTINO, help="carpeta de los recortes")
    args = analizador.parse_args(argv)
    if not (args.verificar or args.materializar):
        analizador.error("indique --verificar, --materializar o ambas")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = argumentos(argv)
    unico = deduplicar(cargar(args.periodos, args.fuente, args.parquet))
    conteo = conteos(unico)
    imprimir(conteo)
    problemas = verificar(conteo, args.periodos)
    for problema in problemas:
        print(f"ERROR: {problema}", file=sys.stderr)
    if problemas:
        return 1
    print("Conteos verificados.")
    if args.materializar:
        for ruta in materializar(publicados(unico), args.destino):
            filas = pl.scan_parquet(ruta).select(pl.len()).collect().item()
            print(f"Escrito {ruta} ({en_espanol(filas)} filas, {ruta.stat().st_size / 2**20:.1f} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
