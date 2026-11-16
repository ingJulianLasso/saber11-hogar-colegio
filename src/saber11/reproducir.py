"""Reproduce el proyecto completo, en el orden del README, y mide el tiempo de cada paso.

Necesita el entorno instalado (`uv sync --locked` y `Rscript src/r/instalar.R`). Primero
comprueba que los recortes de `data/` tengan, periodo por periodo, las filas que deja el
portal tras quitar duplicados y resultados fuera de `PUBLICAR`; después corre cada fase en
orden (intermedios, scripts de R y cuaderno, que jupytext convierte a `.ipynb` y nbconvert
ejecuta en su lugar), el cuaderno de síntesis de la fase 4 y las pruebas. Con `--api`, antes
baja los seis periodos del portal de datos abiertos a `data/intermedio/portal/`, verifica sus
conteos y comprueba que traen las mismas filas que los recortes de `data/`. Estos no se
reescriben: el portal entrega las filas en otro orden, y la partición de la fase 2 se hace por
la posición de cada fila. Se detiene en el primer paso que falla, porque cada uno usa la salida
de los anteriores, y cierra con el tiempo de cada paso.

Uso, desde la raíz del repositorio:
    uv run python -m saber11.reproducir                    # todo, con los recortes de data/
    uv run python -m saber11.reproducir --api              # además, los datos desde el portal
    uv run python -m saber11.reproducir --lista            # solo imprime los pasos
    uv run python -m saber11.reproducir --recortes         # solo comprueba los recortes
    uv run python -m saber11.reproducir --portal <carpeta> # solo los compara con los bajados
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from saber11 import datos
from saber11.cifras import en_espanol

CUADERNOS = {1: "fase-1-eda", 2: "fase-2-preparacion", 3: "fase-3-modelado", 4: "fase-4-sintesis"}
# Código de salida de la consola cuando no encuentra el ejecutable.
SIN_EJECUTABLE = 127
# Carpeta, ignorada por git, a la que --api baja los recortes del portal, relativa a la raíz.
CARPETA_PORTAL = "data/intermedio/portal"

Comando = tuple[str, ...]
Correr = Callable[[Comando], int]


@dataclass(frozen=True)
class Paso:
    nombre: str
    comandos: tuple[Comando, ...]


@dataclass(frozen=True)
class Resultado:
    paso: Paso
    salida: int
    segundos: float


def python(*argumentos: str) -> Comando:
    """Comando de Python; al ejecutarlo se usa el intérprete del entorno actual."""
    return ("python", *argumentos)


def rscript(script: str) -> Comando:
    return ("Rscript", f"src/r/{script}")


def cuaderno(fase: int) -> Paso:
    """Convierte el cuaderno de la fase desde su fuente jupytext y lo ejecuta en su lugar."""
    nombre = CUADERNOS[fase]
    return Paso(f"Cuaderno de la fase {fase}", (
        python("-m", "jupytext", "--to", "ipynb", f"notebooks/{nombre}.py"),
        python("-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
               "--ExecutePreprocessor.timeout=-1", f"notebooks/{nombre}.ipynb"),
    ))


def pasos(api: bool = False) -> list[Paso]:
    """Pasos en el orden del README; con `api`, la descarga del portal y su comparación van primero."""
    descarga = [Paso("Datos desde el portal", (
        python("-m", "saber11.datos", "--fuente", "api", "--verificar", "--materializar", "--destino", CARPETA_PORTAL),
        python("-m", "saber11.reproducir", "--portal", CARPETA_PORTAL),
    ))]
    return [
        *(descarga if api else []),
        Paso("Conteos de los recortes de data/", (python("-m", "saber11.reproducir", "--recortes"),)),
        Paso("R de la fase 1", (rscript("fase-1.R"),)),
        cuaderno(1),
        Paso("Intermedios de la fase 2", (python("-m", "saber11.intermedios"),)),
        Paso("R de la fase 2", (rscript("fase-2.R"),)),
        cuaderno(2),
        Paso("Intermedios de la fase 3", (python("-m", "saber11.intermedios", "--fase", "3"),)),
        Paso("R de la fase 3: hipótesis", (rscript("fase-3-hipotesis.R"),)),
        Paso("R de la fase 3: multivariante", (rscript("fase-3-multivariante.R"),)),
        cuaderno(3),
        cuaderno(4),
        Paso("Pruebas", (python("-m", "pytest", "-q"),)),
    ]


def correr_en(raiz: Path) -> Correr:
    """Ejecuta cada comando desde `raiz`, con la salida a la vista, y devuelve su código."""
    def correr(comando: Comando) -> int:
        ejecutable = sys.executable if comando[0] == "python" else comando[0]
        try:
            return subprocess.run([ejecutable, *comando[1:]], cwd=raiz).returncode
        except FileNotFoundError:
            print(f"No se encontró {comando[0]!r}: instálelo y agréguelo a la ruta de ejecutables (PATH).", file=sys.stderr)
            return SIN_EJECUTABLE
    return correr


def correr_paso(paso: Paso, correr: Correr) -> int:
    for comando in paso.comandos:
        print(f"$ {' '.join(comando)}", flush=True)
        salida = correr(comando)
        if salida != 0:
            return salida
    return 0


def duracion(segundos: float) -> str:
    minutos, resto = divmod(round(segundos), 60)
    horas, minutos = divmod(minutos, 60)
    return f"{horas} h {minutos:02d} min {resto:02d} s" if horas else f"{minutos} min {resto:02d} s"


def ejecutar(lista: Sequence[Paso], correr: Correr, reloj: Callable[[], float] = time.perf_counter) -> list[Resultado]:
    """Corre los pasos en orden y se detiene en el primero que no sale con 0."""
    resultados = []
    for numero, paso in enumerate(lista, start=1):
        print(f"\n== Paso {numero} de {len(lista)}: {paso.nombre}", flush=True)
        inicio = reloj()
        salida = correr_paso(paso, correr)
        resultados.append(Resultado(paso, salida, reloj() - inicio))
        print(f"== Paso {numero} de {len(lista)}: salida {salida} en {duracion(resultados[-1].segundos)}", flush=True)
        if salida != 0:
            break
    return resultados


def resumen(resultados: Sequence[Resultado], total_de_pasos: int) -> str:
    ancho = max(len(r.paso.nombre) for r in resultados)
    lineas = ["", "== Resumen"]
    for numero, r in enumerate(resultados, start=1):
        lineas.append(f"{numero:>2}. {r.paso.nombre:<{ancho}}  salida {r.salida}  {duracion(r.segundos)}")
    correctos = sum(r.salida == 0 for r in resultados)
    lineas.append(f"Total: {duracion(sum(r.segundos for r in resultados))}. {correctos} de {total_de_pasos} pasos con salida 0.")
    return "\n".join(lineas)


def publicadas_esperadas() -> dict[int, int]:
    """Filas en estado PUBLICAR por periodo: las únicas del portal menos las que quedan fuera."""
    return {periodo: datos.FILAS_UNICAS[periodo] - datos.EXCLUIDAS[periodo] for periodo in sorted(datos.FILAS_UNICAS)}


def filas_por_periodo(ruta: Path) -> dict[int, int]:
    return dict(pl.scan_parquet(ruta).group_by("PERIODO").len().collect().iter_rows())


def problemas_de_los_recortes(serie: Path = datos.RECORTE_SERIE, modelado: Path = datos.RECORTE_MODELADO) -> list[str]:
    """Compara columnas y filas por periodo de los dos recortes; una lista vacía significa que cuadran."""
    esperadas = publicadas_esperadas()
    del_modelado = {datos.PERIODO_MODELADO: esperadas[datos.PERIODO_MODELADO]}
    problemas = []
    for ruta, esperado in ((serie, esperadas), (modelado, del_modelado)):
        columnas = pl.scan_parquet(ruta).collect_schema().names()
        if columnas != list(datos.COLUMNAS):
            problemas.append(f"{ruta.name}: las columnas no son las {len(datos.COLUMNAS)} del recorte")
        obtenido = filas_por_periodo(ruta)
        if obtenido != esperado:
            problemas.append(f"{ruta.name}: se esperaban {esperado} filas por periodo y hay {obtenido}")
    return problemas


def comprobar_recortes() -> int:
    en_la_serie = filas_por_periodo(datos.RECORTE_SERIE)
    print(f"{'Periodo':<8} {'Filas únicas':>13} {'Fuera de PUBLICAR':>18} {'Publicadas':>11} {'En la serie':>12}")
    for periodo, publicadas in publicadas_esperadas().items():
        cifras = (datos.FILAS_UNICAS[periodo], datos.EXCLUIDAS[periodo], publicadas, en_la_serie.get(periodo, 0))
        print(f"{periodo:<8} " + " ".join(f"{en_espanol(c):>{a}}" for c, a in zip(cifras, (13, 18, 11, 12))))
    problemas = problemas_de_los_recortes()
    for problema in problemas:
        print(f"ERROR: {problema}", file=sys.stderr)
    if problemas:
        return 1
    print("Los recortes de data/ cuadran con los conteos del portal.")
    return 0


def mismas_filas(a: pl.DataFrame, b: pl.DataFrame) -> bool:
    """Mismo esquema y las mismas filas, cada una repetida las mismas veces, sin importar el orden."""
    if a.schema != b.schema or a.height != b.height:
        return False
    return a.sort(a.columns, nulls_last=True).equals(b.sort(b.columns, nulls_last=True))


def filas_del_periodo(ruta: Path, periodo: int) -> pl.DataFrame:
    return pl.scan_parquet(ruta).filter(pl.col("PERIODO") == periodo).collect()


def mismos_recortes(recorte: Path, bajado: Path) -> bool:
    """Las mismas filas en los dos archivos; compara periodo por periodo para no ordenar la serie entera en memoria."""
    periodos = filas_por_periodo(recorte)
    return filas_por_periodo(bajado) == periodos and all(
        mismas_filas(filas_del_periodo(recorte, periodo), filas_del_periodo(bajado, periodo)) for periodo in periodos
    )


def problemas_con_el_portal(portal: Path, recortes: Sequence[Path] = (datos.RECORTE_MODELADO, datos.RECORTE_SERIE)) -> list[str]:
    """Compara los recortes bajados a `portal` con los de data/; una lista vacía significa que traen las mismas filas."""
    problemas = []
    for recorte in recortes:
        bajado = portal / recorte.name
        if not bajado.is_file():
            problemas.append(f"{bajado}: no existe")
        elif not mismos_recortes(recorte, bajado):
            problemas.append(f"{bajado}: no trae las mismas filas que {recorte.name} de data/")
    return problemas


def comprobar_portal(portal: Path) -> int:
    problemas = problemas_con_el_portal(portal)
    for problema in problemas:
        print(f"ERROR: {problema}", file=sys.stderr)
    if problemas:
        return 1
    print("Los recortes del portal traen las mismas filas que los de data/. Los pasos siguientes usan los de data/, "
          "porque la partición de la fase 2 se hace por la posición de cada fila y el portal puede entregarlas en otro orden.")
    return 0


def argumentos(argv: Sequence[str] | None) -> argparse.Namespace:
    analizador = argparse.ArgumentParser(prog="python -m saber11.reproducir", description="Reproduce el proyecto completo y mide cada paso.")
    analizador.add_argument("--api", action="store_true", help=f"antes, baja los seis periodos del portal a {CARPETA_PORTAL}/, verifica sus conteos y comprueba que traen las mismas filas que los recortes de data/")
    modo = analizador.add_mutually_exclusive_group()
    modo.add_argument("--lista", action="store_true", help="solo imprime los pasos y sus comandos")
    modo.add_argument("--recortes", action="store_true", help="solo comprueba los recortes de data/")
    modo.add_argument("--portal", type=Path, metavar="CARPETA", help="solo comprueba que los recortes bajados a CARPETA traen las mismas filas que los de data/")
    return analizador.parse_args(argv)


def main(argv: Sequence[str] | None = None, correr: Correr | None = None) -> int:
    args = argumentos(argv)
    if args.recortes:
        return comprobar_recortes()
    if args.portal:
        return comprobar_portal(args.portal)
    lista = pasos(args.api)
    if args.lista:
        for numero, paso in enumerate(lista, start=1):
            print(f"{numero:>2}. {paso.nombre}")
            for comando in paso.comandos:
                print(f"    {' '.join(comando)}")
        return 0
    resultados = ejecutar(lista, correr or correr_en(datos.RAIZ_PROYECTO))
    print(resumen(resultados, len(lista)))
    return resultados[-1].salida


if __name__ == "__main__":
    sys.exit(main())
