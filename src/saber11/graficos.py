"""Estilo gráfico, registro de figuras y medición de herramientas de los cuadernos.

El estilo es el de la fase 1: fondo blanco, rejilla mínima, tipografía con serifa y la
paleta categórica de Okabe e Ito, legible también para personas con daltonismo.
"""

from __future__ import annotations

import ast
import inspect
import statistics
import tempfile
import textwrap
import time
from collections.abc import Callable
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.ticker import FuncFormatter

from saber11.cifras import en_espanol

PALETA_CATEGORICA = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00", "#4C4C4C")
COLOR_PRINCIPAL = "#2C4A6E"
COLOR_APOYO = "#7F7F7F"
COLOR_OFICIAL, COLOR_NO_OFICIAL = PALETA_CATEGORICA[0], PALETA_CATEGORICA[1]
ESTILO = {
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.edgecolor": "#4C4C4C",
    "axes.linewidth": 0.7,
    "grid.color": "#D9D9D9",
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.prop_cycle": mpl.cycler(color=list(PALETA_CATEGORICA)),
}
# Las figuras de un informe IEEE se dibujan al ancho de una columna, para que su letra se
# imprima al tamaño con que se escribe, sin reducir la imagen: 6,5 puntos en marcas, ejes y
# leyendas y 7 en los títulos, lo que deja el informe dentro de su límite de páginas.
ANCHO_COLUMNA = 3.45
ESTILO_INFORME = {
    "font.size": 6.5,
    "axes.titlesize": 7,
    "axes.labelsize": 6.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "legend.title_fontsize": 6.5,
}
LETRA_FUENTE = 7
# Ancho medio de un carácter de DejaVu Serif, en fracciones del tamaño de la letra.
ANCHO_CARACTER = 0.58
MENOS = "−"


def con_signo_menos(texto: str) -> str:
    """Cambia el guion de los negativos por el signo menos tipográfico."""
    return texto.replace("-", MENOS)


def formato_fijo(decimales: int) -> FuncFormatter:
    """Marcas del eje con coma decimal y el mismo número de decimales en todas."""
    return FuncFormatter(lambda valor, _: con_signo_menos(en_espanol(valor, decimales)))


FORMATO_MILES = FuncFormatter(lambda valor, _: con_signo_menos(en_espanol(valor)))
FORMATO_DECIMAL = FuncFormatter(lambda valor, _: con_signo_menos(en_espanol(valor, 0 if float(valor).is_integer() else len(f"{valor:g}".split(".")[-1]))))
PUNTOS_POR_PULGADA_PLOTLY = 96
ALTO_MINIMO_HTML = 450


def aplicar_estilo(informe: bool = False) -> None:
    """Estilo de las fases; con `informe`, la letra de las figuras dibujadas al ancho de impresión."""
    mpl.rcParams.update(ESTILO)
    if informe:
        mpl.rcParams.update(ESTILO_INFORME)


def ajustar_al_ancho(texto: str, ancho_pulgadas: float, tamano: float) -> str:
    """Parte cada línea del texto para que quepa en el ancho dado con la letra dada."""
    caracteres = max(20, int(ancho_pulgadas * 72 / (ANCHO_CARACTER * tamano)))
    return "\n".join(textwrap.fill(linea, caracteres) for linea in texto.split("\n"))


def anotar_fuente(eje: plt.Axes, texto: str) -> None:
    """Escribe la fuente de los datos al pie de la figura, debajo del título del eje horizontal.

    El texto se parte al ancho de la figura, para que una nota larga no la ensanche.
    """
    eje.annotate(
        ajustar_al_ancho(texto, eje.figure.get_figwidth(), LETRA_FUENTE),
        xy=(0.0, 0.0),
        xycoords=("figure fraction", eje.xaxis.label),
        xytext=(0, -6),
        textcoords="offset points",
        fontsize=LETRA_FUENTE,
        color="#4C4C4C",
        ha="left",
        va="top",
    )


class Figuras:
    """Escribe las figuras de un cuaderno en su carpeta y lleva la lista para la comprobación final."""

    def __init__(self, carpeta: Path) -> None:
        self.carpeta = carpeta
        self.escritas: list[Path] = []
        carpeta.mkdir(parents=True, exist_ok=True)

    def guardar(self, figura: plt.Figure, nombre: str) -> Path:
        ruta = self.carpeta / f"{nombre}.png"
        figura.savefig(ruta)
        self.registrar(ruta)
        return ruta

    def registrar(self, ruta: Path) -> None:
        self.escritas.append(ruta)

    def faltantes(self) -> list[Path]:
        return [ruta for ruta in self.escritas if not ruta.is_file()]


def instrucciones(funcion: Callable) -> int:
    """Cuenta las instrucciones del cuerpo de la función, sin importar cómo se partan las líneas."""
    arbol = ast.parse(textwrap.dedent(inspect.getsource(funcion)))
    return len(arbol.body[0].body)


def exportar(figura: plt.Figure | go.Figure, ruta: Path) -> None:
    """Exporta a PNG o PDF (Matplotlib) o a PNG o HTML (Plotly) y libera la figura.

    El tamaño del diseño de Plotly es el de la imagen impresa; el HTML, en cambio, ocupa el
    ancho de la ventana y al menos `ALTO_MINIMO_HTML` píxeles de alto.
    """
    if isinstance(figura, go.Figure):
        if ruta.suffix == ".html":
            alto = max(figura.layout.height or 0, ALTO_MINIMO_HTML)
            interactiva = go.Figure(figura).update_layout(width=None, height=None, autosize=True)
            interactiva.write_html(ruta, include_plotlyjs="cdn", default_width="100%", default_height=f"{alto}px")
        else:
            figura.write_image(ruta, scale=mpl.rcParams["savefig.dpi"] / PUNTOS_POR_PULGADA_PLOTLY)
        return
    figura.savefig(ruta)
    plt.close(figura)


def medir(construir: Callable[[object], object], datos: object, ruta_png: Path, repeticiones: int = 3) -> dict:
    """Instrucciones, mediana del tiempo hasta el PNG y peso del PNG y de la segunda salida.

    La segunda salida es un HTML interactivo en Plotly y un PDF vectorial en Matplotlib y
    Seaborn. El PNG queda en `ruta_png`; la segunda salida se escribe en una carpeta temporal.
    """
    tiempos = []
    for _ in range(repeticiones):
        inicio = time.perf_counter()
        exportar(construir(datos), ruta_png)
        tiempos.append(time.perf_counter() - inicio)
    es_plotly = isinstance(construir(datos), go.Figure)
    with tempfile.TemporaryDirectory() as carpeta:
        ruta_segunda = Path(carpeta) / ("figura.html" if es_plotly else "figura.pdf")
        exportar(construir(datos), ruta_segunda)
        peso_segunda = ruta_segunda.stat().st_size / 1024
    plt.close("all")
    return {
        "Instrucciones": instrucciones(construir),
        "Tiempo a PNG (s)": statistics.median(tiempos),
        "PNG (kB)": ruta_png.stat().st_size / 1024,
        "Segunda salida": "HTML interactivo" if es_plotly else "PDF vectorial",
        "Segunda salida (kB)": peso_segunda,
    }
