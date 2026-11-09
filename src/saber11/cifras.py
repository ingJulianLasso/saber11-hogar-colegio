"""Formato numérico en español y registro de las cifras que citan los informes en LaTeX.

Cada cuaderno registra sus cifras con ``Cifras.cifra`` y al final las vuelca con
``Cifras.volcar``: un JSON para el dashboard y un ``.tex`` con una macro por cifra
para el informe. El volcado relee el ``.tex`` y falla si no coincide con el JSON.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

CLAVE_VALIDA = re.compile(r"[A-Za-z]+")
MACRO = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}")
ESPECIALES_LATEX = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def en_espanol(valor: float, decimales: int = 0) -> str:
    """Número con coma decimal y punto de miles; los de cuatro cifras van sin separador, como pide la RAE."""
    texto = f"{valor:,.{decimales}f}" if abs(valor) >= 10_000 else f"{valor:.{decimales}f}"
    return texto.replace(",", " ").replace(".", ",").replace(" ", ".")


def escapar_latex(texto: str) -> str:
    return "".join(ESPECIALES_LATEX.get(caracter, caracter) for caracter in texto)


class Cifras:
    """Cifras de un informe; cada clave es el nombre de una macro de LaTeX (solo letras)."""

    def __init__(self) -> None:
        self._datos: dict[str, dict[str, float | str]] = {}

    def __len__(self) -> int:
        return len(self._datos)

    def cifra(self, clave: str, valor: float | str, decimales: int = 0) -> str:
        """Registra la cifra y devuelve su texto, listo para imprimir en el cuaderno."""
        if not CLAVE_VALIDA.fullmatch(clave):
            raise ValueError(f"La macro {clave!r} tiene caracteres que LaTeX no admite")
        texto = valor if isinstance(valor, str) else en_espanol(valor, decimales)
        self._datos[clave] = {"valor": valor if isinstance(valor, str) else float(valor), "texto": texto}
        return texto

    def volcar(self, carpeta: Path, nombre: str) -> tuple[Path, Path]:
        """Escribe ``<nombre>.json`` y ``<nombre>.tex`` en la carpeta y comprueba la ida y vuelta."""
        carpeta.mkdir(parents=True, exist_ok=True)
        ruta_json, ruta_tex = carpeta / f"{nombre}.json", carpeta / f"{nombre}.tex"
        ruta_json.write_text(json.dumps(self._datos, ensure_ascii=False, indent=2), encoding="utf-8")
        lineas = [f"% Generado por saber11.cifras ({nombre}): no editar a mano."]
        lineas += [f"\\newcommand{{\\{clave}}}{{{escapar_latex(str(dato['texto']))}}}" for clave, dato in sorted(self._datos.items())]
        ruta_tex.write_text("\n".join(lineas) + "\n", encoding="utf-8")
        comprobar_volcado(ruta_json, ruta_tex)
        return ruta_json, ruta_tex


def comprobar_volcado(ruta_json: Path, ruta_tex: Path) -> None:
    """Falla si alguna macro del ``.tex`` no tiene el mismo texto que su clave en el JSON."""
    leidas = dict(MACRO.findall(ruta_tex.read_text(encoding="utf-8")))
    guardadas = json.loads(ruta_json.read_text(encoding="utf-8"))
    esperadas = {clave: escapar_latex(dato["texto"]) for clave, dato in guardadas.items()}
    if leidas != esperadas:
        raise ValueError(f"{ruta_tex.name} no coincide con {ruta_json.name}")
