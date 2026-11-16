"""Formato numérico en español y registro de las cifras que citan los informes en LaTeX.

Cada cuaderno registra sus cifras con ``Cifras.cifra`` y al final las vuelca con
``Cifras.volcar``: un JSON para el dashboard y un ``.tex`` con una macro por cifra
para el informe. El volcado relee el ``.tex`` y falla si no coincide con el JSON.

La fase 4 parte de ``Cifras.consolidar``, que une las cifras de las fases anteriores en un
solo registro, y de ``Cifras.metricas_de_prueba``, que da una macro a cada métrica de los
registros de evaluación en prueba sin crear dos nombres para la misma cifra.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

CLAVE_VALIDA = re.compile(r"[A-Za-z]+")
MACRO = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}")
FASE_DEL_ARCHIVO = re.compile(r"fase-(\d)$")
DIGITOS = {"0": "Cero", "1": "Uno", "2": "Dos", "3": "Tres", "4": "Cuatro", "5": "Cinco", "6": "Seis", "7": "Siete", "8": "Ocho", "9": "Nueve"}
# R² se lee «R cuadrado»; las demás métricas pasan sus dígitos a palabra (f1 a fUno).
NOMBRE_ESPECIAL = {"r2": "rCuadrado"}
DECIMALES_METRICA = {"rmse": 2, "mae": 2}
DECIMALES_POR_OMISION = 3
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


def _comprobar_clave(clave: str) -> None:
    if not CLAVE_VALIDA.fullmatch(clave):
        raise ValueError(f"La macro {clave!r} tiene caracteres que LaTeX no admite")


def _en_palabras(texto: str) -> str:
    """Pasa cada dígito a su nombre con mayúscula inicial y quita lo que no sea letra."""
    return re.sub(r"[^A-Za-z]", "", re.sub(r"\d", lambda digito: DIGITOS[digito.group()], texto))


def sufijo_de_fase(ruta: Path) -> str:
    """`cifras-fase-2.json` da `FaseDos`: el sufijo de las claves que chocan entre fases."""
    coincidencia = FASE_DEL_ARCHIVO.search(ruta.stem)
    if coincidencia is None:
        raise ValueError(f"{ruta.name} no termina en «fase-N»")
    return "Fase" + DIGITOS[coincidencia.group(1)]


def nombre_metrica(metrica: str) -> str:
    """Nombre de macro de una métrica: `r2` da `rCuadrado`, `f1` da `fUno` y `roc_auc`, `rocAuc`."""
    if metrica in NOMBRE_ESPECIAL:
        return NOMBRE_ESPECIAL[metrica]
    primera, *resto = metrica.split("_")
    return _en_palabras(primera + "".join(parte[:1].upper() + parte[1:] for parte in resto))


class Cifras:
    """Cifras de un informe; cada clave es el nombre de una macro de LaTeX (solo letras)."""

    def __init__(self) -> None:
        self._datos: dict[str, dict[str, float | str]] = {}

    def __len__(self) -> int:
        return len(self._datos)

    def __contains__(self, clave: object) -> bool:
        return clave in self._datos

    def valor(self, clave: str) -> float | str:
        return self._datos[clave]["valor"]

    def texto(self, clave: str) -> str:
        return str(self._datos[clave]["texto"])

    def cifra(self, clave: str, valor: float | str, decimales: int = 0) -> str:
        """Registra la cifra y devuelve su texto, listo para imprimir en el cuaderno."""
        _comprobar_clave(clave)
        texto = valor if isinstance(valor, str) else en_espanol(valor, decimales)
        self._datos[clave] = {"valor": valor if isinstance(valor, str) else float(valor), "texto": texto}
        return texto

    def _agregar_nueva(self, clave: str, dato: dict[str, float | str]) -> None:
        _comprobar_clave(clave)
        if clave in self._datos:
            raise ValueError(f"La macro {clave!r} quedaría dos veces al consolidar")
        self._datos[clave] = dict(dato)

    @classmethod
    def consolidar(cls, rutas: Sequence[Path]) -> Cifras:
        """Une los JSON de cifras de varias fases (`cifras-fase-N.json`) en un solo registro.

        Una clave que se repite con el mismo texto queda una vez; si el texto cambia entre
        fases, sale solo con el sufijo de cada fase (`FaseUno`, `FaseDos`...), para que
        ninguna macro tenga dos significados. Falla con una clave que LaTeX no admite o si
        un sufijo produce una clave que ya existe.
        """
        apariciones: dict[str, list[tuple[str, dict[str, float | str]]]] = {}
        for ruta in rutas:
            sufijo = sufijo_de_fase(ruta)
            for clave, dato in json.loads(ruta.read_text(encoding="utf-8")).items():
                _comprobar_clave(clave)
                apariciones.setdefault(clave, []).append((sufijo, dato))
        consolidadas = cls()
        for clave, lista in apariciones.items():
            if len({dato["texto"] for _, dato in lista}) == 1:
                consolidadas._agregar_nueva(clave, lista[0][1])
            else:
                for sufijo, dato in lista:
                    consolidadas._agregar_nueva(clave + sufijo, dato)
        return consolidadas

    def metricas_de_prueba(self, ruta: Path) -> dict[tuple[str, str], str]:
        """Macro de cada métrica de un registro de prueba (lista de `{"modelo", "metricas"}`).

        Si ya hay una macro `<metrica>Prueba...` con el mismo valor, se usa su nombre; si no
        hay ninguna, se crea `<metrica>Prueba<Modelo>`. Falla si dos macros tienen ese valor,
        porque no se sabría cuál es la del modelo.
        """
        nombres = {}
        for fila in json.loads(ruta.read_text(encoding="utf-8")):
            modelo = fila["modelo"]
            for metrica, valor in fila["metricas"].items():
                prefijo = f"{nombre_metrica(metrica)}Prueba"
                iguales = [clave for clave, dato in self._datos.items() if clave.startswith(prefijo) and dato["valor"] == valor]
                if len(iguales) > 1:
                    raise ValueError(f"{modelo}, {metrica}: varias macros tienen el mismo valor ({iguales})")
                if iguales:
                    nombres[modelo, metrica] = iguales[0]
                else:
                    clave = prefijo + _en_palabras(modelo[:1].upper() + modelo[1:])
                    self.cifra(clave, valor, DECIMALES_METRICA.get(metrica, DECIMALES_POR_OMISION))
                    nombres[modelo, metrica] = clave
        return nombres

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
