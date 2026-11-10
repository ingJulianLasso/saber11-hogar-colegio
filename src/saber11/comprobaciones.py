"""Comprobaciones de los cuadernos: detienen la ejecución si algo falla y llevan la cuenta.

Como una comprobación fallida detiene el cuaderno, el resumen final siempre dice "N de N";
un número distinto en el cuaderno ejecutado delataría una ejecución incompleta.
"""

from __future__ import annotations


class Comprobaciones:
    """Registro de las comprobaciones superadas de un cuaderno."""

    def __init__(self) -> None:
        self.superadas: list[str] = []

    def __len__(self) -> int:
        return len(self.superadas)

    def comprobar(self, descripcion: str, condicion: object) -> None:
        """Detiene la ejecución si la condición es falsa; si no, la registra e imprime `[ok]`."""
        if not condicion:
            raise AssertionError(f"Comprobación fallida: {descripcion}")
        self.superadas.append(descripcion)
        print(f"[ok] {descripcion}")

    def resumen(self) -> str:
        total = len(self.superadas)
        return f"{total} de {total} comprobaciones superadas"
