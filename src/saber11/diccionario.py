"""Diccionario de las 51 columnas del Saber 11: las 27 del análisis y las 24 excluidas.

Las descripciones son las oficiales de la metadata de datos.gov.co (recurso ``kgxf-xxbe``),
copiadas tal como las publica el ICFES, erratas incluidas. El tipo es el estadístico, no el
de almacenamiento. Las 24 columnas excluidas no están en los recortes, así que su motivo
vive aquí y no se puede recalcular desde el repositorio.
"""

from __future__ import annotations

from dataclasses import dataclass

CONTINUA = "cuantitativa continua"
DISCRETA = "cuantitativa discreta"
NOMINAL = "cualitativa nominal"
ORDINAL = "cualitativa ordinal"


@dataclass(frozen=True)
class Variable:
    tipo: str
    descripcion: str
    grupo: str


def _puntaje(descripcion: str) -> Variable:
    return Variable(DISCRETA, descripcion, "puntaje")


def _hogar(tipo: str, descripcion: str) -> Variable:
    return Variable(tipo, descripcion, "hogar")


def _colegio(descripcion: str) -> Variable:
    return Variable(NOMINAL, descripcion, "colegio")


VARIABLES: dict[str, Variable] = {
    "PERIODO": Variable(ORDINAL, "Periodo de los resultados", "periodo"),
    "PUNT_GLOBAL": _puntaje("Puntaje Global"),
    "PUNT_LECTURA_CRITICA": _puntaje("Puntaje Lectura crítica"),
    "PUNT_MATEMATICAS": _puntaje("Puntaje Matemáticas"),
    "PUNT_C_NATURALES": _puntaje("Puntaje Ciencias Naturales"),
    "PUNT_SOCIALES_CIUDADANAS": _puntaje("Puntaje Sociales y Ciudadanas"),
    "PUNT_INGLES": _puntaje("Puntaje Inglés"),
    "FAMI_ESTRATOVIVIENDA": _hogar(ORDINAL, "Estrato del examinando"),
    "FAMI_EDUCACIONMADRE": _hogar(ORDINAL, "Nivel de estudios de la madre"),
    "FAMI_EDUCACIONPADRE": _hogar(ORDINAL, "Nivel de estudios de la padre"),
    "FAMI_TIENEINTERNET": _hogar(NOMINAL, "¿Tiene internet?"),
    "FAMI_TIENECOMPUTADOR": _hogar(NOMINAL, "¿Tiene computador?"),
    "FAMI_TIENEAUTOMOVIL": _hogar(NOMINAL, "¿Tiene automóvil?"),
    "FAMI_TIENELAVADORA": _hogar(NOMINAL, "¿Tiene lavadora?"),
    "FAMI_CUARTOSHOGAR": _hogar(ORDINAL, "¿Cuántos cuartos tiene su hogar?"),
    "FAMI_PERSONASHOGAR": _hogar(ORDINAL, "¿Con cuantas personas vive?"),
    "COLE_NATURALEZA": _colegio("Naturaleza del establecimiento"),
    "COLE_AREA_UBICACION": _colegio("Ubicación de la Sede"),
    "COLE_JORNADA": _colegio("Jornada de la Sede"),
    "COLE_CALENDARIO": _colegio("Calendario del Establecimiento"),
    "COLE_CARACTER": _colegio("Carácter del establecimiento"),
    "COLE_BILINGUE": _colegio("¿Es colegio bilingüe?"),
    "COLE_GENERO": _colegio("Genero del Establecimiento"),
    "COLE_COD_DEPTO_UBICACION": _colegio("Departamento de la sede"),
    "COLE_DEPTO_UBICACION": _colegio("Departamento de la sede"),
    "ESTU_GENERO": Variable(NOMINAL, "Genero del examinando", "estudiante"),
    "ESTU_FECHANACIMIENTO": Variable(CONTINUA, "Fecha de nacimiento del examinando", "estudiante"),
}

# Motivo de exclusión -> columnas. Las cifras de los motivos se midieron el 8-nov-2026
# sobre el Parquet completo del dataset, en las filas publicadas de los seis periodos del
# proyecto (3.258.118) y del 20224 (532.566).
COLUMNAS_EXCLUIDAS: dict[str, tuple[str, ...]] = {
    "Control del registro, sin valor explicativo: el consecutivo y el tipo de documento "
    "identifican el registro, y el estado de investigación solo sirve para filtrar PUBLICAR": (
        "ESTU_CONSECUTIVO",
        "ESTU_TIPODOCUMENTO",
        "ESTU_ESTADOINVESTIGACION",
    ),
    "Identificadores y nombres del colegio y la sede: señalan instituciones concretas y "
    "no describen el hogar ni el tipo de colegio": (
        "COLE_COD_DANE_ESTABLECIMIENTO",
        "COLE_COD_DANE_SEDE",
        "COLE_CODIGO_ICFES",
        "COLE_NOMBRE_ESTABLECIMIENTO",
        "COLE_NOMBRE_SEDE",
        "COLE_SEDE_PRINCIPAL",
    ),
    "Geografía fina o redundante con el departamento del colegio: el municipio parte las "
    "celdas por debajo del mínimo publicable (7 de los 1.115 municipios del 20224 tienen "
    "menos de 10 estudiantes), y el departamento de presentación y el de residencia "
    "coinciden con el de la sede en el 98,0 % y el 99,0 % del 20224": (
        "COLE_COD_MCPIO_UBICACION",
        "COLE_MCPIO_UBICACION",
        "ESTU_COD_DEPTO_PRESENTACION",
        "ESTU_DEPTO_PRESENTACION",
        "ESTU_COD_MCPIO_PRESENTACION",
        "ESTU_MCPIO_PRESENTACION",
        "ESTU_COD_RESIDE_DEPTO",
        "ESTU_DEPTO_RESIDE",
        "ESTU_COD_RESIDE_MCPIO",
        "ESTU_MCPIO_RESIDE",
    ),
    "Constantes o casi constantes: ESTU_ESTUDIANTE vale ESTUDIANTE en el 100 % de las filas; "
    "la nacionalidad y el país de residencia son COLOMBIA en el 98,7 % del 20224 y el 99,7 % "
    "de la serie, y ESTU_PRIVADO_LIBERTAD es N en más del 99,9 %": (
        "ESTU_ESTUDIANTE",
        "ESTU_NACIONALIDAD",
        "ESTU_PAIS_RESIDE",
        "ESTU_PRIVADO_LIBERTAD",
    ),
    "Derivada del puntaje de inglés (nivel de desempeño): usarla junto al puntaje sería "
    "fuga de información": ("DESEMP_INGLES",),
}
