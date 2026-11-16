"""Modelo de predicción del tablero: margen, hojas, observaciones guardadas y ficha del formulario.

El tablero publica un `Pipeline` de regresión que recibe las columnas de entrada tal como
las pide su formulario y estima el puntaje global esperado de un perfil. Este módulo
reúne lo que el cuaderno de síntesis y `tests/test_modelo_tablero.py` comprueban sobre él:
el margen de predicción por cuantil de residuos de validación, que toda hoja de XGBoost
cubra 10 o más estudiantes y que ningún objeto guarde una matriz con filas de estudiantes.
Para deserializar el modelo, la app solo necesita `saber11.variables`; `perfil` le sirve para
armar y validar la fila del formulario contra la ficha, sin dependencias nuevas. Por último,
`memoria_maxima_mb` mide la memoria que pide el dashboard completo, en un proceso aparte.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from collections.abc import Iterator, Sequence
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from saber11 import limpieza
from saber11 import variables as v

MINIMO_POR_HOJA = 10
MINIMO_POR_NIVEL = 10
FILAS_MAXIMAS_MATRIZ = 1000
NIVEL_MARGEN = 0.9
PESO_MAXIMO_MB = 50
PAQUETES = ("scikit-learn", "xgboost", "joblib", "numpy", "pandas", "scipy")
# Clases que guardan observaciones de entrenamiento para predecir: vecinos y vectores de soporte.
CLASES_CON_OBSERVACIONES = ("KNeighbors", "RadiusNeighbors", "SVR", "SVC", "NuSV", "OneClassSVM")
MODULOS_RECORRIDOS = ("sklearn", "saber11", "xgboost")
ETIQUETAS = {
    v.ESTRATO: "Estrato de la vivienda",
    v.EDUCACION_MADRE: "Educación de la madre",
    v.EDUCACION_PADRE: "Educación del padre",
    "FAMI_TIENEINTERNET": "Internet en el hogar",
    "FAMI_TIENECOMPUTADOR": "Computador en el hogar",
    "FAMI_TIENEAUTOMOVIL": "Automóvil en el hogar",
    "FAMI_TIENELAVADORA": "Lavadora en el hogar",
    v.CUARTOS_HOGAR: "Cuartos del hogar",
    v.PERSONAS_HOGAR: "Personas del hogar",
    "COLE_NATURALEZA": "Naturaleza del colegio",
    "COLE_AREA_UBICACION": "Zona de la sede",
    "COLE_JORNADA": "Jornada",
    "COLE_CALENDARIO": "Calendario",
    "COLE_CARACTER": "Carácter del colegio",
    "COLE_BILINGUE": "Colegio bilingüe",
    "COLE_GENERO": "Género del colegio",
    v.DEPARTAMENTO: "Departamento de la sede",
    "ESTU_GENERO": "Género del estudiante",
    v.EDAD: "Edad (años cumplidos)",
}
GRUPOS = {"hogar": v.HOGAR, "colegio": v.COLEGIO, "estudiante": v.ESTUDIANTE}
# Perfil fijo con que el cuaderno y las pruebas comprueban una predicción del formulario completo.
PERFIL_DE_EJEMPLO = {
    v.ESTRATO: "Estrato 2",
    v.EDUCACION_MADRE: "Secundaria (Bachillerato) completa",
    v.EDUCACION_PADRE: "Secundaria (Bachillerato) completa",
    "FAMI_TIENEINTERNET": "Si",
    "FAMI_TIENECOMPUTADOR": "Si",
    "FAMI_TIENEAUTOMOVIL": "No",
    "FAMI_TIENELAVADORA": "Si",
    v.CUARTOS_HOGAR: 3.0,
    v.PERSONAS_HOGAR: 3.5,
    "COLE_NATURALEZA": "OFICIAL",
    "COLE_AREA_UBICACION": "URBANO",
    "COLE_JORNADA": "MAÑANA",
    "COLE_CALENDARIO": "A",
    "COLE_CARACTER": "ACADÉMICO",
    "COLE_BILINGUE": "N",
    "COLE_GENERO": "MIXTO",
    v.DEPARTAMENTO: "BOGOTÁ",
    "ESTU_GENERO": "F",
    v.EDAD: 17.0,
}
# Las numéricas discretas llegan armonizadas (`limpieza.armonizar`); el formulario muestra su respuesta original.
RESPUESTAS = {v.CUARTOS_HOGAR: limpieza.CUARTOS, v.PERSONAS_HOGAR: limpieza.PERSONAS}
# Proceso hijo de `memoria_maxima_mb`: ejecuta la app como lo hace Streamlit en cada interacción (el guion
# entero, con todas las pestañas), envía el formulario de Predicción con otra audiencia y otro filtro, e
# informa el pico de memoria residente. En Linux lo lee de VmHWM, que empieza de cero con el programa nuevo:
# `ru_maxrss` arrastra la memoria del proceso padre (en el cuaderno, la del núcleo) del instante del fork.
# En otros sistemas usa `ru_maxrss` (en bytes en macOS).
_RECORRIDO_DEL_DASHBOARD = """
import resource, sys
from pathlib import Path
from streamlit.testing.v1 import AppTest

def pico_mb():
    estado = Path("/proc/self/status")
    if estado.is_file():
        linea = next(l for l in estado.read_text().splitlines() if l.startswith("VmHWM:"))
        return int(linea.split()[1]) / 1024
    pico = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return pico / 1024**2 if sys.platform == "darwin" else pico / 1024

app = AppTest.from_file(sys.argv[1], default_timeout=300)
app.run()
app.selectbox(key="audiencia").set_value("Equipo técnico")
app.selectbox(key="estrato").set_value("Estrato 3")
app.button(key="estimar").click()
app.run()
if app.exception or app.error:
    sys.exit(f"El dashboard falló: {[e.value for e in [*app.exception, *app.error]]}")
print(pico_mb())
"""


def margen_conformal(residuos_absolutos: Sequence[float], nivel: float = NIVEL_MARGEN) -> float:
    """Margen con corrección de muestra finita: el residuo absoluto en la posición ⌈(n + 1)·nivel⌉.

    Si los residuos de validación y los de un estudiante nuevo son intercambiables, el
    puntaje del nuevo cae a esa distancia de la estimación con probabilidad de al menos `nivel`.
    """
    ordenados = np.sort(np.asarray(residuos_absolutos, dtype=float))
    posicion = math.ceil((len(ordenados) + 1) * nivel)
    if posicion > len(ordenados):
        raise ValueError(f"Con {len(ordenados)} residuos no hay margen al nivel {nivel}")
    return float(ordenados[posicion - 1])


def cobertura_minima_de_hojas(modelo: object) -> float:
    """Menor `Cover` de las hojas de un XGBoost: con pérdida cuadrática, los estudiantes de la hoja."""
    arboles = modelo.get_booster().trees_to_dataframe()
    return float(arboles.loc[arboles["Feature"] == "Leaf", "Cover"].min())


def _recorrer(objeto: object, ruta: str, vistos: set[int]) -> Iterator[tuple[str, object]]:
    if id(objeto) in vistos:
        return
    vistos.add(id(objeto))
    yield ruta, objeto
    if isinstance(objeto, dict):
        hijos = ((f"{ruta}[{clave!r}]", valor) for clave, valor in objeto.items())
    elif isinstance(objeto, (list, tuple, set)):
        hijos = ((f"{ruta}[{i}]", valor) for i, valor in enumerate(objeto))
    elif type(objeto).__module__.startswith(MODULOS_RECORRIDOS) and hasattr(objeto, "__dict__"):
        hijos = ((f"{ruta}.{nombre}", valor) for nombre, valor in vars(objeto).items())
    else:
        return
    for ruta_hijo, hijo in hijos:
        yield from _recorrer(hijo, ruta_hijo, vistos)


def objetos(modelo: object) -> list[tuple[str, object]]:
    """Cada objeto que guarda el modelo, con su ruta de atributos, recorriendo las clases de sklearn, saber11 y xgboost."""
    return list(_recorrer(modelo, "modelo", set()))


def matrices_grandes(modelo: object, filas: int = FILAS_MAXIMAS_MATRIZ) -> list[str]:
    """Rutas de las matrices de dos dimensiones con `filas` o más filas: serían observaciones guardadas."""
    return [
        ruta for ruta, objeto in objetos(modelo)
        if (isinstance(objeto, (np.ndarray, pd.DataFrame)) or sparse.issparse(objeto)) and objeto.ndim == 2 and objeto.shape[0] >= filas
    ]


def clases_con_observaciones(modelo: object) -> list[str]:
    """Clases de vecinos o de vectores de soporte dentro del modelo."""
    return sorted({type(objeto).__name__ for _, objeto in objetos(modelo) if type(objeto).__name__.startswith(CLASES_CON_OBSERVACIONES)})


def versiones() -> dict[str, str]:
    return {paquete: version(paquete) for paquete in PAQUETES}


def _grupo(columna: str) -> str:
    return next(grupo for grupo, columnas in GRUPOS.items() if columna in columnas)


def _niveles(serie: pd.Series) -> list[str]:
    """Niveles con `MINIMO_POR_NIVEL` o más estudiantes; los de una escala ordinal, en su orden."""
    conteos = serie.dropna().value_counts()
    vistos = conteos[conteos >= MINIMO_POR_NIVEL].index
    escala = v.ESCALAS.get(serie.name, ())
    return [nivel for nivel in escala if nivel in vistos] + sorted(str(nivel) for nivel in vistos if nivel not in escala)


def _valores(serie: pd.Series) -> list[float]:
    """Valores numéricos con `MINIMO_POR_NIVEL` o más estudiantes, de menor a mayor."""
    conteos = serie.dropna().value_counts()
    return sorted(float(valor) for valor in conteos[conteos >= MINIMO_POR_NIVEL].index)


def entradas(X_entrenamiento: pd.DataFrame, con_faltantes: Sequence[str]) -> list[dict]:
    """Una entrada del formulario por columna, con los niveles o valores vistos en entrenamiento.

    Solo se ofrecen niveles y valores con 10 o más estudiantes de entrenamiento, y «Sin dato»
    solo en las columnas que el `Pipeline` imputa (las que tenían faltantes al ajustarlo).
    """
    lista = []
    for columna in X_entrenamiento.columns:
        entrada = {"columna": columna, "etiqueta": ETIQUETAS[columna], "grupo": _grupo(columna), "sin_dato": columna in con_faltantes}
        if columna in v.NUMERICAS:
            valores = _valores(X_entrenamiento[columna])
            entrada |= {"tipo": "numerica", "minimo": valores[0], "maximo": valores[-1], "valores": valores}
            if columna in RESPUESTAS:
                entrada["respuestas"] = [{"valor": valor, "respuesta": respuesta} for respuesta, valor in RESPUESTAS[columna].items() if valor in valores]
        else:
            entrada |= {"tipo": "categorica", "niveles": _niveles(X_entrenamiento[columna])}
        lista.append(entrada)
    return lista


def perfil(ficha: dict, valores: dict[str, object]) -> pd.DataFrame:
    """Una fila con las columnas de entrada en el orden de la ficha; `None` es «Sin dato».

    El «Sin dato» se escribe como `NaN` también en las categóricas: el imputador de la moda
    (`SimpleImputer`) no reconoce `None` como faltante en una columna de objetos, y el nivel
    vacío llegaría sin imputar al one-hot. Falla si un valor no es un nivel o un valor
    admitido de su entrada, o si una entrada que no admite «Sin dato» llega vacía.
    """
    fila = {}
    for entrada in ficha["entrada"]:
        valor = valores[entrada["columna"]]
        if valor is None:
            if not entrada["sin_dato"]:
                raise ValueError(f"{entrada['etiqueta']} no admite «Sin dato»")
            fila[entrada["columna"]] = np.nan
            continue
        admitidos = entrada["valores"] if entrada["tipo"] == "numerica" else entrada["niveles"]
        if valor not in admitidos:
            raise ValueError(f"{entrada['etiqueta']}: {valor!r} no es un valor admitido")
        fila[entrada["columna"]] = float(valor) if entrada["tipo"] == "numerica" else valor
    marco = pd.DataFrame([fila])
    tipos = {e["columna"]: (float if e["tipo"] == "numerica" else object) for e in ficha["entrada"]}
    return marco.astype(tipos)


def memoria_maxima_mb(app: Path) -> float:
    """Pico de memoria residente, en MiB, de un proceso aparte que recorre el dashboard con `AppTest`.

    Mide el guion de la app con sus datos y su modelo, sin GPU visible, como en Streamlit Community
    Cloud; no incluye el servidor web de Streamlit.
    """
    entorno = os.environ | {"CUDA_VISIBLE_DEVICES": ""}
    salida = subprocess.run([sys.executable, "-c", _RECORRIDO_DEL_DASHBOARD, str(app)], capture_output=True, text=True, check=True, env=entorno)
    return float(salida.stdout.strip().splitlines()[-1])
