"""Escribe lo que lee el script de R de la fase 2: el entrenamiento preparado y sus niveles.

R no reproduce la imputación iterativa por su cuenta (no hay un par exacto de
`IterativeImputer`), así que lee el entrenamiento ya preparado con las tres decisiones de
`saber11.modelos`. Se corre después de fijarlas y antes de `src/r/fase-2.R`:

    uv run python -m saber11.intermedios
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from saber11 import datos, limpieza, modelos, particion
from saber11 import variables as v

CARPETA_INTERMEDIA = datos.DESTINO / "intermedio"
RUTA_ENTRENAMIENTO = CARPETA_INTERMEDIA / "fase-2-entrenamiento.parquet"
RUTA_NIVELES = datos.RAIZ_PROYECTO / "resultados" / "fase-2-niveles.json"
PREFIJO_ANTES = "ANTES_"
# Variables del laboratorio descriptivo (A3), que R resume antes y después de preparar.
DESCRIPTIVAS_NUMERICAS = (v.EDAD, v.CUARTOS_HOGAR, v.PERSONAS_HOGAR, v.HACINAMIENTO, v.OBJETIVO_REGRESION)
DESCRIPTIVAS_CATEGORICAS = (v.ESTRATO, "FAMI_TIENEINTERNET", v.EDUCACION_MADRE)
# Términos de la figura de coeficientes, la misma en Matplotlib, Seaborn, Plotly y ggplot2.
TERMINOS_FIGURA = {
    "FAMI_TIENEINTERNET=Si": "Internet en el hogar",
    "FAMI_TIENECOMPUTADOR=Si": "Computador en el hogar",
    "FAMI_TIENEAUTOMOVIL=Si": "Automóvil en el hogar",
    "FAMI_TIENELAVADORA=Si": "Lavadora en el hogar",
    "FAMI_EDUCACIONMADRE=Educación profesional completa": "Madre profesional",
    "FAMI_EDUCACIONMADRE=Postgrado": "Madre con posgrado",
    "FAMI_EDUCACIONPADRE=Educación profesional completa": "Padre profesional",
    "FAMI_ESTRATOVIVIENDA=Estrato 4": "Estrato 4",
    "COLE_NATURALEZA=NO OFICIAL": "Colegio no oficial",
    "COLE_AREA_UBICACION=RURAL": "Sede rural",
    "COLE_JORNADA=COMPLETA": "Jornada completa",
    "COLE_JORNADA=NOCHE": "Jornada nocturna",
    "COLE_JORNADA=SABATINA": "Jornada sabatina",
    "ESTU_GENERO=M": "Hombre",
}


def cargar_limpio(recorte: Path = datos.RECORTE_MODELADO) -> tuple[pd.DataFrame, list[limpieza.Exclusion]]:
    return limpieza.limpiar(pd.read_parquet(recorte))


def antes_de_preparar(entrenamiento: pd.DataFrame) -> pd.DataFrame:
    """Variables del laboratorio tal como quedan tras la limpieza, sin imputar ni tratar atípicos."""
    antes = entrenamiento[[c for c in (*DESCRIPTIVAS_NUMERICAS, *DESCRIPTIVAS_CATEGORICAS) if c != v.HACINAMIENTO]].assign(
        **{v.HACINAMIENTO: entrenamiento[v.PERSONAS_HOGAR] / entrenamiento[v.CUARTOS_HOGAR]}
    )
    return antes.add_prefix(PREFIJO_ANTES)


def entrenamiento_preparado(limpio: pd.DataFrame, conjuntos: pd.Series) -> pd.DataFrame:
    """Entrenamiento con las decisiones de `modelos`, sin escalar, más las columnas de antes."""
    entrenamiento = limpio.loc[conjuntos.eq(particion.ENTRENAMIENTO)]
    predictores = v.PREDICTORES_REGRESION[modelos.ESPECIFICACION]
    preparado = modelos.preparacion(modelos.IMPUTACION).fit_transform(entrenamiento[list(v.ENTRADA_REGRESION)])
    indicadores = [c for c in preparado.columns if v.es_indicador(c)]
    descriptivas = [c for c in (*DESCRIPTIVAS_NUMERICAS, *DESCRIPTIVAS_CATEGORICAS) if c != v.OBJETIVO_REGRESION]
    despues = preparado[[*dict.fromkeys([*predictores, *v.DERIVADAS, *v.ESTUDIANTE, *descriptivas]), *indicadores]]
    return pd.concat([entrenamiento[[v.OBJETIVO_REGRESION]], despues, antes_de_preparar(entrenamiento)], axis=1)


def niveles(preparado: pd.DataFrame) -> dict:
    """Términos del modelo elegido y referencias, para que R arme el mismo `lm`."""
    predictores = v.PREDICTORES_REGRESION[modelos.ESPECIFICACION]
    simple, rho = modelos.predictor_simple(preparado, preparado[v.OBJETIVO_REGRESION])
    indicadores = [c for c in preparado.columns if v.es_indicador(c)]
    return {
        "objetivo": v.OBJETIVO_REGRESION,
        "predictor_simple": simple,
        "spearman_simple": rho,
        "imputacion": modelos.IMPUTACION,
        "especificacion": modelos.ESPECIFICACION,
        "atipicos": modelos.ATIPICOS,
        "numericas": [c for c in predictores if c in v.NUMERICAS],
        "categoricas": {
            c: {"referencia": v.REFERENCIAS[c], "niveles": sorted(preparado[c].unique().tolist())}
            for c in predictores
            if c in v.REFERENCIAS
        },
        "indicadores": indicadores,
        "descriptivas_numericas": list(DESCRIPTIVAS_NUMERICAS),
        "descriptivas_categoricas": list(DESCRIPTIVAS_CATEGORICAS),
        "prefijo_antes": PREFIJO_ANTES,
        "terminos_figura": TERMINOS_FIGURA,
        "filas": len(preparado),
    }


def main() -> int:
    limpio, _ = cargar_limpio()
    conjuntos = particion.particionar(limpio[v.OBJETIVO_CLASIFICACION])
    preparado = entrenamiento_preparado(limpio, conjuntos)
    CARPETA_INTERMEDIA.mkdir(parents=True, exist_ok=True)
    RUTA_NIVELES.parent.mkdir(parents=True, exist_ok=True)
    preparado.reset_index(names="posicion").to_parquet(RUTA_ENTRENAMIENTO, index=False)
    RUTA_NIVELES.write_text(json.dumps(niveles(preparado), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Escrito {RUTA_ENTRENAMIENTO.relative_to(datos.RAIZ_PROYECTO)} ({len(preparado)} filas)")
    print(f"Escrito {RUTA_NIVELES.relative_to(datos.RAIZ_PROYECTO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
