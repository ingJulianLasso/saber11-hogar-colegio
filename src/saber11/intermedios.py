"""Escribe lo que leen los scripts de R: el entrenamiento preparado de la fase 2 y lo de la fase 3.

Fase 2: R no reproduce la imputación iterativa por su cuenta (no hay un par exacto de
`IterativeImputer`), así que lee el entrenamiento ya preparado con las tres decisiones de
`saber11.modelos`. Se corre después de fijarlas y antes de `src/r/fase-2.R`:

    uv run python -m saber11.intermedios

Fase 3: las variables de H1 a H4 sobre todo el 20224 limpio, con los valores observados
(sin imputar) y el marco de cada hipótesis (`MARCO_HIPOTESIS`: niveles y orden), y, cuando `segmentacion.K_KMEANS` está fijada, las áreas estandarizadas del
entrenamiento con sus clústeres y las celdas del plano PCA. Los scripts de R de la fase 3
solo leen esto (y `fase-2-niveles.json`); se corre antes de ellos:

    uv run python -m saber11.intermedios --fase 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from saber11 import datos, limpieza, modelos, particion, segmentacion
from saber11 import variables as v

CARPETA_INTERMEDIA = datos.DESTINO / "intermedio"
RUTA_ENTRENAMIENTO = CARPETA_INTERMEDIA / "fase-2-entrenamiento.parquet"
RUTA_NIVELES = datos.RAIZ_PROYECTO / "resultados" / "fase-2-niveles.json"
RUTA_CIFRAS_FASE_2 = datos.RAIZ_PROYECTO / "resultados" / "cifras-fase-2.json"
RUTA_HIPOTESIS = CARPETA_INTERMEDIA / "fase-3-hipotesis.parquet"
RUTA_MULTIVARIANTE = CARPETA_INTERMEDIA / "fase-3-multivariante.parquet"
RUTA_PCA_CELDAS = CARPETA_INTERMEDIA / "fase-3-pca-celdas.parquet"
# H1: internet; H2: educación de la madre; H3: estrato por naturaleza; H4: estrato por
# naturaleza e internet por zona. Todas contra el puntaje global, salvo la H4.
VARIABLES_HIPOTESIS = (v.OBJETIVO_REGRESION, "FAMI_TIENEINTERNET", v.EDUCACION_MADRE, v.ESTRATO, v.OBJETIVO_CLASIFICACION, "COLE_AREA_UBICACION")
RUTA_MARCO_HIPOTESIS = CARPETA_INTERMEDIA / "fase-3-hipotesis.json"
INTERNET, ZONA = "FAMI_TIENEINTERNET", "COLE_AREA_UBICACION"
ESTRATOS_CON_SIN_ESTRATO = [*v.ESTRATOS, limpieza.SIN_ESTRATO]
NATURALEZAS = ["OFICIAL", v.CLASE_POSITIVA]
# Niveles y orden de cada hipótesis, los mismos en el cuaderno y en src/r/fase-3-hipotesis.R.
# Cada prueba usa los casos completos de sus variables. H1 compara Sí con No; el faltante de
# internet es un grupo descriptivo. H2 prueba los diez niveles de la escala: "No sabe" y "No
# Aplica" no dicen un nivel educativo y quedan como grupos descriptivos. "Sin Estrato" sí
# describe el hogar (vive donde no hay estratificación, regla de la fase 1) y entra en H3 y H4
# como séptimo nivel. En H3 y en los efectos simples, la brecha es "NO OFICIAL" menos "OFICIAL".
# Las etiquetas son las de las figuras en los dos lenguajes.
SIN_DATO = "Sin dato"
MARCO_HIPOTESIS = {
    "objetivo": v.OBJETIVO_REGRESION,
    "sin_dato": SIN_DATO,
    "h1": {"variable": INTERNET, "grupos": ["Si", "No"], "etiquetas": {"Si": "Sí", "No": "No", SIN_DATO: SIN_DATO}},
    "h2": {
        "variable": v.EDUCACION_MADRE,
        "niveles": list(v.EDUCACION),
        "descriptivos": ["No sabe", "No Aplica"],
        "etiquetas": dict(zip(
            [*v.EDUCACION, "No sabe", "No Aplica"],
            [
                "Ninguno", "Primaria inc.", "Primaria comp.", "Secundaria inc.", "Secundaria comp.", "Técnica inc.",
                "Técnica comp.", "Profesional inc.", "Profesional comp.", "Posgrado", "No sabe", "No aplica",
            ],
            strict=True,
        )),
    },
    "h3": {
        "a": v.ESTRATO,
        "niveles_a": ESTRATOS_CON_SIN_ESTRATO,
        "b": v.OBJETIVO_CLASIFICACION,
        "niveles_b": NATURALEZAS,
        "etiquetas_a": {**{e: e.removeprefix("Estrato ") for e in v.ESTRATOS}, limpieza.SIN_ESTRATO: "Sin estrato"},
        "etiquetas_b": {"OFICIAL": "Oficial", v.CLASE_POSITIVA: "No oficial"},
    },
    "h4": {
        "estrato_naturaleza": {"filas": v.ESTRATO, "niveles_filas": ESTRATOS_CON_SIN_ESTRATO, "columnas": v.OBJETIVO_CLASIFICACION, "niveles_columnas": NATURALEZAS},
        "internet_zona": {"filas": INTERNET, "niveles_filas": ["Si", "No"], "columnas": ZONA, "niveles_columnas": ["URBANO", "RURAL"]},
    },
}
# Variables con las que se perfilan los clústeres (χ² de clúster por estrato, zona y naturaleza).
PERFILES = (v.ESTRATO, "COLE_AREA_UBICACION", v.OBJETIVO_CLASIFICACION)
PREFIJO_ESTANDARIZADA = "Z_"
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


def hipotesis(limpio: pd.DataFrame) -> pd.DataFrame:
    """Variables de H1 a H4 de todo el 20224 limpio, con los faltantes como están."""
    return limpio[list(VARIABLES_HIPOTESIS)].reset_index(names="posicion")


def muestra_de_la_fase_2(limpio: pd.DataFrame, conjuntos: pd.Series, ruta_cifras: Path = RUTA_CIFRAS_FASE_2) -> pd.Series:
    """Muestra de la fase 2 recalculada con su tamaño; falla si la partición o la muestra no tienen sus huellas."""
    cifras = json.loads(ruta_cifras.read_text(encoding="utf-8"))
    if particion.huella(conjuntos) != cifras["huellaParticion"]["valor"]:
        raise ValueError("La partición no tiene la huella de la fase 2")
    muestra = particion.muestra_estratificada(conjuntos, limpio[v.OBJETIVO_CLASIFICACION], int(cifras["tamanoMuestra"]["valor"]))
    # Como en el cuaderno de la fase 2, la huella de la muestra se toma sobre "1" y "0".
    if particion.huella(muestra.map({True: "1", False: "0"})) != cifras["huellaMuestra"]["valor"]:
        raise ValueError("La muestra no tiene la huella de la fase 2")
    return muestra


def multivariante(limpio: pd.DataFrame, conjuntos: pd.Series, muestra: pd.Series, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entrenamiento con las cinco áreas: crudas, estandarizadas y con su clúster; y las celdas del plano PCA."""
    areas = limpio.loc[conjuntos.eq(particion.ENTRENAMIENTO), list(segmentacion.AREAS)].dropna().astype(float)
    z = segmentacion.estandarizar(areas)
    en_muestra = muestra.loc[z.index].to_numpy()
    asignaciones = segmentacion.segmentar(z, en_muestra, k)
    plano = segmentacion.pca(z).transform(z)[:, :2]
    tabla = pd.concat(
        [areas, z.add_prefix(PREFIJO_ESTANDARIZADA), asignaciones, limpio.loc[z.index, list(PERFILES)]], axis=1
    ).assign(muestra=en_muestra)
    return tabla.reset_index(names="posicion"), segmentacion.celdas_plano(plano, asignaciones["kmeans"].to_numpy())


def escribir(tabla: pd.DataFrame, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_parquet(ruta, index=False)
    print(f"Escrito {ruta.relative_to(datos.RAIZ_PROYECTO)} ({len(tabla)} filas)")


def fase_2() -> None:
    limpio, _ = cargar_limpio()
    conjuntos = particion.particionar(limpio[v.OBJETIVO_CLASIFICACION])
    preparado = entrenamiento_preparado(limpio, conjuntos)
    escribir(preparado.reset_index(names="posicion"), RUTA_ENTRENAMIENTO)
    RUTA_NIVELES.parent.mkdir(parents=True, exist_ok=True)
    RUTA_NIVELES.write_text(json.dumps(niveles(preparado), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Escrito {RUTA_NIVELES.relative_to(datos.RAIZ_PROYECTO)}")


def fase_3() -> None:
    limpio, _ = cargar_limpio()
    escribir(hipotesis(limpio), RUTA_HIPOTESIS)
    RUTA_MARCO_HIPOTESIS.write_text(json.dumps(MARCO_HIPOTESIS, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Escrito {RUTA_MARCO_HIPOTESIS.relative_to(datos.RAIZ_PROYECTO)}")
    if segmentacion.K_KMEANS is None:
        print("segmentacion.K_KMEANS sin fijar: no se escribe lo del multivariante")
        return
    conjuntos = particion.particionar(limpio[v.OBJETIVO_CLASIFICACION])
    tabla, celdas = multivariante(limpio, conjuntos, muestra_de_la_fase_2(limpio, conjuntos), segmentacion.K_KMEANS)
    escribir(tabla, RUTA_MULTIVARIANTE)
    escribir(celdas, RUTA_PCA_CELDAS)


def main(argumentos: list[str] | None = None) -> int:
    lector = argparse.ArgumentParser(description="Intermedios que leen los scripts de R.")
    lector.add_argument("--fase", type=int, choices=(2, 3), default=2)
    {2: fase_2, 3: fase_3}[lector.parse_args(argumentos).fase]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
