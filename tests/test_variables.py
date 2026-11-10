"""Pruebas de saber11.variables: derivadas, predictores prohibidos e imputación."""

import numpy as np
import pandas as pd
import pytest

from saber11 import datos, limpieza
from saber11 import variables as v


def hogar(**valores: object) -> pd.DataFrame:
    base = {
        v.EDUCACION_MADRE: "Primaria completa",
        v.EDUCACION_PADRE: "Postgrado",
        **{bien: "No" for bien in v.BIENES},
        v.CUARTOS_HOGAR: 2.0,
        v.PERSONAS_HOGAR: 5.5,
    }
    base.update(valores)
    return pd.DataFrame([base]).astype({v.EDUCACION_MADRE: "str", v.EDUCACION_PADRE: "str"})


def test_indice_de_bienes_cuenta_los_si():
    derivadas = v.agregar_derivadas(hogar(FAMI_TIENEINTERNET="Si", FAMI_TIENELAVADORA="Si"))
    assert derivadas[v.INDICE_BIENES].item() == 2


@pytest.mark.parametrize(
    ("madre", "padre", "maxima"),
    [
        ("Primaria completa", "Postgrado", "Postgrado"),
        ("No sabe", "Ninguno", "Ninguno"),
        ("No sabe", "No Aplica", v.EDUCACION_DESCONOCIDA),
        ("Educación profesional completa", "No sabe", "Educación profesional completa"),
    ],
)
def test_maxima_educacion_de_los_padres_ignora_lo_que_esta_fuera_de_la_escala(madre, padre, maxima):
    derivadas = v.agregar_derivadas(hogar(FAMI_EDUCACIONMADRE=madre, FAMI_EDUCACIONPADRE=padre))
    assert derivadas[v.EDUCACION_MAXIMA].item() == maxima


def test_hacinamiento_son_personas_por_cuarto():
    assert v.agregar_derivadas(hogar())[v.HACINAMIENTO].item() == pytest.approx(2.75)


def test_edad_al_presentar_es_la_cuarta_derivada():
    fechas = limpieza.fecha_nacimiento(pd.Series(["14/02/2004", "20/12/2005"]))
    assert limpieza.edad_cumplida(fechas).tolist() == [18.0, 17.0]


def test_prohibidos_de_regresion_cubren_los_cinco_puntajes_y_el_desempeno_en_ingles():
    areas = set(datos.PUNTAJES) - {v.OBJETIVO_REGRESION}
    assert areas | {"DESEMP_INGLES"} <= v.PROHIBIDOS_REGRESION
    assert len(areas) == 5


def test_imputacion_iterativa_devuelve_niveles_de_la_escala_y_respeta_lo_que_esta_fuera(limpio):
    imputador = v.Imputador("iterativa").fit(limpio[list(v.ENTRADA_REGRESION)])
    imputado = imputador.transform(limpio[list(v.ENTRADA_REGRESION)])
    faltaba = limpio[v.ESTRATO].isna()
    assert imputado[list(v.ENTRADA_REGRESION)].notna().all().all()
    assert set(imputado.loc[faltaba, v.ESTRATO]) <= set(v.ESTRATOS)
    fuera = limpio[v.EDUCACION_MADRE].isin(["No sabe", "No Aplica"])
    assert imputado.loc[fuera, v.EDUCACION_MADRE].equals(limpio.loc[fuera, v.EDUCACION_MADRE])
    assert set(imputado.loc[limpio[v.CUARTOS_HOGAR].isna(), v.CUARTOS_HOGAR]) <= set(v.VALORES_POSIBLES[v.CUARTOS_HOGAR])
    assert imputado[f"{v.PREFIJO_FALTA}{v.ESTRATO}"].sum() == faltaba.sum()


def test_imputacion_por_moda_rellena_con_el_valor_mas_frecuente(limpio):
    imputado = v.Imputador("moda").fit_transform(limpio[list(v.ENTRADA_REGRESION)])
    moda = limpio["COLE_BILINGUE"].mode().iloc[0]
    assert (imputado.loc[limpio["COLE_BILINGUE"].isna(), "COLE_BILINGUE"] == moda).all()


def test_atipicos_winsoriza_con_el_tope_de_entrenamiento_y_transforma_con_logaritmo():
    entrenamiento = pd.DataFrame({"a": np.arange(1.0, 101.0), "b": np.full(100, np.e)})
    atipicos = v.Atipicos({"a": "winsorizar", "b": "log"}).fit(entrenamiento)
    nuevo = atipicos.transform(pd.DataFrame({"a": [1000.0], "b": [np.e]}))
    assert nuevo["a"].item() == pytest.approx(np.percentile(entrenamiento["a"], 99))
    assert nuevo["b"].item() == pytest.approx(1.0)


def test_sin_indicador_para_columnas_con_muy_pocos_faltantes(limpio):
    pocos = limpio.copy()
    pocos["ESTU_GENERO"] = pocos["ESTU_GENERO"].astype(object)
    pocos.loc[pocos.index[:3], "ESTU_GENERO"] = np.nan
    imputado = v.Imputador("moda").fit_transform(pocos[list(v.ENTRADA_REGRESION)])
    assert f"{v.PREFIJO_FALTA}ESTU_GENERO" not in imputado.columns
    assert imputado["ESTU_GENERO"].notna().all()
    assert f"{v.PREFIJO_FALTA}{v.ESTRATO}" in imputado.columns
