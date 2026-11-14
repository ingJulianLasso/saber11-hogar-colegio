"""Pruebas de hipótesis de la fase 3: envoltorios de scipy y statsmodels con efecto e intervalo.

Cada prueba devuelve el estadístico, sus grados de libertad, el valor p, el tamaño de efecto
y su intervalo. Las funciones reciben casos completos: el cuaderno quita los faltantes de
cada prueba antes de llamarlas. Los intervalos de la diferencia de Welch y de la d de Cohen
son analíticos; los de η², ε², η² parcial y V de Cramér son bootstrap percentil con
`scipy.stats.bootstrap` (1.000 réplicas y semilla 42). Para la V se remuestrea la tabla con
una multinomial: se remuestrean las celdas de las observaciones, que es lo mismo. R calcula los
mismos efectos con las mismas fórmulas, pero no sus intervalos bootstrap.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.oneway import anova_oneway

SEMILLA = 42
REPLICAS = 1000
CONFIANZA = 0.95
# Réplicas que se remuestrean a la vez: acota la memoria con medio millón de filas.
LOTE_BOOTSTRAP = 20
NAN = float("nan")


@dataclass(frozen=True)
class Contraste:
    """Resultado de una prueba; `detalle` lleva lo propio de cada una (medias, conteos)."""

    prueba: str
    estadistico: float
    gl: tuple[float, ...]
    p: float
    n: int
    efecto: str = ""
    valor: float = NAN
    ic: tuple[float, float] = (NAN, NAN)
    detalle: Mapping[str, float] = field(default_factory=dict)


def codigos(grupos: Sequence) -> tuple[np.ndarray, np.ndarray]:
    """Niveles ordenados y código entero de cada observación."""
    niveles, codigo = np.unique(np.asarray(grupos), return_inverse=True)
    return niveles, codigo


def intervalo_bootstrap(
    estadistico: Callable[..., float | np.ndarray],
    *datos: np.ndarray,
    replicas: int = REPLICAS,
    semilla: int = SEMILLA,
    confianza: float = CONFIANZA,
) -> tuple[np.ndarray, np.ndarray]:
    """Intervalo bootstrap percentil, remuestreando las filas de `datos` de forma pareada.

    Con `replicas=0` no remuestrea y devuelve NaN, para las pruebas rápidas y la paridad.
    """
    if replicas == 0:
        forma = np.shape(estadistico(*datos))
        return np.full(forma, np.nan), np.full(forma, np.nan)
    tipos = [np.asarray(d).dtype for d in datos]

    def con_sus_tipos(*remuestreados: np.ndarray) -> float | np.ndarray:
        # scipy remuestrea las muestras pareadas como reales; los códigos vuelven a ser enteros.
        return estadistico(*(r.astype(t, copy=False) for r, t in zip(remuestreados, tipos, strict=True)))

    resultado = stats.bootstrap(
        datos,
        con_sus_tipos,
        paired=True,
        vectorized=False,
        n_resamples=replicas,
        batch=LOTE_BOOTSTRAP,
        method="percentile",
        confidence_level=confianza,
        rng=np.random.default_rng(semilla),
    )
    return resultado.confidence_interval.low, resultado.confidence_interval.high


def _par(bajo: np.ndarray, alto: np.ndarray) -> tuple[float, float]:
    return float(bajo), float(alto)


# Tamaños de efecto


def d_cohen(a: np.ndarray, b: np.ndarray) -> float:
    """Diferencia de medias (a menos b) sobre la desviación combinada con n - 1."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    combinada = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / combinada)


def ic_d_cohen(d: float, n_a: int, n_b: int, confianza: float = CONFIANZA) -> tuple[float, float]:
    """Intervalo analítico de la d con el error estándar de Hedges y Olkin."""
    error = np.sqrt((n_a + n_b) / (n_a * n_b) + d**2 / (2 * (n_a + n_b)))
    z = stats.norm.ppf(0.5 + confianza / 2)
    return float(d - z * error), float(d + z * error)


def eta_cuadrado(y: np.ndarray, codigo: np.ndarray) -> float:
    """Suma de cuadrados entre grupos sobre la total."""
    y = np.asarray(y, dtype=float)
    conteos = np.bincount(codigo)
    presentes = conteos > 0
    medias = np.bincount(codigo, weights=y)[presentes] / conteos[presentes]
    entre = np.sum(conteos[presentes] * (medias - y.mean()) ** 2)
    return float(entre / np.sum((y - y.mean()) ** 2))


def epsilon_cuadrado(h: float, n: int) -> float:
    """ε² de Kruskal-Wallis: H / ((n² - 1) / (n + 1)), que es H / (n - 1)."""
    return float(h / (n - 1))


def _chi_cuadrado_de(tabla: np.ndarray) -> float:
    esperadas = np.outer(tabla.sum(axis=1), tabla.sum(axis=0)) / tabla.sum()
    llenas = esperadas > 0
    return float(np.sum((tabla[llenas] - esperadas[llenas]) ** 2 / esperadas[llenas]))


def v_cramer(tabla: np.ndarray) -> float:
    """V de Cramér sin corrección de sesgo, desde el χ² sin corrección de Yates."""
    tabla = np.asarray(tabla, dtype=float)
    return float(np.sqrt(_chi_cuadrado_de(tabla) / (tabla.sum() * (min(tabla.shape) - 1))))


def sumas_tipo_dos(conteos: np.ndarray, sumas: np.ndarray, cuadrados: np.ndarray) -> dict[str, float]:
    """Sumas de cuadrados tipo II de A, B y A:B, y la residual del modelo con interacción.

    Recibe, por celda de la tabla A × B, el número de observaciones, la suma de y y la suma
    de y²; así el bootstrap no reajusta un modelo por réplica. El modelo aditivo se resuelve
    con sus ecuaciones normales, que solo dependen de esos agregados.
    """
    total = cuadrados.sum()
    n_a, n_b, s_a, s_b = conteos.sum(axis=1), conteos.sum(axis=0), sumas.sum(axis=1), sumas.sum(axis=0)

    def residual_por_medias(n: np.ndarray, s: np.ndarray) -> float:
        llenas = n > 0
        return float(total - np.sum(s[llenas] ** 2 / n[llenas]))

    k_a, k_b = conteos.shape
    columnas_a, columnas_b = np.arange(1, k_a), np.arange(k_a, k_a + k_b - 1)
    normal = np.zeros((k_a + k_b - 1, k_a + k_b - 1))
    normal[0, 0] = conteos.sum()
    normal[0, columnas_a] = normal[columnas_a, 0] = n_a[1:]
    normal[0, columnas_b] = normal[columnas_b, 0] = n_b[1:]
    normal[columnas_a, columnas_a] = n_a[1:]
    normal[columnas_b, columnas_b] = n_b[1:]
    normal[np.ix_(columnas_a, columnas_b)] = conteos[1:, 1:]
    normal[np.ix_(columnas_b, columnas_a)] = conteos[1:, 1:].T
    derecha = np.concatenate([[sumas.sum()], s_a[1:], s_b[1:]])
    coeficientes = np.linalg.lstsq(normal, derecha, rcond=None)[0]
    aditivo = float(total - coeficientes @ derecha)
    completo = residual_por_medias(conteos.ravel(), sumas.ravel())
    return {
        "A": residual_por_medias(n_b, s_b) - aditivo,
        "B": residual_por_medias(n_a, s_a) - aditivo,
        "A:B": aditivo - completo,
        "residual": completo,
    }


def eta_parcial_tipo_dos(y: np.ndarray, celda: np.ndarray, forma: tuple[int, int]) -> np.ndarray:
    """η² parcial tipo II de A, B y A:B; `celda` es el código `a * k_b + b` de cada fila."""
    y = np.asarray(y, dtype=float)
    centrada = y - y.mean()
    agregados = [np.bincount(celda, weights=w, minlength=forma[0] * forma[1]).reshape(forma) for w in (None, centrada, centrada**2)]
    sumas = sumas_tipo_dos(*agregados)
    return np.array([sumas[t] / (sumas[t] + sumas["residual"]) for t in ("A", "B", "A:B")])


# Descriptivos


def medias_con_ic(y: pd.Series, por: pd.Series | pd.DataFrame, confianza: float = CONFIANZA) -> pd.DataFrame:
    """n, media y desviación por grupo, con el intervalo t de la media (el de `t.test` de una muestra).

    `por` es una serie o un marco con una columna por factor; los grupos con faltante se omiten.
    """
    claves = [por[c] for c in por.columns] if isinstance(por, pd.DataFrame) else [por]
    resumen = pd.Series(np.asarray(y, dtype=float), index=y.index).groupby(claves, observed=True).agg(n="size", media="mean", desviacion="std")
    margen = stats.t.ppf(0.5 + confianza / 2, resumen["n"] - 1) * resumen["desviacion"] / np.sqrt(resumen["n"])
    return resumen.assign(inferior=resumen["media"] - margen, superior=resumen["media"] + margen)


# Pruebas


def welch(a: np.ndarray, b: np.ndarray, confianza: float = CONFIANZA) -> Contraste:
    """t de Welch de a frente a b, con la diferencia de medias y la d de Cohen con sus intervalos."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    resultado = stats.ttest_ind(a, b, equal_var=False)
    diferencia = resultado.confidence_interval(confianza)
    d = d_cohen(a, b)
    return Contraste(
        prueba="t de Welch",
        estadistico=float(resultado.statistic),
        gl=(float(resultado.df),),
        p=float(resultado.pvalue),
        n=len(a) + len(b),
        efecto="d de Cohen",
        valor=d,
        ic=ic_d_cohen(d, len(a), len(b), confianza),
        detalle={
            "media_a": float(a.mean()),
            "media_b": float(b.mean()),
            "n_a": len(a),
            "n_b": len(b),
            "diferencia": float(a.mean() - b.mean()),
            "diferencia_inferior": float(diferencia.low),
            "diferencia_superior": float(diferencia.high),
        },
    )


def levene(y: np.ndarray, grupos: Sequence) -> Contraste:
    """Levene centrado en la mediana (Brown-Forsythe)."""
    y = np.asarray(y, dtype=float)
    niveles, codigo = codigos(grupos)
    resultado = stats.levene(*(y[codigo == i] for i in range(len(niveles))), center="median")
    gl = (float(len(niveles) - 1), float(len(y) - len(niveles)))
    return Contraste("Levene con mediana", float(resultado.statistic), gl, float(resultado.pvalue), len(y))


def anova_welch(y: np.ndarray, grupos: Sequence, replicas: int = REPLICAS, semilla: int = SEMILLA) -> Contraste:
    """ANOVA de Welch (la de `oneway.test` de R) con η² y su intervalo bootstrap."""
    y = np.asarray(y, dtype=float)
    niveles, codigo = codigos(grupos)
    resultado = anova_oneway(y, codigo, use_var="unequal", welch_correction=True)
    ic = intervalo_bootstrap(eta_cuadrado, y, codigo, replicas=replicas, semilla=semilla)
    return Contraste(
        prueba="ANOVA de Welch",
        estadistico=float(resultado.statistic),
        gl=tuple(float(g) for g in resultado.df),
        p=float(resultado.pvalue),
        n=len(y),
        efecto="η²",
        valor=eta_cuadrado(y, codigo),
        ic=_par(*ic),
        detalle={"grupos": len(niveles)},
    )


def kruskal(y: np.ndarray, grupos: Sequence, replicas: int = REPLICAS, semilla: int = SEMILLA) -> Contraste:
    """Kruskal-Wallis con corrección por empates (la de `kruskal.test`) y ε² con su intervalo."""
    y = np.asarray(y, dtype=float)
    niveles, codigo = codigos(grupos)

    def h_y_p(valores: np.ndarray, codigo: np.ndarray) -> tuple[float, float]:
        resultado = stats.kruskal(*(valores[codigo == i] for i in np.unique(codigo)))
        return float(resultado.statistic), float(resultado.pvalue)

    h, p = h_y_p(y, codigo)
    ic = intervalo_bootstrap(
        lambda valores, codigo: epsilon_cuadrado(h_y_p(valores, codigo)[0], len(valores)), y, codigo, replicas=replicas, semilla=semilla
    )
    return Contraste(
        prueba="Kruskal-Wallis",
        estadistico=h,
        gl=(len(niveles) - 1.0,),
        p=p,
        n=len(y),
        efecto="ε²",
        valor=epsilon_cuadrado(h, len(y)),
        ic=_par(*ic),
    )


def tukey(y: np.ndarray, grupos: Sequence, orden: Sequence[str], varianzas_iguales: bool = True, confianza: float = CONFIANZA) -> pd.DataFrame:
    """Comparaciones por pares de Tukey HSD o, con `varianzas_iguales=False`, de Games-Howell.

    Cada fila compara `grupo_b` con `grupo_a` (b posterior en `orden`) y su diferencia es la
    media de b menos la de a, como el `TukeyHSD` de R con los niveles en ese orden.
    """
    y, grupos = np.asarray(y, dtype=float), np.asarray(grupos)
    resultado = stats.tukey_hsd(*(y[grupos == g] for g in orden), equal_var=varianzas_iguales)
    intervalo = resultado.confidence_interval(confianza)
    filas = [
        {
            "grupo_a": orden[i],
            "grupo_b": orden[j],
            "diferencia": float(resultado.statistic[j, i]),
            "inferior": float(intervalo.low[j, i]),
            "superior": float(intervalo.high[j, i]),
            "p": float(resultado.pvalue[j, i]),
        }
        for i in range(len(orden))
        for j in range(i + 1, len(orden))
    ]
    return pd.DataFrame(filas)


def dos_vias(y: pd.Series, a: pd.Series, b: pd.Series, replicas: int = REPLICAS, semilla: int = SEMILLA) -> pd.DataFrame:
    """ANOVA de dos vías con interacción, tipo II, clásica y con HC3, y η² parcial con su intervalo.

    Las filas son A, B y A:B con los nombres de las series. Las sumas de cuadrados de la
    tabla salen de statsmodels; el η² parcial, de `sumas_tipo_dos`, que da las mismas.
    """
    datos = pd.DataFrame({"y": np.asarray(y, dtype=float), "a": np.asarray(a), "b": np.asarray(b)})
    ajuste = smf.ols("y ~ C(a) * C(b)", datos).fit()
    clasica, robusta = anova_lm(ajuste, typ=2), anova_lm(ajuste, typ=2, robust="hc3")
    niveles_a, codigo_a = codigos(datos["a"])
    niveles_b, codigo_b = codigos(datos["b"])
    forma = (len(niveles_a), len(niveles_b))
    celda = codigo_a * forma[1] + codigo_b
    valores = eta_parcial_tipo_dos(datos["y"], celda, forma)
    bajo, alto = intervalo_bootstrap(lambda y, celda: eta_parcial_tipo_dos(y, celda, forma), datos["y"].to_numpy(), celda, replicas=replicas, semilla=semilla)
    terminos = {"C(a)": str(a.name), "C(b)": str(b.name), "C(a):C(b)": f"{a.name}:{b.name}"}
    return pd.DataFrame(
        {
            "termino": list(terminos.values()),
            "suma_cuadrados": clasica.loc[list(terminos), "sum_sq"].to_numpy(),
            "gl": clasica.loc[list(terminos), "df"].to_numpy(),
            "gl_residual": float(clasica.loc["Residual", "df"]),
            "F": clasica.loc[list(terminos), "F"].to_numpy(),
            "p": clasica.loc[list(terminos), "PR(>F)"].to_numpy(),
            "F_hc3": robusta.loc[list(terminos), "F"].to_numpy(),
            "p_hc3": robusta.loc[list(terminos), "PR(>F)"].to_numpy(),
            "eta_parcial": valores,
            "inferior": bajo,
            "superior": alto,
        }
    )


def efectos_simples(y: pd.Series, a: pd.Series, b: pd.Series, nivel: str, referencia: str, confianza: float = CONFIANZA) -> pd.DataFrame:
    """Brecha de `b` (`nivel` menos `referencia`) dentro de cada nivel de `a`, con Welch y d de Cohen."""
    datos = pd.DataFrame({"y": np.asarray(y, dtype=float), "a": np.asarray(a), "b": np.asarray(b)})
    filas = []
    for valor in sorted(datos["a"].unique()):
        grupo = datos.loc[datos["a"].eq(valor)]
        c = welch(grupo.loc[grupo["b"].eq(nivel), "y"], grupo.loc[grupo["b"].eq(referencia), "y"], confianza)
        filas.append({
            str(a.name): valor,
            "n": c.n,
            "diferencia": c.detalle["diferencia"],
            "inferior": c.detalle["diferencia_inferior"],
            "superior": c.detalle["diferencia_superior"],
            "t": c.estadistico,
            "gl": c.gl[0],
            "p": c.p,
            "d": c.valor,
            "d_inferior": c.ic[0],
            "d_superior": c.ic[1],
        })
    return pd.DataFrame(filas)


def chi_cuadrado(tabla: pd.DataFrame, replicas: int = REPLICAS, semilla: int = SEMILLA) -> Contraste:
    """χ² de independencia sin corrección de Yates y V de Cramér con su intervalo bootstrap."""
    observada = tabla.to_numpy(dtype=float)
    resultado = stats.chi2_contingency(observada, correction=False)
    celda = np.repeat(np.arange(observada.size), observada.ravel().astype(int))
    ic = intervalo_bootstrap(
        lambda celda: v_cramer(np.bincount(celda, minlength=observada.size).reshape(observada.shape)), celda, replicas=replicas, semilla=semilla
    )
    return Contraste(
        prueba="χ² de independencia",
        estadistico=float(resultado.statistic),
        gl=(float(resultado.dof),),
        p=float(resultado.pvalue),
        n=int(observada.sum()),
        efecto="V de Cramér",
        valor=v_cramer(observada),
        ic=_par(*ic),
        detalle={"esperada_minima": float(resultado.expected_freq.min())},
    )


def residuos_estandarizados(tabla: pd.DataFrame) -> pd.DataFrame:
    """Residuos estandarizados ajustados, los de `chisq.test()$stdres` en R."""
    observada = tabla.to_numpy(dtype=float)
    n = observada.sum()
    filas, columnas = observada.sum(axis=1, keepdims=True), observada.sum(axis=0, keepdims=True)
    esperada = filas @ columnas / n
    residuos = (observada - esperada) / np.sqrt(esperada * (1 - filas / n) * (1 - columnas / n))
    return pd.DataFrame(residuos, index=tabla.index, columns=tabla.columns)
