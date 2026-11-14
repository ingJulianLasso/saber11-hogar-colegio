"""Pruebas de saber11.hipotesis contra cálculos a mano, con datos sintéticos."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from saber11 import hipotesis


@pytest.fixture
def dos_grupos() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    return rng.normal(260, 40, 900), rng.normal(240, 55, 600)


@pytest.fixture
def tres_grupos() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    tamanos, medias, desviaciones = (400, 250, 120), (230.0, 245.0, 270.0), (35.0, 45.0, 60.0)
    return pd.DataFrame({
        "y": np.concatenate([rng.normal(m, s, n) for n, m, s in zip(tamanos, medias, desviaciones, strict=True)]),
        "grupo": np.repeat(["bajo", "medio", "alto"], tamanos),
    })


@pytest.fixture
def dos_factores() -> pd.DataFrame:
    """Diseño desbalanceado de 3 × 2 con interacción, para que el tipo II no coincida con el I."""
    rng = np.random.default_rng(5)
    a = rng.choice(["E1", "E2", "E3"], size=1500, p=[0.5, 0.3, 0.2])
    b = np.where(rng.random(1500) < np.select([a == "E1", a == "E2"], [0.1, 0.3], 0.6), "NO OFICIAL", "OFICIAL")
    efecto = pd.Series(a).map({"E1": 0, "E2": 10, "E3": 25}).to_numpy() + np.where(b == "NO OFICIAL", 20 - 5 * (a == "E3"), 0)
    return pd.DataFrame({"y": 230 + efecto + rng.normal(0, 30, 1500), "estrato": a, "naturaleza": b})


def test_welch_y_d_de_cohen_coinciden_con_el_calculo_a_mano(dos_grupos):
    a, b = dos_grupos
    va, vb, na, nb = a.var(ddof=1), b.var(ddof=1), len(a), len(b)
    t = (a.mean() - b.mean()) / np.sqrt(va / na + vb / nb)
    gl = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    d = (a.mean() - b.mean()) / np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    error_d = np.sqrt((na + nb) / (na * nb) + d**2 / (2 * (na + nb)))
    c = hipotesis.welch(a, b)
    assert c.estadistico == pytest.approx(t)
    assert c.gl[0] == pytest.approx(gl)
    assert c.p == pytest.approx(2 * stats.t.sf(abs(t), gl))
    assert c.valor == pytest.approx(d)
    assert c.ic == pytest.approx((d - 1.959964 * error_d, d + 1.959964 * error_d), rel=1e-6)
    margen = stats.t.ppf(0.975, gl) * np.sqrt(va / na + vb / nb)
    assert (c.detalle["diferencia_inferior"], c.detalle["diferencia_superior"]) == pytest.approx((a.mean() - b.mean() - margen, a.mean() - b.mean() + margen))


def test_anova_de_welch_y_eta_cuadrado_coinciden_con_el_calculo_a_mano(tres_grupos):
    grupos = tres_grupos.groupby("grupo")["y"]
    n, m, s2 = grupos.size(), grupos.mean(), grupos.var(ddof=1)
    k = len(n)
    w = n / s2
    media_ponderada = (w * m).sum() / w.sum()
    suma = ((1 - w / w.sum()) ** 2 / (n - 1)).sum()
    f = ((w * (m - media_ponderada) ** 2).sum() / (k - 1)) / (1 + 2 * (k - 2) / (k**2 - 1) * suma)
    y = tres_grupos["y"]
    eta = (n * (m - y.mean()) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    c = hipotesis.anova_welch(y, tres_grupos["grupo"], replicas=0)
    assert c.estadistico == pytest.approx(f)
    assert c.gl == pytest.approx((k - 1, (k**2 - 1) / (3 * suma)))
    assert c.valor == pytest.approx(eta)


def test_epsilon_cuadrado_de_kruskal_wallis_coincide_con_el_calculo_a_mano(tres_grupos):
    y, grupo = tres_grupos["y"].to_numpy(), tres_grupos["grupo"].to_numpy()
    rangos, n = stats.rankdata(y), len(y)
    h = 12 / (n * (n + 1)) * sum(rangos[grupo == g].sum() ** 2 / np.sum(grupo == g) for g in np.unique(grupo)) - 3 * (n + 1)
    c = hipotesis.kruskal(y, grupo, replicas=0)
    assert c.estadistico == pytest.approx(h)
    assert c.valor == pytest.approx(h / ((n**2 - 1) / (n + 1)))
    assert c.gl == (2.0,)


def test_chi_cuadrado_sin_yates_y_v_de_cramer_coinciden_con_el_calculo_a_mano():
    tabla = pd.DataFrame([[20, 30], [40, 10]], index=["Si", "No"], columns=["URBANO", "RURAL"])
    observada = tabla.to_numpy(dtype=float)
    esperada = np.outer(observada.sum(axis=1), observada.sum(axis=0)) / observada.sum()
    chi2 = ((observada - esperada) ** 2 / esperada).sum()
    c = hipotesis.chi_cuadrado(tabla, replicas=0)
    assert c.estadistico == pytest.approx(chi2)
    assert c.estadistico != pytest.approx(stats.chi2_contingency(observada, correction=True).statistic)
    assert c.valor == pytest.approx(np.sqrt(chi2 / (observada.sum() * 1)))
    assert c.detalle["esperada_minima"] == pytest.approx(esperada.min())


def test_residuos_estandarizados_son_los_ajustados():
    tabla = pd.DataFrame([[30, 10, 5], [20, 25, 30]], index=["E1", "E2"], columns=["a", "b", "c"])
    o = tabla.to_numpy(dtype=float)
    n, fila, columna = o.sum(), o[0].sum(), o[:, 1].sum()
    esperada = fila * columna / n
    a_mano = (o[0, 1] - esperada) / np.sqrt(esperada * (1 - fila / n) * (1 - columna / n))
    residuos = hipotesis.residuos_estandarizados(tabla)
    assert residuos.loc["E1", "b"] == pytest.approx(a_mano)
    assert residuos.loc["E2", "b"] == pytest.approx(-a_mano)


def test_sumas_tipo_dos_coinciden_con_statsmodels(dos_factores):
    referencia = anova_lm(ols("y ~ C(estrato) * C(naturaleza)", dos_factores).fit(), typ=2)
    tabla = hipotesis.dos_vias(dos_factores["y"], dos_factores["estrato"], dos_factores["naturaleza"], replicas=0)
    residual = referencia.loc["Residual", "sum_sq"]
    for termino, fila in zip(["C(estrato)", "C(naturaleza)", "C(estrato):C(naturaleza)"], tabla.itertuples(), strict=True):
        suma = referencia.loc[termino, "sum_sq"]
        assert fila.suma_cuadrados == pytest.approx(suma)
        assert fila.eta_parcial == pytest.approx(suma / (suma + residual), rel=1e-6)
    assert list(tabla["termino"]) == ["estrato", "naturaleza", "estrato:naturaleza"]
    assert tabla["F_hc3"].notna().all() and not np.allclose(tabla["F_hc3"], tabla["F"])


def test_medias_con_ic_dan_el_intervalo_t_de_cada_grupo_y_de_cada_celda(dos_factores):
    por_estrato = hipotesis.medias_con_ic(dos_factores["y"], dos_factores["estrato"])
    grupo = dos_factores.loc[dos_factores["estrato"].eq("E2"), "y"]
    intervalo = stats.ttest_1samp(grupo, 0).confidence_interval(0.95)
    assert por_estrato.loc["E2", "n"] == len(grupo)
    assert por_estrato.loc["E2", ["media", "desviacion"]].tolist() == pytest.approx([grupo.mean(), grupo.std(ddof=1)])
    assert por_estrato.loc["E2", ["inferior", "superior"]].tolist() == pytest.approx([intervalo.low, intervalo.high])
    por_celda = hipotesis.medias_con_ic(dos_factores["y"], dos_factores[["estrato", "naturaleza"]])
    assert por_celda["n"].sum() == len(dos_factores)
    assert por_celda.loc[("E3", "OFICIAL"), "media"] == pytest.approx(dos_factores.query("estrato == 'E3' and naturaleza == 'OFICIAL'")["y"].mean())


def test_efectos_simples_dan_la_brecha_dentro_de_cada_nivel(dos_factores):
    tabla = hipotesis.efectos_simples(dos_factores["y"], dos_factores["estrato"], dos_factores["naturaleza"], "NO OFICIAL", "OFICIAL")
    medias = dos_factores.groupby(["estrato", "naturaleza"])["y"].mean()
    for fila in tabla.itertuples():
        assert fila.diferencia == pytest.approx(medias[(fila.estrato, "NO OFICIAL")] - medias[(fila.estrato, "OFICIAL")])
        assert fila.inferior < fila.diferencia < fila.superior


def test_tukey_resta_el_grupo_posterior_y_games_howell_no_supone_varianzas_iguales(tres_grupos):
    orden = ["bajo", "medio", "alto"]
    tukey = hipotesis.tukey(tres_grupos["y"], tres_grupos["grupo"], orden)
    medias = tres_grupos.groupby("grupo")["y"].mean()
    fila = tukey.loc[tukey["grupo_a"].eq("bajo") & tukey["grupo_b"].eq("alto")].iloc[0]
    assert fila["diferencia"] == pytest.approx(medias["alto"] - medias["bajo"])
    assert len(tukey) == 3
    games_howell = hipotesis.tukey(tres_grupos["y"], tres_grupos["grupo"], orden, varianzas_iguales=False)
    assert games_howell["diferencia"].to_numpy() == pytest.approx(tukey["diferencia"].to_numpy())
    assert not np.allclose(games_howell["superior"], tukey["superior"])


def test_bootstrap_es_reproducible_con_la_semilla(tres_grupos):
    y, grupo = tres_grupos["y"], tres_grupos["grupo"]
    primera = hipotesis.anova_welch(y, grupo, replicas=200, semilla=42)
    segunda = hipotesis.anova_welch(y, grupo, replicas=200, semilla=42)
    otra = hipotesis.anova_welch(y, grupo, replicas=200, semilla=7)
    assert primera.ic == segunda.ic
    assert primera.ic != otra.ic
    assert primera.ic[0] < primera.valor < primera.ic[1]


def test_bootstrap_de_v_remuestrea_la_tabla_y_es_reproducible():
    tabla = pd.DataFrame([[200, 300], [400, 100]])
    primera, segunda = hipotesis.chi_cuadrado(tabla, replicas=200), hipotesis.chi_cuadrado(tabla, replicas=200)
    assert primera.ic == segunda.ic
    assert primera.ic[0] < primera.valor < primera.ic[1]


def test_bootstrap_del_eta_parcial_cubre_los_tres_terminos(dos_factores):
    tabla = hipotesis.dos_vias(dos_factores["y"], dos_factores["estrato"], dos_factores["naturaleza"], replicas=100)
    assert ((tabla["inferior"] <= tabla["eta_parcial"]) & (tabla["eta_parcial"] <= tabla["superior"])).all()
