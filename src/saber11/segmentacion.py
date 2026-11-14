"""Análisis multivariante de la fase 3: PCA, K-Means, Ward, DBSCAN y celdas del plano.

Todo corre sobre el entrenamiento con las cinco áreas estandarizadas (sin `PUNT_GLOBAL`, que
es combinación lineal de ellas). K-Means se ajusta con todo el entrenamiento y su silueta se
mide en la muestra; Ward y DBSCAN, cuadráticos en memoria, corren en la muestra, y Ward se
extiende al resto asignando cada fila al centroide más cercano de sus grupos. Los clústeres
se numeran de 1 a K por su nivel medio, para que las etiquetas de los dos algoritmos se lean
igual. Las celdas del plano PCA agregan a los estudiantes en una rejilla en la que toda celda
no vacía tiene 10 o más, así que ninguna figura dibuja un punto por estudiante. La rejilla va
de los percentiles 2 a 98 de cada eje, con los bordes exteriores abiertos: con los percentiles
0,5 y 99,5, las colas del plano dejaban celdas con menos de 10 en todos los tamaños de la lista.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

from saber11.datos import PUNTAJES

SEMILLA = 42
AREAS = PUNTAJES[1:]
K_CANDIDATOS = range(2, 11)
# Dos K cuya silueta difiere en menos de esto se consideran empatados y gana el menor.
TOLERANCIA_SILUETA = 0.01
MINIMO_POR_GRUPO = 10
VECINOS_DBSCAN = MINIMO_POR_GRUPO
CELDAS_POR_EJE = (60, 50, 40, 30, 25, 20, 15, 10)
PERCENTILES_REJILLA = (2, 98)
HOJAS_MAXIMAS = 30
# K elegido por silueta en §17 del cuaderno de la fase 3; una comprobación lo ata a la regla.
# Mientras sea None, `saber11.intermedios --fase 3` no escribe lo del multivariante.
K_KMEANS: int | None = 2


def estandarizar(areas: pd.DataFrame) -> pd.DataFrame:
    """Puntaje z con la media y la desviación (n) de las filas recibidas, que son las de entrenamiento.

    Va en doble precisión: los puntajes llegan en float32, y el PCA y las medias de la paridad con R
    se comparan a 1e-6 relativo.
    """
    areas = areas.astype(float)
    return (areas - areas.mean()) / areas.std(ddof=0)


def pca(z: pd.DataFrame) -> PCA:
    """PCA de todas las componentes, con el signo de cada una orientado a cargas de suma positiva."""
    modelo = PCA(svd_solver="full").fit(z)
    modelo.components_ *= np.where(modelo.components_.sum(axis=1) < 0, -1.0, 1.0)[:, None]
    return modelo


def kmeans(z: pd.DataFrame, k: int, semilla: int = SEMILLA) -> KMeans:
    return KMeans(n_clusters=k, n_init=10, random_state=semilla).fit(z)


def barrido_kmeans(z: pd.DataFrame, en_muestra: np.ndarray, candidatos: Iterable[int] = K_CANDIDATOS, semilla: int = SEMILLA) -> pd.DataFrame:
    """Inercia con todo `z` y silueta en la muestra, para cada K candidato."""
    filas = []
    for k in candidatos:
        modelo = kmeans(z, k, semilla)
        silueta = silhouette_score(z.loc[en_muestra], modelo.labels_[en_muestra])
        filas.append({"k": k, "inercia": float(modelo.inertia_), "silueta": float(silueta)})
    return pd.DataFrame(filas)


def k_por_silueta(barrido: pd.DataFrame, tolerancia: float = TOLERANCIA_SILUETA) -> int:
    """K de mayor silueta; entre los que quedan a menos de `tolerancia` de ella, el menor."""
    empatados = barrido.loc[barrido["silueta"].max() - barrido["silueta"] < tolerancia, "k"]
    return int(empatados.min())


def punto_mas_lejano(x: np.ndarray, y: np.ndarray) -> int:
    """Posición del punto más lejano de la recta que une el primero y el último.

    La distancia perpendicular es proporcional a la vertical, así que el resultado no
    depende de la escala de los ejes.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    distancia = np.abs((y[-1] - y[0]) * (x - x[0]) - (x[-1] - x[0]) * (y - y[0]))
    return int(np.argmax(distancia))


def codo(barrido: pd.DataFrame) -> int:
    """K del codo: el más lejano de la recta entre el primer y el último K de la curva de inercia."""
    return int(barrido["k"].iloc[punto_mas_lejano(barrido["k"], barrido["inercia"])])


def k_distancias(z: pd.DataFrame, vecinos: int = VECINOS_DBSCAN) -> np.ndarray:
    """Distancia de cada fila a su vecino número `vecinos`, contando la propia fila, en orden creciente.

    Es la misma cuenta que `min_samples` de DBSCAN, que también incluye al punto.
    """
    distancias, _ = NearestNeighbors(n_neighbors=vecinos).fit(z).kneighbors(z)
    return np.sort(distancias[:, -1])


def eps_por_k_distancias(distancias: np.ndarray) -> float:
    """`eps` en el punto de la curva ordenada de k-distancias más lejano de la recta entre sus extremos."""
    return float(distancias[punto_mas_lejano(np.arange(len(distancias)), distancias)])


def dbscan(z: pd.DataFrame, eps: float, minimo: int = VECINOS_DBSCAN) -> np.ndarray:
    """Etiquetas de DBSCAN; -1 marca el ruido."""
    return DBSCAN(eps=eps, min_samples=minimo).fit_predict(z)


def ward(z: pd.DataFrame) -> np.ndarray:
    """Matriz de enlace de Ward, para cortar y para el dendrograma truncado."""
    return linkage(np.asarray(z, dtype=float), method="ward")


def cortar(enlace: np.ndarray, k: int) -> np.ndarray:
    """Etiquetas de 0 a k - 1 al cortar el árbol en k grupos."""
    return fcluster(enlace, k, criterion="maxclust") - 1


def asignar_por_centroides(z: pd.DataFrame, z_grupos: pd.DataFrame, etiquetas: np.ndarray) -> np.ndarray:
    """Etiqueta del centroide más cercano (de los grupos de `z_grupos`) para cada fila de `z`."""
    centroides = pd.DataFrame(np.asarray(z_grupos, dtype=float)).groupby(np.asarray(etiquetas)).mean()
    distancias = ((np.asarray(z, dtype=float)[:, None, :] - centroides.to_numpy()[None, :, :]) ** 2).sum(axis=2)
    return centroides.index.to_numpy()[distancias.argmin(axis=1)]


def ordenar(etiquetas: np.ndarray, nivel: np.ndarray) -> np.ndarray:
    """Renumera los grupos de 1 a K de menor a mayor `nivel` medio."""
    medias = pd.Series(np.asarray(nivel, dtype=float)).groupby(np.asarray(etiquetas)).mean().sort_values()
    nuevas = {etiqueta: i for i, etiqueta in enumerate(medias.index, start=1)}
    return pd.Series(np.asarray(etiquetas)).map(nuevas).to_numpy()


def perfil_relativo(z: pd.DataFrame) -> pd.DataFrame:
    """Cada fila menos su propia media: deja la forma del perfil por área y quita el nivel general."""
    return z.sub(z.mean(axis=1), axis=0)


def segmentar(z: pd.DataFrame, en_muestra: np.ndarray, k: int, semilla: int = SEMILLA) -> pd.DataFrame:
    """Clúster de K-Means (todo `z`) y de Ward (muestra, extendido por centroides) de cada fila.

    Los dos se numeran por el promedio de las cinco áreas estandarizadas.
    """
    nivel = z.mean(axis=1).to_numpy()
    etiquetas_kmeans = kmeans(z, k, semilla).labels_
    muestra = z.loc[en_muestra]
    etiquetas_ward = asignar_por_centroides(z, muestra, cortar(ward(muestra), k))
    return pd.DataFrame({"kmeans": ordenar(etiquetas_kmeans, nivel), "ward": ordenar(etiquetas_ward, nivel)}, index=z.index)


def hojas_truncadas(enlace: np.ndarray, minimo: int = MINIMO_POR_GRUPO, maximo: int = HOJAS_MAXIMAS) -> int:
    """Mayor número de hojas, hasta `maximo`, en el que cada hoja del dendrograma tiene `minimo` o más."""
    for hojas in range(maximo, 1, -1):
        if np.bincount(fcluster(enlace, hojas, criterion="maxclust"))[1:].min() >= minimo:
            return hojas
    return 1


def ari(etiquetas: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Índice de Rand ajustado entre cada par de agrupamientos de las mismas filas."""
    return pd.DataFrame([
        {"a": a, "b": b, "ari": float(adjusted_rand_score(etiquetas[a], etiquetas[b]))} for a, b in combinations(etiquetas, 2)
    ])


def _indices(valores: np.ndarray, bordes: np.ndarray) -> np.ndarray:
    """Celda de cada valor; los bordes exteriores quedan abiertos (lo de afuera va a la celda extrema)."""
    return np.digitize(valores, bordes[1:-1])


def bordes_rejilla(plano: np.ndarray, por_eje: int, percentiles: tuple[float, float] = PERCENTILES_REJILLA) -> tuple[np.ndarray, np.ndarray]:
    """Bordes de una rejilla de `por_eje` celdas iguales por eje entre los `percentiles` de cada eje."""
    limites = np.percentile(np.asarray(plano, dtype=float), percentiles, axis=0)
    return tuple(np.linspace(limites[0, j], limites[1, j], por_eje + 1) for j in range(2))


def menor_celda(plano: np.ndarray, bordes_x: np.ndarray, bordes_y: np.ndarray) -> int:
    """Estudiantes de la celda no vacía más pequeña, con los bordes exteriores abiertos."""
    plano = np.asarray(plano, dtype=float)
    conteo = np.bincount(_indices(plano[:, 0], bordes_x) * (len(bordes_y) - 1) + _indices(plano[:, 1], bordes_y))
    return int(conteo[conteo > 0].min())


def rejilla(
    plano: np.ndarray, minimo: int = MINIMO_POR_GRUPO, celdas: Iterable[int] = CELDAS_POR_EJE, percentiles: tuple[float, float] = PERCENTILES_REJILLA
) -> tuple[np.ndarray, np.ndarray]:
    """Bordes de la rejilla más fina de `celdas` en la que toda celda no vacía tiene `minimo` o más.

    Va de los `percentiles` de cada eje, con los bordes exteriores abiertos.
    """
    for por_eje in sorted(celdas, reverse=True):
        bordes_x, bordes_y = bordes_rejilla(plano, por_eje, percentiles)
        if menor_celda(plano, bordes_x, bordes_y) >= minimo:
            return bordes_x, bordes_y
    raise ValueError(f"Ninguna rejilla de {sorted(celdas)} deja todas las celdas no vacías con {minimo} o más")


def celdas_plano(plano: np.ndarray, etiquetas: np.ndarray, minimo: int = MINIMO_POR_GRUPO) -> pd.DataFrame:
    """Celdas no vacías del plano PCA con su `n`, el clúster mayoritario y su proporción.

    Los límites de cada celda son los de la rejilla; en las extremas, el percentil 2 o 98, aunque la celda
    también recoge a quienes quedan más afuera.
    """
    plano = np.asarray(plano, dtype=float)
    bordes_x, bordes_y = rejilla(plano, minimo)
    datos = pd.DataFrame({"celda_x": _indices(plano[:, 0], bordes_x), "celda_y": _indices(plano[:, 1], bordes_y), "cluster": np.asarray(etiquetas)})
    conteos = datos.groupby(["celda_x", "celda_y", "cluster"]).size().rename("n_cluster").reset_index()
    mayoritario = conteos.sort_values(["celda_x", "celda_y", "n_cluster", "cluster"], ascending=[True, True, False, True]).drop_duplicates(["celda_x", "celda_y"])
    tabla = mayoritario.merge(datos.groupby(["celda_x", "celda_y"]).size().rename("n").reset_index(), on=["celda_x", "celda_y"])
    return pd.DataFrame({
        "celda_x": tabla["celda_x"],
        "celda_y": tabla["celda_y"],
        "x_inferior": bordes_x[tabla["celda_x"]],
        "x_superior": bordes_x[tabla["celda_x"] + 1],
        "y_inferior": bordes_y[tabla["celda_y"]],
        "y_superior": bordes_y[tabla["celda_y"] + 1],
        "n": tabla["n"],
        "cluster": tabla["cluster"],
        "proporcion": tabla["n_cluster"] / tabla["n"],
    }).reset_index(drop=True)
