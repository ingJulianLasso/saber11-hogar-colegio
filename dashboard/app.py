"""Dashboard del Saber 11: panorama, serie, territorio y regresión (parte 1, fase 2); pruebas
de hipótesis y segmentación (parte 2, fase 3); y, en la fase 4, resumen, jornada, modelos,
predicción, método y lecturas por audiencia.

Lee solo los agregados de data/agregados/, donde cada archivo es un cruce completo con 10 o
más estudiantes por celda. Cada combinación de filtros responde con el cruce más fino que la
contiene; si ninguno la contiene, o si un cruce no se publicó, la app lo dice en lugar de
estimarlo. La pestaña Predicción usa el modelo de dashboard/modelo/, cuyas hojas cubren 10 o
más estudiantes, y las figuras de los modelos salen de figuras/.

Las seis pestañas de las fases 2 y 3 conservan su contenido y su nombre: Panorama, Serie y
Territorio quedan dentro de Exploración, y Regresión dentro de Modelos.

Uso local: uv run streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RAIZ = Path(__file__).resolve().parent.parent
AGREGADOS = RAIZ / "data" / "agregados"
MODELO = Path(__file__).resolve().parent / "modelo"
FIGURAS = RAIZ / "figuras"
# El paquete saber11 no se instala en la app (arrastraría torch): basta su carpeta en la ruta de importación.
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

from saber11 import tablero  # noqa: E402  (después de agregar src/ a la ruta)

TITULO = "Saber 11: hogar, colegio y puntaje global"
TODOS = "Todos"
ESTRATOS = (*(f"Estrato {n}" for n in range(1, 7)), "Sin Estrato", "Sin dato")
NATURALEZAS = ("OFICIAL", "NO OFICIAL")
ZONAS = ("URBANO", "RURAL")
BLOQUES = ("Hogar", "Colegio", "Estudiante")
CONJUNTOS = ("entrenamiento", "validación", "prueba")
MODELOS = ("Media del entrenamiento", "Lineal simple", "Regresión múltiple")
COLOR_PRINCIPAL, COLOR_APOYO, COLOR_RESALTADO = "#2C4A6E", "#BFBFBF", "#D55E00"
COLOR_AGRUPADO = "#A9C4E0"
COLOR_NATURALEZA = {"OFICIAL": "#0072B2", "NO OFICIAL": "#D55E00"}
PALETA_ESTRATOS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00", "#4C4C4C", "#999999")
COLOR_DESCRIPTIVO = "#7F7F7F"
# Los mismos colores de los clústeres que las figuras del cuaderno y del script de R de la fase 3.
COLORES_CLUSTER = ("#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#4C4C4C")
RESPUESTAS = {"Si": "Sí"}
HIPOTESIS_MEDIAS = {"H1": "H1: internet en el hogar", "H2": "H2: educación de la madre"}
METODOS_TUKEY = ("Tukey", "Games-Howell")
TABLAS_H4 = {
    "estrato_naturaleza": ("Estrato y naturaleza del colegio", "Estrato", "Naturaleza del colegio"),
    "internet_zona": ("Internet en el hogar y zona de la sede", "Internet en el hogar", "Zona de la sede"),
}
AREAS_PERFIL = {
    "lectura_critica": "Lectura crítica", "matematicas": "Matemáticas", "ciencias_naturales": "Ciencias naturales",
    "sociales_ciudadanas": "Sociales y ciudadanas", "ingles": "Inglés",
}
ESCALAS_PERFIL = ("Puntaje del área", "Estandarizado")
CRUCES_CLUSTER = {"Estrato": "estrato", "Zona": "zona", "Naturaleza del colegio": "naturaleza"}
FUENTE = (
    "Fuente: ICFES, Resultados únicos Saber 11 (datos.gov.co), licencia CC BY-SA 4.0. "
    "El dashboard solo usa agregados con 10 o más estudiantes por celda."
)
NO_PUBLICADO = (
    "La combinación de departamento con estrato o con naturaleza del colegio no está publicada: "
    "en ese cruce hay celdas con menos de 10 estudiantes, que podrían identificar a alguien. "
    "Quite uno de los dos filtros."
)

# Fase 4: pestañas, audiencias, modelos y formulario.
PESTANAS = ("Resumen", "Exploración", "Pruebas de hipótesis", "Modelos", "Predicción", "Segmentación", "Método y fuente")
SUBPESTANAS_EXPLORACION = ("Panorama", "Serie", "Territorio", "Jornada")
SUBPESTANAS_MODELOS = ("Regresión", "Con y sin IA", "Clasificación", "Interpretabilidad")
AUDIENCIAS = {
    "Secretaría de educación": "la secretaría de educación", "Rectoría": "la rectoría",
    "Familia": "las familias", "Equipo técnico": "el equipo técnico",
}
URL_REPOSITORIO = "https://github.com/ingJulianLasso/saber11-hogar-colegio"
AVISO_PERIODO = "Periodo 2022-2: el filtro de periodos no aplica."
AVISO_PREDICCION = "Estima el promedio de un perfil, no el resultado de una persona; no sirve para decidir sobre nadie."
SIN_DATO = "Sin dato"
JORNADA = "jornada-estrato-naturaleza-20224.csv"
# Los elegidos por validación en la fase 3 (mejorRegresion y mejorClasificacion) y la referencia sin IA.
MODELO_REGRESION, MODELO_CLASIFICACION, MODELO_REFERENCIA = "xgboostRegresion", "xgboostClasificacion", "regresionMultiple"
COLOR_ENFOQUE = {"Sin IA": COLOR_APOYO, "Con IA": COLOR_PRINCIPAL}
GRUPOS_FORMULARIO = {"hogar": "Hogar", "colegio": "Colegio", "estudiante": "Estudiante"}
NIVELES_LEGIBLES = {
    "COLE_BILINGUE": {"N": "No", "S": "Sí"}, "ESTU_GENERO": {"F": "Femenino", "M": "Masculino"}, "COLE_JORNADA": {"UNICA": "Única"},
}
LECTURAS = {
    "resumen": {
        "Secretaría de educación": (
            "Las tarjetas dan el tamaño del examen y cuánto se puede anticipar con datos del hogar y del colegio. El mejor modelo "
            "explica una parte de la variación del puntaje (R² de {r2} en prueba); el resto depende de lo que estos datos no recogen. "
            "Sirve para orientar recursos hacia grupos, no para clasificar colegios ni estudiantes."
        ),
        "Rectoría": (
            "Use las tarjetas de arriba con los filtros de su contexto (estrato, naturaleza, zona o departamento) para comparar su "
            "colegio con el grupo que se le parece, no con el promedio nacional."
        ),
        "Familia": (
            "El puntaje global va de 0 a 500. Las tarjetas resumen a todos los estudiantes que presentaron el examen en 2022-2; "
            "el resultado de un grupo no dice cuánto obtendrá un estudiante en particular."
        ),
        "Equipo técnico": (
            "Las métricas de prueba son las de la fase 3, con XGBoost elegido por validación y la prueba usada una sola vez por modelo. "
            "El modelo de la pestaña Predicción se reajustó con hojas de 10 o más estudiantes y solo se evaluó en validación "
            "(RMSE de {rmse_tablero} puntos)."
        ),
    },
    "exploracion": {
        "Secretaría de educación": (
            "En 2022-2, la media de los colegios no oficiales supera a la de los oficiales en {brecha_naturaleza} puntos, y la de las "
            "sedes urbanas a la de las rurales en {brecha_zona}. Territorio y Jornada ayudan a ubicar dónde se concentran los puntajes bajos."
        ),
        "Rectoría": (
            "Fije en los filtros el estrato, la naturaleza y la zona de su sede: Panorama y Jornada responden con el grupo comparable, "
            "y Serie muestra cómo ha cambiado ese grupo desde {primer_periodo}."
        ),
        "Familia": (
            "Las barras muestran la mediana de cada grupo y el rango donde está la mitad central de sus estudiantes. Esos rangos se "
            "solapan mucho: una mediana más alta no garantiza un mejor resultado a cada estudiante."
        ),
        "Equipo técnico": (
            "Cada vista lee el cruce más fino publicado con 10 o más estudiantes por celda; por eso algunos filtros no aplican en "
            "todas las subpestañas, y la app lo avisa en lugar de estimar."
        ),
    },
    "hipotesis": {
        "Secretaría de educación": (
            "Con miles de estudiantes por grupo, hasta diferencias pequeñas resultan significativas; para decidir importa el tamaño "
            "de la diferencia en puntos, que cada gráfico muestra con su intervalo."
        ),
        "Rectoría": (
            "H3 muestra si la brecha entre colegios oficiales y no oficiales cambia con el estrato: si las líneas no son paralelas, "
            "la ventaja de un tipo de colegio depende del hogar de sus estudiantes."
        ),
        "Familia": (
            "Tener internet en casa o una madre con más estudios se asocia con puntajes más altos en promedio. Es una asociación "
            "entre grupos, no una causa ni una promesa para un estudiante."
        ),
        "Equipo técnico": (
            "Cada prueba usa a los estudiantes con sus dos variables registradas; Tukey supone varianzas iguales y Games-Howell no, "
            "por eso se muestran los dos. Los supuestos y la réplica en R están en el repositorio."
        ),
    },
    "modelos": {
        "Secretaría de educación": (
            "Los modelos con IA mejoran poco a la estadística clásica: XGBoost baja el error de prueba en {mejora_rmse} puntos frente "
            "a la regresión múltiple. La regresión, más fácil de explicar, sigue siendo una buena base para comunicar resultados."
        ),
        "Rectoría": (
            "Con solo datos del hogar y del estudiante, el clasificador distingue a quienes estudian en colegios no oficiales con un "
            "AUC-ROC de {auc} en prueba: el hogar y el tipo de colegio van de la mano, algo que conviene recordar al comparar colegios."
        ),
        "Familia": (
            "Los modelos estiman el promedio de grupos de estudiantes parecidos. Dentro de cada grupo los resultados varían mucho, "
            "así que no describen a un estudiante en particular."
        ),
        "Equipo técnico": (
            "Sin IA: {sin_ia}. Con IA: {con_ia}. Las métricas de la tabla son de prueba; las curvas, de validación."
        ),
    },
    "prediccion": {
        "Secretaría de educación": (
            "La estimación sirve para explorar escenarios (por ejemplo, qué cambia con internet en el hogar), no para asignar "
            "recursos a una persona ni a un colegio."
        ),
        "Rectoría": (
            "Pruebe el perfil típico de sus estudiantes como punto de referencia; el margen de ±{margen} puntos recuerda que "
            "estudiantes con el mismo perfil obtienen resultados muy distintos."
        ),
        "Familia": (
            "La estimación es el promedio de los estudiantes con un perfil parecido al que usted describe; no predice el resultado "
            "de su hijo o de su hija."
        ),
        "Equipo técnico": (
            "XGBoost de la fase 3 reajustado en la CPU con hojas de 10 o más estudiantes. El margen es el cuantil {nivel} de los "
            "residuos absolutos de validación, con corrección de muestra finita; su cobertura es aproximada, porque la validación "
            "también eligió el modelo."
        ),
    },
    "segmentacion": {
        "Secretaría de educación": (
            "El clúster de menor nivel reúne a una parte mayor de los estudiantes de los estratos bajos y de los hogares sin estrato: "
            "es un criterio para focalizar el refuerzo."
        ),
        "Rectoría": (
            "Los perfiles de los clústeres son casi paralelos: los grupos difieren en el nivel general más que en un área concreta, "
            "así que el refuerzo rinde más si abarca todas las áreas."
        ),
        "Familia": (
            "Los clústeres agrupan a estudiantes con resultados parecidos en las cinco áreas. Pertenecer a uno no es una etiqueta "
            "para un estudiante, sino una forma de resumir patrones de grupo."
        ),
        "Equipo técnico": (
            "K-Means sobre las cinco áreas estandarizadas; el plano del PCA se publica por celdas de 10 o más estudiantes, nunca con "
            "un punto por estudiante."
        ),
    },
}


@dataclass(frozen=True)
class Filtros:
    periodos: tuple[int, int]
    departamento: str
    estrato: str
    naturaleza: str
    zona: str
    puntaje: tuple[int, int]

    @property
    def publicado(self) -> bool:
        """El departamento solo se cruza con la zona: con estrato o naturaleza no hay agregado."""
        return self.departamento == TODOS or (self.estrato == TODOS and self.naturaleza == TODOS)


@st.cache_data
def leer(nombre: str) -> pd.DataFrame:
    return pd.read_csv(AGREGADOS / nombre, dtype={"cod_depto": str})


def leer_publicado(nombre: str) -> pd.DataFrame | None:
    """El agregado, o None si no está publicado: el cuaderno no escribe un cruce con alguna celda de menos de 10."""
    return leer(nombre) if (AGREGADOS / nombre).is_file() else None


def no_publicado(cruce: str) -> None:
    st.info(f"{cruce} no está publicado: solo se publican los cruces con 10 o más estudiantes en cada celda, para que nadie pueda ser identificado.")


def numero(valor: float, decimales: int = 0) -> str:
    """Número con coma decimal y punto de miles; los de cuatro cifras van sin separador, como pide la RAE."""
    valor = round(float(valor), decimales) + 0.0  # sin «-0,000» cuando el valor se redondea a cero
    texto = f"{valor:,.{decimales}f}" if abs(valor) >= 10_000 else f"{valor:.{decimales}f}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".")


def nombre_periodo(periodo: int) -> str:
    """Los periodos de la serie son todos segundos semestres: 20194 y 20224 son 2019-2 y 2022-2."""
    return f"{str(periodo)[:4]}-2"


def capitalizar(texto: str) -> str:
    return texto.capitalize() if texto.isupper() else texto


def legible(texto: str) -> str:
    """Nivel como se lee: «Sí» con tilde y las categorías en mayúsculas con solo la inicial."""
    return RESPUESTAS.get(texto, capitalizar(texto))


def texto_p(p: float) -> str:
    return "< 0,001" if p < 0.001 else numero(p, 3)


def con_barras(columna: pd.Series, inferior: pd.Series, superior: pd.Series) -> dict:
    """Barras de error asimétricas de Plotly desde los límites de un intervalo."""
    return {"type": "data", "symmetric": False, "array": superior - columna, "arrayminus": columna - inferior, "width": 0}


def estilo(figura: go.Figure, alto: int, titulo_x: str, titulo_y: str = "") -> go.Figure:
    figura.update_layout(
        template="simple_white",
        height=alto,
        separators=",.",
        font={"family": "serif", "size": 13},
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis={"title": titulo_x, "showgrid": True, "automargin": True},
        yaxis={"title": titulo_y, "showgrid": True, "automargin": True},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    return figura


def filtros_laterales(serie: pd.DataFrame, departamentos: list[str]) -> Filtros:
    st.sidebar.header("Filtros")
    periodos = sorted(serie["periodo"].unique().tolist())
    desde, hasta = st.sidebar.select_slider(
        "Periodos (pestaña Serie)", options=periodos, value=(periodos[0], periodos[-1]), format_func=nombre_periodo, key="periodos"
    )
    departamento = st.sidebar.selectbox("Departamento de la sede", (TODOS, *departamentos), format_func=str.title, key="departamento")
    estrato = st.sidebar.selectbox("Estrato de la vivienda", (TODOS, *ESTRATOS), key="estrato")
    naturaleza = st.sidebar.selectbox("Naturaleza del colegio", (TODOS, *NATURALEZAS), format_func=capitalizar, key="naturaleza")
    zona = st.sidebar.selectbox("Zona de la sede", (TODOS, *ZONAS), format_func=capitalizar, key="zona")
    puntaje = st.sidebar.slider("Rango de puntaje global (pestaña Panorama)", 0, 500, (0, 500), step=10, key="puntaje")
    st.sidebar.caption("El departamento se cruza solo con la zona. Las demás combinaciones salen del cruce de estrato, naturaleza y zona.")
    return Filtros((desde, hasta), departamento, estrato, naturaleza, zona, puntaje)


def fila_resumen(filtros: Filtros, cubo: pd.DataFrame, departamento_zona: pd.DataFrame) -> pd.Series:
    """La celda del agregado que responde a los filtros: departamento y zona, o estrato, naturaleza y zona."""
    if filtros.departamento != TODOS:
        mascara = departamento_zona["departamento"].eq(filtros.departamento) & departamento_zona["zona"].eq(filtros.zona)
        return departamento_zona.loc[mascara].iloc[0]
    mascara = cubo["estrato"].eq(filtros.estrato) & cubo["naturaleza"].eq(filtros.naturaleza) & cubo["zona"].eq(filtros.zona)
    return cubo.loc[mascara].iloc[0]


def brecha_naturaleza(filtros: Filtros, cubo: pd.DataFrame) -> float:
    medias = cubo.loc[cubo["estrato"].eq(filtros.estrato) & cubo["zona"].eq(filtros.zona)].set_index("naturaleza")["media"]
    return float(medias["NO OFICIAL"] - medias["OFICIAL"])


def tarjetas(filtros: Filtros, cubo: pd.DataFrame, departamento_zona: pd.DataFrame) -> None:
    fila = fila_resumen(filtros, cubo, departamento_zona)
    sin_departamento = filtros.departamento == TODOS
    columnas = st.columns(4 if sin_departamento else 3)
    columnas[0].metric("Estudiantes", numero(fila["n"]))
    columnas[1].metric("Media del puntaje global", numero(fila["media"], 1))
    columnas[2].metric("Mediana", numero(fila["mediana"]), help=f"Rango intercuartílico: {numero(fila['q1'])} a {numero(fila['q3'])}")
    if sin_departamento:
        columnas[3].metric(
            "Brecha no oficial frente a oficial",
            f"{numero(brecha_naturaleza(filtros, cubo), 1)} puntos",
            help="Diferencia de medias entre colegios no oficiales y oficiales, con el estrato y la zona elegidos.",
        )


def panorama(filtros: Filtros, cubo: pd.DataFrame, distribucion: pd.DataFrame) -> None:
    st.subheader("Mediana por estrato")
    if filtros.departamento != TODOS:
        st.info("Este gráfico sale del cruce de estrato, naturaleza y zona, que no se publica por departamento; se muestra para todo el país.")
    por_estrato = cubo.loc[cubo["estrato"].ne(TODOS) & cubo["naturaleza"].eq(filtros.naturaleza) & cubo["zona"].eq(filtros.zona)]
    por_estrato = por_estrato.set_index("estrato").reindex(ESTRATOS).reset_index()
    colores = [COLOR_RESALTADO if e == filtros.estrato else COLOR_PRINCIPAL for e in por_estrato["estrato"]]
    figura = go.Figure(go.Bar(
        x=por_estrato["estrato"], y=por_estrato["mediana"], marker_color=colores,
        error_y={"type": "data", "symmetric": False, "array": por_estrato["q3"] - por_estrato["mediana"], "arrayminus": por_estrato["mediana"] - por_estrato["q1"], "color": COLOR_APOYO},
        customdata=por_estrato[["n", "q1", "q3"]], hovertemplate="%{x}: mediana %{y}, cuartiles %{customdata[1]} a %{customdata[2]} (%{customdata[0]:,.0f} estudiantes)<extra></extra>",
    ))
    st.plotly_chart(estilo(figura, 360, "", "Puntaje global (mediana y cuartiles)"), key="grafico_estrato")
    st.caption(f"Naturaleza: {capitalizar(filtros.naturaleza)}. Zona: {capitalizar(filtros.zona)}. Las barras grises van del primer al tercer cuartil.")

    st.subheader("Distribución del puntaje global")
    if filtros.estrato != TODOS or filtros.departamento != TODOS:
        st.info("La distribución por intervalos se publica solo por naturaleza y zona, así que no aplica el filtro de estrato ni el de departamento.")
    celdas = distribucion.loc[distribucion["naturaleza"].eq(filtros.naturaleza) & distribucion["zona"].eq(filtros.zona)].sort_values("desde")
    minimo, maximo = filtros.puntaje
    en_rango = celdas["desde"].ge(minimo) & celdas["hasta"].le(maximo)
    porcentaje = 100 * celdas.loc[en_rango, "n"].sum() / celdas["n"].sum()
    st.metric(f"Estudiantes con puntaje entre {minimo} y {maximo}", f"{numero(porcentaje, 1)} %", help="Cuenta los intervalos que caen completos dentro del rango; los de los extremos son abiertos.")
    figura = go.Figure(go.Bar(
        x=celdas["intervalo"], y=celdas["n"], marker_color=[COLOR_PRINCIPAL if dentro else COLOR_APOYO for dentro in en_rango],
        hovertemplate="%{x}: %{y:,.0f} estudiantes<extra></extra>",
    ))
    figura.update_xaxes(tickangle=-45, type="category")
    st.plotly_chart(estilo(figura, 360, "Puntaje global", "Estudiantes"), key="grafico_distribucion")


def serie_temporal(filtros: Filtros, serie: pd.DataFrame) -> None:
    st.subheader("Mediana del puntaje global por periodo")
    if filtros.departamento != TODOS or filtros.zona != TODOS:
        st.info("La serie se publica por estrato y naturaleza, así que no aplica el filtro de departamento ni el de zona.")
    desde, hasta = filtros.periodos
    vista = serie.loc[serie["periodo"].between(desde, hasta)]
    if filtros.estrato != TODOS:
        vista = vista.loc[vista["estrato"].eq(filtros.estrato)]
    if filtros.naturaleza != TODOS:
        vista = vista.loc[vista["naturaleza"].eq(filtros.naturaleza)]
    if vista.empty:
        st.info("El estrato elegido no aparece en la serie de esos periodos.")
        return
    figura = go.Figure()
    orden = [e for e in ESTRATOS if e in set(vista["estrato"])]
    for (estrato, naturaleza), grupo in vista.groupby(["estrato", "naturaleza"]):
        figura.add_trace(go.Scatter(
            x=grupo["periodo"].map(nombre_periodo), y=grupo["mediana"], mode="lines+markers", name=f"{estrato}, {capitalizar(naturaleza)}",
            line={"color": PALETA_ESTRATOS[orden.index(estrato)], "dash": "solid" if naturaleza == "OFICIAL" else "dot"},
            hovertemplate="%{x}: mediana %{y}<extra>" + f"{estrato}, {capitalizar(naturaleza)}" + "</extra>",
        ))
    figura.update_xaxes(type="category")
    st.plotly_chart(estilo(figura, 420, "Periodo", "Mediana del puntaje global"), key="grafico_serie")
    st.caption("Línea continua: colegios oficiales; punteada: no oficiales. Desde 2017-2 el formulario agrega la respuesta «Sin Estrato».")


def territorio(filtros: Filtros, departamento_zona: pd.DataFrame) -> None:
    st.subheader(f"Mediana por departamento, zona: {capitalizar(filtros.zona).lower()}")
    if filtros.estrato != TODOS or filtros.naturaleza != TODOS:
        st.info("Los departamentos se publican solo por zona, así que no aplica el filtro de estrato ni el de naturaleza.")
    vista = departamento_zona.loc[departamento_zona["zona"].eq(filtros.zona)].sort_values("mediana")
    colores = [COLOR_RESALTADO if d == filtros.departamento else COLOR_PRINCIPAL for d in vista["departamento"]]
    figura = go.Figure(go.Bar(
        x=vista["mediana"], y=vista["departamento"].str.title(), orientation="h", marker_color=colores,
        customdata=vista[["n", "media"]], hovertemplate="%{y}: mediana %{x}, media %{customdata[1]:.1f} (%{customdata[0]:,.0f} estudiantes)<extra></extra>",
    ))
    st.plotly_chart(estilo(figura, 760, "Mediana del puntaje global"), key="grafico_territorio")


def regresion(filtros: Filtros, metricas: pd.DataFrame, coeficientes: pd.DataFrame, grupos: pd.DataFrame) -> None:
    multiple = metricas.loc[metricas["modelo"].eq("Regresión múltiple")].set_index("conjunto")
    columnas = st.columns(4)
    columnas[0].metric("R² en validación", numero(multiple.loc["validación", "r2"], 3))
    columnas[1].metric("R² en prueba", numero(multiple.loc["prueba", "r2"], 3))
    columnas[2].metric("RMSE en validación", f"{numero(multiple.loc['validación', 'rmse'], 1)} puntos")
    columnas[3].metric("RMSE en prueba", f"{numero(multiple.loc['prueba', 'rmse'], 1)} puntos")
    st.subheader("Coeficientes de la regresión múltiple")
    elegidos = st.multiselect("Bloques", BLOQUES, default=BLOQUES, key="bloques")
    con_departamentos = st.checkbox("Incluir los departamentos", value=False, key="departamentos")
    con_faltantes = st.checkbox("Incluir los indicadores de faltante", value=False, key="faltantes")
    vista = coeficientes.loc[coeficientes["bloque"].isin(elegidos)]
    if not con_departamentos:
        vista = vista.loc[vista["variable"].ne("Departamento")]
    if not con_faltantes:
        vista = vista.loc[~vista["variable"].str.startswith("Sin dato")]
    if vista.empty:
        st.info("Elija al menos un bloque.")
    else:
        vista = vista.sort_values("coeficiente").assign(etiqueta=lambda m: (m["variable"] + ": " + m["nivel"].fillna("")).str.rstrip(": "))
        agrupado = vista.dropna(subset=["ic_agrupado_inferior"])
        figura = go.Figure()
        figura.add_trace(go.Scatter(
            x=agrupado["coeficiente"], y=agrupado["etiqueta"], mode="markers", marker={"symbol": "square", "size": 8, "color": COLOR_AGRUPADO}, name="IC agrupado por departamento",
            error_x={"type": "data", "array": agrupado["ic_agrupado_superior"] - agrupado["coeficiente"], "arrayminus": agrupado["coeficiente"] - agrupado["ic_agrupado_inferior"], "width": 0, "thickness": 7, "color": COLOR_AGRUPADO},
            hoverinfo="skip",
        ))
        figura.add_trace(go.Scatter(
            x=vista["coeficiente"], y=vista["etiqueta"], mode="markers", marker={"color": COLOR_PRINCIPAL, "size": 7}, name="Coeficiente e IC clásico",
            error_x={"type": "data", "array": vista["ic_superior"] - vista["coeficiente"], "arrayminus": vista["coeficiente"] - vista["ic_inferior"], "width": 0},
            customdata=vista[["ic_inferior", "ic_superior", "ic_agrupado_inferior", "ic_agrupado_superior", "n"]],
            hovertemplate="%{y}: %{x:.1f} puntos; IC clásico de %{customdata[0]:.1f} a %{customdata[1]:.1f}; IC agrupado de %{customdata[2]:.1f} a %{customdata[3]:.1f}; %{customdata[4]:,.0f} estudiantes<extra></extra>",
        ))
        figura.add_vline(x=0, line={"color": COLOR_APOYO})
        figura.update_yaxes(tickmode="array", tickvals=vista["etiqueta"].tolist())  # una etiqueta por término, sin saltos
        st.plotly_chart(estilo(figura, max(340, 26 * len(vista)), "Puntos frente a la referencia, con lo demás constante"), key="grafico_coeficientes")
    st.caption(
        "Barra ancha: intervalo del 95 % con errores agrupados por departamento, el que conviene leer, porque los estudiantes de un mismo "
        "territorio no son independientes. Barra fina: intervalo clásico. Los departamentos solo llevan el clásico, porque sus propios "
        "errores no se pueden agrupar por departamento. Son asociaciones, no efectos causales: el modelo compara hogares y colegios que ya son distintos."
    )

    st.subheader("Observado frente a predicho en validación")
    vista = grupos if filtros.naturaleza == TODOS or "naturaleza" not in grupos else grupos.loc[grupos["naturaleza"].eq(filtros.naturaleza)]
    vista = vista.assign(orden=vista["estrato"].map({e: i for i, e in enumerate(ESTRATOS)})).sort_values("orden")
    figura = go.Figure()
    series = vista.groupby("naturaleza") if "naturaleza" in vista else [("", vista)]
    for naturaleza, grupo in series:
        color = COLOR_NATURALEZA.get(naturaleza, COLOR_PRINCIPAL)
        nombre = capitalizar(naturaleza) if naturaleza else "Todos"
        figura.add_trace(go.Scatter(x=grupo["estrato"], y=grupo["observado"], mode="markers", name=f"Observado, {nombre}", marker={"color": color, "size": 10}))
        figura.add_trace(go.Scatter(x=grupo["estrato"], y=grupo["predicho"], mode="markers", name=f"Predicho, {nombre}", marker={"color": color, "size": 10, "symbol": "x"}))
    figura.update_xaxes(type="category")
    st.plotly_chart(estilo(figura, 380, "", "Puntaje global medio"), key="grafico_grupos")
    st.caption("Medias por grupo en el conjunto de validación, que el modelo no vio al ajustarse.")

    st.subheader("Métricas de los tres modelos de regresión")
    tabla = metricas.assign(
        orden_modelo=metricas["modelo"].map(MODELOS.index), orden_conjunto=metricas["conjunto"].map(CONJUNTOS.index)
    ).sort_values(["orden_modelo", "orden_conjunto"])
    st.dataframe(
        pd.DataFrame({
            "Modelo": tabla["modelo"],
            "Conjunto": tabla["conjunto"].str.capitalize(),
            "Estudiantes": tabla["n"].map(numero),
            "R²": tabla["r2"].map(lambda valor: numero(valor, 3)),
            "RMSE": tabla["rmse"].map(lambda valor: numero(valor, 2)),
            "MAE": tabla["mae"].map(lambda valor: numero(valor, 2)),
        }),
        hide_index=True, key="tabla_metricas",
    )


def medias_por_grupo(medias: pd.DataFrame | None) -> None:
    st.subheader("Medias por grupo con su intervalo de confianza (H1 y H2)")
    if medias is None:
        no_publicado("El agregado de medias de H1 y H2")
        return
    hipotesis = st.radio("Hipótesis", list(HIPOTESIS_MEDIAS), format_func=HIPOTESIS_MEDIAS.get, horizontal=True, key="hipotesis_medias")
    vista = medias.loc[medias["hipotesis"].eq(hipotesis)].sort_values("orden")
    figura = go.Figure(go.Bar(
        x=2 * vista["desviacion"], base=vista["media"] - vista["desviacion"], y=vista["etiqueta"], orientation="h", width=0.35,
        marker_color=COLOR_APOYO, name="Media ± 1 desviación estándar", hoverinfo="skip",
    ))
    for probado, nombre in ((True, "Grupo de la prueba"), (False, "Grupo descriptivo, fuera de la prueba")):
        parte = vista.loc[vista["probado"].eq(probado)]
        figura.add_trace(go.Scatter(
            x=parte["media"], y=parte["etiqueta"], mode="markers", name=nombre, error_x=con_barras(parte["media"], parte["inferior"], parte["superior"]),
            marker={"color": COLOR_PRINCIPAL if probado else COLOR_DESCRIPTIVO, "symbol": "circle" if probado else "circle-open", "size": 9},
            customdata=parte[["n", "desviacion", "inferior", "superior"]],
            hovertemplate="%{y}: media %{x:.1f} (IC del 95 %: %{customdata[2]:.1f} a %{customdata[3]:.1f}); desviación %{customdata[1]:.1f}; %{customdata[0]:,.0f} estudiantes<extra></extra>",
        ))
    figura.update_yaxes(categoryorder="array", categoryarray=vista["etiqueta"].tolist()[::-1])
    st.plotly_chart(estilo(figura, max(300, 34 * len(vista)), "Puntaje global (media e IC del 95 %)"), key="grafico_medias")
    st.caption(
        "El intervalo de cada media es más angosto que su marcador porque los grupos tienen miles de estudiantes; la barra gris muestra "
        "cuánto se solapan los puntajes de grupos cuyas medias difieren. Los grupos descriptivos («Sin dato», «No sabe» y «No aplica») no entran en la prueba."
    )


def pares_tukey(tukey: pd.DataFrame | None) -> None:
    st.subheader("Comparaciones por pares de la educación de la madre (H2)")
    if tukey is None:
        no_publicado("El agregado de comparaciones por pares")
        return
    columnas = st.columns(2)
    metodos = columnas[0].multiselect("Método", METODOS_TUKEY, default=list(METODOS_TUKEY), key="metodos_tukey")
    solo_consecutivos = columnas[1].checkbox("Solo niveles consecutivos", value=True, key="consecutivos")
    vista = tukey.loc[tukey["metodo"].isin(metodos) & (tukey["consecutivo"] | (not solo_consecutivos))]
    if vista.empty:
        st.info("Elija al menos un método.")
        return
    figura = go.Figure()
    for metodo, color in zip(METODOS_TUKEY, (COLOR_PRINCIPAL, COLOR_RESALTADO), strict=True):
        parte = vista.loc[vista["metodo"].eq(metodo)]
        figura.add_trace(go.Scatter(
            x=parte["diferencia"], y=parte["par"], orientation="h", mode="markers", name=metodo, marker={"color": color, "size": 7},
            error_x=con_barras(parte["diferencia"], parte["inferior"], parte["superior"]),
            customdata=parte[["inferior", "superior"]].assign(p=parte["p"].map(texto_p), n=parte["n"]),
            hovertemplate="%{y}: %{x:.1f} puntos (%{customdata[0]:.1f} a %{customdata[1]:.1f}); p ajustado %{customdata[2]}<extra>" + metodo + "</extra>",
        ))
    figura.add_vline(x=0, line={"color": COLOR_APOYO})
    figura.update_layout(scattermode="group")
    figura.update_yaxes(categoryorder="array", categoryarray=vista["par"].unique().tolist()[::-1])
    st.plotly_chart(estilo(figura, max(320, 30 * vista["par"].nunique()), "Diferencia de medias (puntos), intervalo simultáneo del 95 %"), key="grafico_tukey")
    st.caption("Cada par compara el nivel de la derecha con el de la izquierda. Games-Howell no supone varianzas iguales entre los grupos; Tukey sí.")


def interaccion_estrato(interaccion: pd.DataFrame | None) -> None:
    st.subheader("Puntaje por estrato y naturaleza del colegio (H3)")
    if interaccion is None:
        no_publicado("El agregado de medias por estrato y naturaleza")
        return
    estratos = [e for e in ESTRATOS if e in set(interaccion["estrato"])]
    figura = go.Figure()
    for naturaleza in NATURALEZAS:
        celdas = interaccion.loc[interaccion["naturaleza"].eq(naturaleza)].set_index("estrato").reindex(estratos)
        figura.add_trace(go.Scatter(
            x=estratos, y=celdas["media"], mode="lines+markers", name=capitalizar(naturaleza), line={"color": COLOR_NATURALEZA[naturaleza]},
            error_y=con_barras(celdas["media"], celdas["inferior"], celdas["superior"]), customdata=celdas[["n", "inferior", "superior"]],
            hovertemplate="%{x}: media %{y:.1f} (IC del 95 %: %{customdata[1]:.1f} a %{customdata[2]:.1f}); %{customdata[0]:,.0f} estudiantes<extra>%{fullData.name}</extra>",
        ))
    figura.update_xaxes(type="category")
    st.plotly_chart(estilo(figura, 400, "", "Puntaje global medio (IC del 95 %)"), key="grafico_interaccion")
    st.caption("Si las dos líneas fueran paralelas, la brecha entre colegios oficiales y no oficiales sería la misma en cada estrato.")


def residuos_independencia(residuos: pd.DataFrame | None) -> None:
    st.subheader("Residuos de las pruebas de independencia (H4)")
    if residuos is None:
        no_publicado("El agregado de residuos de H4")
        return
    tabla = st.radio("Tabla", list(TABLAS_H4), format_func=lambda t: TABLAS_H4[t][0], horizontal=True, key="tabla_h4")
    vista = residuos.loc[residuos["tabla"].eq(tabla)]
    filas, columnas = vista["fila"].unique(), vista["columna"].unique()
    valores = vista.pivot(index="fila", columns="columna", values="residuo").reindex(index=filas, columns=columnas)
    conteos = vista.pivot(index="fila", columns="columna", values="n").reindex(index=filas, columns=columnas)
    limite = residuos["residuo"].abs().max()
    _, nombre_filas, nombre_columnas = TABLAS_H4[tabla]
    figura = go.Figure(go.Heatmap(
        z=valores.to_numpy(), x=[legible(c) for c in columnas], y=[legible(f) for f in filas], customdata=conteos.to_numpy(),
        zmid=0, zmin=-limite, zmax=limite, colorscale="RdBu_r", texttemplate="%{z:.1f}", colorbar={"title": {"text": "Residuo ajustado", "side": "right"}},
        hovertemplate=f"{nombre_filas} %{{y}}, {nombre_columnas.lower()} %{{x}}: residuo %{{z:.1f}}; %{{customdata:,.0f}} estudiantes<extra></extra>",
    ))
    figura.update_yaxes(autorange="reversed", type="category")
    figura.update_xaxes(type="category")
    st.plotly_chart(estilo(figura, 120 + 50 * len(filas), nombre_columnas, nombre_filas), key="grafico_residuos")
    st.caption(
        "Un residuo mayor que 2 marca una celda con más estudiantes de los que habría si las dos variables fueran independientes, y uno menor que -2, con menos. "
        "Es una asociación, no un efecto causal."
    )


def pruebas_de_hipotesis(medias: pd.DataFrame | None, tukey: pd.DataFrame | None, interaccion: pd.DataFrame | None, residuos: pd.DataFrame | None) -> None:
    st.caption("Pruebas de hipótesis de la fase 3 sobre el periodo 2022-2: cada prueba usa a los estudiantes con sus dos variables registradas.")
    medias_por_grupo(medias)
    pares_tukey(tukey)
    interaccion_estrato(interaccion)
    residuos_independencia(residuos)


def tarjetas_clusteres(perfiles: pd.DataFrame) -> None:
    columnas = st.columns(len(perfiles))
    for columna, (_, perfil) in zip(columnas, perfiles.iterrows(), strict=True):
        columna.metric(
            f"Clúster {int(perfil['cluster'])}", f"{numero(perfil['porcentaje'], 1)} %",
            help=f"{numero(perfil['n'])} estudiantes de entrenamiento; puntaje global medio de {numero(perfil['puntaje_global'], 1)}.",
        )


def plano_clusteres(celdas: pd.DataFrame | None, cargas: pd.DataFrame | None) -> None:
    st.subheader("Plano de los dos primeros componentes principales")
    if celdas is None:
        no_publicado("El plano por celdas")
        return
    varianza = {} if cargas is None else cargas.drop_duplicates("componente").set_index("componente")["varianza"].to_dict()
    titulos = [f"Componente {j}" + (f" ({numero(varianza[j], 1)} % de la varianza)" if j in varianza else "") for j in (1, 2)]
    proporcion = celdas["proporcion"]
    rango = proporcion.max() - proporcion.min()
    alfas = 0.25 + 0.75 * ((proporcion - proporcion.min()) / rango if rango > 0 else 1.0)
    formas = [
        {
            "type": "rect", "x0": celda["x_inferior"], "x1": celda["x_superior"], "y0": celda["y_inferior"], "y1": celda["y_superior"], "layer": "below", "opacity": alfa,
            "line": {"color": "white", "width": 0.5}, "fillcolor": COLORES_CLUSTER[int(celda["cluster"]) - 1],
        }
        for (_, celda), alfa in zip(celdas.iterrows(), alfas, strict=True)
    ]
    figura = go.Figure(layout={"shapes": formas})
    for cluster, grupo in celdas.groupby("cluster"):
        # Las formas no responden al cursor: un marcador invisible en el centro de cada celda da su n, y otro sin datos, la leyenda.
        figura.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"Clúster {cluster}", marker={"symbol": "square", "size": 12, "color": COLORES_CLUSTER[cluster - 1]}))
        figura.add_trace(go.Scatter(
            x=(grupo["x_inferior"] + grupo["x_superior"]) / 2, y=(grupo["y_inferior"] + grupo["y_superior"]) / 2, mode="markers", showlegend=False,
            marker={"symbol": "square", "size": 12, "opacity": 0}, customdata=grupo[["n"]].assign(porcentaje=100 * grupo["proporcion"]),
            hovertemplate=f"Clúster {cluster}: el %{{customdata[1]:.0f}} % de %{{customdata[0]:,.0f}} estudiantes<extra></extra>",
        ))
    figura = estilo(figura, 460, titulos[0], titulos[1])
    figura.update_xaxes(range=[celdas["x_inferior"].min(), celdas["x_superior"].max()], showgrid=False, zeroline=False)
    figura.update_yaxes(range=[celdas["y_inferior"].min(), celdas["y_superior"].max()], showgrid=False, zeroline=False)
    st.plotly_chart(figura, key="grafico_plano")
    st.caption(
        "Cada celda reúne a 10 o más estudiantes y toma el color del clúster mayoritario; su intensidad es la proporción de ese clúster en la celda. "
        "Las celdas en blanco no tienen estudiantes. Pase el cursor por una celda para ver cuántos tiene."
    )
    if cargas is not None:
        tabla = cargas.loc[cargas["componente"].le(2)].pivot(index="area", columns="componente", values="carga").reindex(AREAS_PERFIL.values())
        st.dataframe(
            pd.DataFrame({"Área": tabla.index, **{f"Carga en el componente {j}": tabla[j].map(lambda valor: numero(valor, 2)) for j in tabla.columns}}),
            hide_index=True, key="tabla_cargas",
        )


def perfiles_clusteres(perfiles: pd.DataFrame) -> None:
    st.subheader("Perfil de cada clúster por área")
    escala = st.radio("Escala", ESCALAS_PERFIL, horizontal=True, key="escala_perfiles")
    prefijo, titulo, formato = ("", "Puntaje medio del área", ".1f") if escala == ESCALAS_PERFIL[0] else ("z_", "Media estandarizada (z)", ".2f")
    figura = go.Figure()
    for _, perfil in perfiles.iterrows():
        cluster = int(perfil["cluster"])
        figura.add_trace(go.Scatter(
            x=list(AREAS_PERFIL.values()), y=[perfil[prefijo + area] for area in AREAS_PERFIL], mode="lines+markers", name=f"Clúster {cluster}",
            line={"color": COLORES_CLUSTER[cluster - 1]}, hovertemplate="%{x}: %{y:" + formato + "}<extra>" + f"Clúster {cluster}" + "</extra>",
        ))
    figura.update_xaxes(type="category")
    st.plotly_chart(estilo(figura, 380, "", titulo), key="grafico_perfiles")
    st.caption("Los clústeres se numeran de menor a mayor nivel. Si las líneas son paralelas, los clústeres difieren en el nivel general y no en las áreas.")


def reparto_clusteres(cruces: dict[str, pd.DataFrame | None]) -> None:
    st.subheader("Reparto de los clústeres")
    dimension = st.radio("Por", list(CRUCES_CLUSTER), horizontal=True, key="reparto")
    cruce = cruces[dimension]
    if cruce is None:
        no_publicado(f"El cruce de clúster por {dimension.lower()}")
        return
    columna = CRUCES_CLUSTER[dimension]
    niveles = cruce[columna].unique().tolist()
    figura = go.Figure()
    for cluster, grupo in cruce.groupby("cluster"):
        grupo = grupo.set_index(columna).reindex(niveles)
        figura.add_trace(go.Bar(
            x=grupo["porcentaje"], y=[legible(n) for n in niveles], orientation="h", name=f"Clúster {cluster}", marker_color=COLORES_CLUSTER[cluster - 1],
            customdata=grupo[["n"]], hovertemplate="%{y}: %{x:.1f} % (%{customdata[0]:,.0f} estudiantes)<extra>" + f"Clúster {cluster}" + "</extra>",
        ))
    figura.update_layout(barmode="stack", legend_traceorder="normal")
    figura.update_yaxes(autorange="reversed", type="category")
    st.plotly_chart(estilo(figura, max(260, 46 * len(niveles)), "Porcentaje del grupo en cada clúster"), key="grafico_reparto")
    st.caption("Cada barra suma 100 %: es la parte de ese grupo de estudiantes que cae en cada clúster. Es una asociación, no un efecto causal.")


def segmentacion(perfiles: pd.DataFrame | None, celdas: pd.DataFrame | None, cargas: pd.DataFrame | None, cruces: dict[str, pd.DataFrame | None]) -> None:
    st.caption(
        "K-Means sobre las cinco áreas estandarizadas de los estudiantes de entrenamiento que tienen las cinco (el puntaje global queda fuera). "
        "Cada tarjeta da la parte de esos estudiantes que reúne cada clúster."
    )
    if perfiles is None:
        no_publicado("El agregado de perfiles por clúster")
    else:
        tarjetas_clusteres(perfiles)
    plano_clusteres(celdas, cargas)
    if perfiles is not None:
        perfiles_clusteres(perfiles)
    reparto_clusteres(cruces)




# Fase 4 ---------------------------------------------------------------------------------------


@st.cache_data
def leer_ficha() -> dict:
    return json.loads((MODELO / "ficha.json").read_text(encoding="utf-8"))


@st.cache_resource
def cargar_modelo(archivo: str) -> object:
    """El `Pipeline` del tablero; deserializarlo solo pide `saber11.variables`, scikit-learn y xgboost."""
    return joblib.load(MODELO / archivo)


def estimar(ficha: dict, valores: dict[str, object]) -> float:
    """Puntaje global esperado del perfil; `tablero.perfil` falla si un valor no es admitido por la ficha."""
    return float(cargar_modelo(ficha["archivo"]).predict(tablero.perfil(ficha, valores))[0])


@st.cache_data
def efecto_del_sin_dato() -> tuple[float, float]:
    """Estimación del perfil de ejemplo completo y con «Sin dato» en todo lo que lo admite."""
    ficha = leer_ficha()
    vacio = {e["columna"]: None if e["sin_dato"] else tablero.PERFIL_DE_EJEMPLO[e["columna"]] for e in ficha["entrada"]}
    return estimar(ficha, tablero.PERFIL_DE_EJEMPLO), estimar(ficha, vacio)


def celda(cubo: pd.DataFrame, estrato: str = TODOS, naturaleza: str = TODOS, zona: str = TODOS) -> pd.Series:
    return cubo.loc[cubo["estrato"].eq(estrato) & cubo["naturaleza"].eq(naturaleza) & cubo["zona"].eq(zona)].iloc[0]


def metrica(metricas: pd.DataFrame, modelo: str, columna: str) -> float:
    return float(metricas.set_index("modelo").loc[modelo, columna])


def huecos(serie: pd.DataFrame) -> list[str]:
    """Segundos semestres sin datos entre el primero y el último de la serie (el periodo 20194 es 2019-2)."""
    anios = set(serie["periodo"] // 10)
    return [f"{anio}-2" for anio in range(min(anios), max(anios) + 1) if anio not in anios]


def enumerar(elementos: list[str], conjuncion: str = "y") -> str:
    return elementos[0] if len(elementos) == 1 else ", ".join(elementos[:-1]) + f" {conjuncion} " + elementos[-1]


def valores_de_lectura(cubo: pd.DataFrame, serie: pd.DataFrame, metricas: pd.DataFrame, ficha: dict) -> dict[str, str]:
    """Las cifras que citan los textos de lectura, todas calculadas de los agregados y de la ficha."""
    def modelos_de(enfoque: str) -> str:
        return enumerar(metricas.loc[metricas["enfoque"].eq(enfoque), "etiqueta"].drop_duplicates().tolist())

    return {
        "brecha_naturaleza": numero(celda(cubo, naturaleza="NO OFICIAL")["media"] - celda(cubo, naturaleza="OFICIAL")["media"], 1),
        "brecha_zona": numero(celda(cubo, zona="URBANO")["media"] - celda(cubo, zona="RURAL")["media"], 1),
        "primer_periodo": nombre_periodo(serie["periodo"].min()),
        "r2": numero(metrica(metricas, MODELO_REGRESION, "r2"), 3),
        "mejora_rmse": numero(metrica(metricas, MODELO_REFERENCIA, "rmse") - metrica(metricas, MODELO_REGRESION, "rmse"), 1),
        "auc": numero(metrica(metricas, MODELO_CLASIFICACION, "auc"), 3),
        "rmse_tablero": numero(ficha["metricas_validacion"]["rmse"], 1),
        "margen": numero(ficha["margen"]["puntos"], 1),
        "nivel": numero(ficha["margen"]["nivel"], 1),
        "sin_ia": modelos_de("Sin IA"),
        "con_ia": modelos_de("Con IA"),
    }


def selector_audiencia() -> str:
    st.sidebar.header("Lectura")
    return st.sidebar.selectbox(
        "Leer como", list(AUDIENCIAS), key="audiencia",
        help="Cambia el texto de lectura de cada pestaña; los gráficos y las cifras son los mismos para todas las audiencias.",
    )


def lectura(vista: str, audiencia: str, valores: dict[str, str]) -> None:
    st.container(border=True).markdown(f"**Lectura para {AUDIENCIAS[audiencia]}.** " + LECTURAS[vista][audiencia].format(**valores))


def nombre_jornada(jornada: str) -> str:
    return "Todas" if jornada == TODOS else NIVELES_LEGIBLES["COLE_JORNADA"].get(jornada, capitalizar(jornada))


def filtro_jornada(jornadas: pd.DataFrame | None) -> str:
    """El selector solo existe si el agregado de jornada está publicado."""
    if jornadas is None:
        return TODOS
    opciones = (TODOS, *jornadas.loc[jornadas["jornada"].ne(TODOS), "jornada"].unique())
    return st.sidebar.selectbox("Jornada (subpestaña Jornada)", opciones, format_func=nombre_jornada, key="jornada")


def resumen(cubo: pd.DataFrame, departamento_zona: pd.DataFrame, serie: pd.DataFrame, metricas: pd.DataFrame, ficha: dict) -> None:
    nacional, oficial = celda(cubo), celda(cubo, naturaleza="OFICIAL")
    st.subheader("El examen en 2022-2")
    columnas = st.columns(4)
    columnas[0].metric("Estudiantes en 2022-2", numero(nacional["n"]))
    columnas[1].metric("Media nacional del puntaje global", numero(nacional["media"], 1), help=f"Mediana: {numero(nacional['mediana'])}.")
    columnas[2].metric("En colegios oficiales", f"{numero(100 * oficial['n'] / nacional['n'], 1)} %")
    columnas[3].metric("Departamentos", numero(departamento_zona["departamento"].nunique()))
    columnas = st.columns(4)
    columnas[0].metric(
        "Brecha nacional por naturaleza", f"{numero(celda(cubo, naturaleza='NO OFICIAL')['media'] - oficial['media'], 1)} puntos",
        help="Media nacional de los colegios no oficiales menos la de los oficiales.",
    )
    columnas[1].metric(
        "Brecha nacional por zona", f"{numero(celda(cubo, zona='URBANO')['media'] - celda(cubo, zona='RURAL')['media'], 1)} puntos",
        help="Media nacional de las sedes urbanas menos la de las rurales.",
    )
    columnas[2].metric(
        "Periodos de la serie", numero(serie["periodo"].nunique()),
        help=f"De {nombre_periodo(serie['periodo'].min())} a {nombre_periodo(serie['periodo'].max())}, sin {enumerar(huecos(serie), "ni")}.",
    )

    st.subheader("Los modelos en el conjunto de prueba (fase 3)")
    prueba = f"{numero(metrica(metricas, MODELO_REGRESION, 'n'))} estudiantes de prueba."
    rmse, referencia = metrica(metricas, MODELO_REGRESION, "rmse"), metrica(metricas, MODELO_REFERENCIA, "rmse")
    columnas = st.columns(4)
    columnas[0].metric("R² de XGBoost", numero(metrica(metricas, MODELO_REGRESION, "r2"), 3), help=f"Regresión del puntaje global; {prueba}")
    columnas[1].metric("RMSE de XGBoost", f"{numero(rmse, 1)} puntos", help=f"Regresión del puntaje global; {prueba}")
    columnas[2].metric(
        "Mejora frente a la múltiple", f"{numero(referencia - rmse, 1)} puntos",
        help=f"RMSE de la regresión múltiple, sin IA ({numero(referencia, 1)}), menos el de XGBoost ({numero(rmse, 1)}).",
    )
    columnas[3].metric(
        "AUC-ROC de XGBoost (naturaleza)", numero(metrica(metricas, MODELO_CLASIFICACION, "auc"), 3),
        help=f"Clasificación de la naturaleza del colegio (no oficial) con datos del hogar y del estudiante; {prueba}",
    )

    st.subheader("El modelo de la pestaña Predicción (validación)")
    columnas = st.columns(4)
    columnas[0].metric("RMSE en validación", f"{numero(ficha['metricas_validacion']['rmse'], 1)} puntos")
    columnas[1].metric(f"Margen del {numero(100 * ficha['margen']['nivel'])} %", f"±{numero(ficha['margen']['puntos'], 1)} puntos")
    columnas[2].metric("Estudiantes en la hoja más pequeña", numero(ficha["cobertura_minima_hoja"]))
    columnas[3].metric("Peso del modelo", f"{numero(ficha['peso_mb'], 1)} MB")
    st.caption(
        "Las métricas de prueba son las de la fase 3: cada modelo tocó la prueba una sola vez. El modelo de la pestaña Predicción es "
        "el XGBoost de la fase 3 reajustado con hojas de 10 o más estudiantes, y solo se evaluó en validación."
    )


def nota_de_huecos(serie: pd.DataFrame) -> None:
    st.caption(
        f"La serie no tiene {enumerar(huecos(serie), "ni")}, que no están en los datos abiertos del ICFES; el eje los salta y el filtro "
        "de periodos solo recorre los periodos publicados."
    )


def vista_jornada(filtros: Filtros, jornada: str, jornadas: pd.DataFrame | None) -> None:
    st.subheader("Mediana por jornada")
    if jornadas is None:
        no_publicado("El agregado por jornada")
        return
    if filtros.departamento != TODOS or filtros.zona != TODOS:
        st.info("La jornada se publica por estrato y naturaleza, así que no aplica el filtro de departamento ni el de zona.")
    celdas = jornadas.loc[jornadas["estrato"].eq(filtros.estrato) & jornadas["naturaleza"].eq(filtros.naturaleza)]
    elegida = celdas.loc[celdas["jornada"].eq(jornada)].iloc[0]
    columnas = st.columns(3)
    columnas[0].metric("Estudiantes en la jornada", numero(elegida["n"]))
    columnas[1].metric("Media en la jornada", numero(elegida["media"], 1))
    columnas[2].metric("Mediana en la jornada", numero(elegida["mediana"]), help=f"Rango intercuartílico: {numero(elegida['q1'])} a {numero(elegida['q3'])}")
    vista = celdas.loc[celdas["jornada"].ne(TODOS)].sort_values("mediana", ascending=False)
    figura = go.Figure(go.Bar(
        x=vista["jornada"].map(nombre_jornada), y=vista["mediana"],
        marker_color=[COLOR_RESALTADO if j == jornada else COLOR_PRINCIPAL for j in vista["jornada"]],
        error_y={"type": "data", "symmetric": False, "array": vista["q3"] - vista["mediana"], "arrayminus": vista["mediana"] - vista["q1"], "color": COLOR_APOYO},
        customdata=vista[["n", "q1", "q3"]], hovertemplate="%{x}: mediana %{y}, cuartiles %{customdata[1]} a %{customdata[2]} (%{customdata[0]:,.0f} estudiantes)<extra></extra>",
    ))
    figura.update_xaxes(type="category")
    st.plotly_chart(estilo(figura, 360, "", "Puntaje global (mediana y cuartiles)"), key="grafico_jornada")
    st.caption(
        f"Estrato: {filtros.estrato}. Naturaleza: {capitalizar(filtros.naturaleza)}. Las barras grises van del primer al tercer cuartil; "
        "cada celda reúne a 10 o más estudiantes. Es una asociación: la jornada va de la mano con el tipo de colegio y de hogar."
    )


def barras_con_y_sin_ia(metricas: pd.DataFrame, tarea: str, columna: str, titulo: str, decimales: int, clave: str) -> None:
    vista = metricas.loc[metricas["tarea"].eq(tarea)].sort_values(columna, ascending=columna != "rmse")
    figura = go.Figure()
    for enfoque, color in COLOR_ENFOQUE.items():
        parte = vista.loc[vista["enfoque"].eq(enfoque)]
        figura.add_trace(go.Bar(
            x=parte[columna], y=parte["etiqueta"], orientation="h", name=enfoque, marker_color=color,
            text=parte[columna].map(lambda valor: numero(valor, decimales)), textposition="outside",
            hovertemplate="%{y}: %{text}<extra>" + enfoque + "</extra>",
        ))
    figura.update_yaxes(categoryorder="array", categoryarray=vista["etiqueta"].tolist())  # el mejor arriba
    figura.update_xaxes(range=[0, 1.15 * vista[columna].max()])
    st.plotly_chart(estilo(figura, 60 + 34 * len(vista), titulo), key=clave)


def con_y_sin_ia(metricas: pd.DataFrame) -> None:
    st.subheader("Estadística clásica (sin IA) frente a aprendizaje automático (con IA)")
    columnas = st.columns(2)
    with columnas[0]:
        barras_con_y_sin_ia(metricas, "Regresión", "rmse", "RMSE en prueba (puntos; menor es mejor)", 1, "grafico_con_sin_ia_regresion")
    with columnas[1]:
        barras_con_y_sin_ia(metricas, "Clasificación", "auc", "AUC-ROC en prueba (mayor es mejor)", 3, "grafico_con_sin_ia_clasificacion")
    formato = {"r2": 3, "rmse": 2, "mae": 2, "accuracy": 3, "f1": 3, "auc": 3, "auc_pr": 3}
    nombres = {"r2": "R²", "rmse": "RMSE", "mae": "MAE", "accuracy": "Exactitud", "f1": "F1", "auc": "AUC-ROC", "auc_pr": "AUC-PR"}
    tabla = pd.DataFrame({"Tarea": metricas["tarea"], "Enfoque": metricas["enfoque"], "Modelo": metricas["etiqueta"], "Fase": metricas["fase"]})
    for columna, decimales in formato.items():
        tabla[nombres[columna]] = metricas[columna].map(lambda valor, d=decimales: "" if pd.isna(valor) else numero(valor, d))
    st.dataframe(tabla, hide_index=True, key="tabla_con_sin_ia")
    st.caption(
        f"Métricas en el conjunto de prueba ({numero(metricas['n'].iloc[0])} estudiantes), que cada modelo usó una sola vez. "
        "La clasificación predice si el colegio es no oficial con datos del hogar y del estudiante; el F1 es el de la clase no oficial."
    )


def figura_del_proyecto(ruta: str, pie: str) -> None:
    """Una figura de figuras/ con su pie; si falta, la app lo dice."""
    archivo = FIGURAS / ruta
    if not archivo.is_file():
        no_publicado(f"La figura {ruta}")
        return
    st.image(str(archivo), width=900)
    st.caption(pie)


def clasificacion(metricas: pd.DataFrame) -> None:
    st.subheader("Curvas de los clasificadores")
    figura_del_proyecto(
        "fase-3/clasificacion-curvas.png",
        "Fase 3. (a) Curvas ROC y (b) de precisión y exhaustividad de los clasificadores en validación, con la clase no oficial como "
        "positiva; el punto marca el umbral elegido. La diagonal de la ROC es el azar y la línea horizontal de la PR, la proporción de no oficiales.",
    )
    st.subheader("Métricas de prueba")
    vista = metricas.loc[metricas["tarea"].eq("Clasificación")]
    st.dataframe(
        pd.DataFrame({
            "Modelo": vista["etiqueta"], "Enfoque": vista["enfoque"],
            **{nombre: vista[columna].map(lambda valor: "" if pd.isna(valor) else numero(valor, 3)) for columna, nombre in (
                ("accuracy", "Exactitud"), ("precision", "Precisión"), ("recall", "Exhaustividad"), ("f1", "F1"), ("auc", "AUC-ROC"), ("auc_pr", "AUC-PR"),
            )},
        }),
        hide_index=True, key="tabla_clasificacion",
    )
    st.caption("La exactitud engaña con clases desbalanceadas: la clase mayoritaria acierta mucho sin encontrar un solo colegio no oficial.")


def interpretabilidad() -> None:
    st.subheader("Qué variables pesan en la estimación")
    figura_del_proyecto(
        "fase-3/importancia-variables.png",
        "Fase 3. Importancia de las variables del XGBoost de regresión, por variable original: (a) contribución SHAP absoluta media; "
        "(b) aumento medio del RMSE al permutar cada variable. Muestra estratificada de validación.",
    )
    figura_del_proyecto(
        "fase-3/shap-por-categoria.png",
        "Fase 3. Contribución SHAP media de cada nivel, en puntos frente al valor base, para las cinco variables más importantes; "
        "solo niveles con 10 o más estudiantes de validación.",
    )
    figura_del_proyecto(
        "fase-4/eta-shap.png",
        "Fase 4. Tamaño de efecto de las pruebas de la fase 3 (η² o η² parcial) frente al SHAP medio de las mismas variables en el "
        "modelo de la pestaña Predicción: por sí sola una variable puede explicar mucho y aportar poco cuando el modelo ya tiene las demás.",
    )


def nivel_legible(entrada: dict, nivel: object) -> str:
    if nivel == SIN_DATO:
        return SIN_DATO
    respuestas = {r["valor"]: r["respuesta"] for r in entrada.get("respuestas", [])}
    if nivel in respuestas:
        return respuestas[nivel]
    if entrada["tipo"] == "numerica":
        return numero(nivel)
    if entrada["columna"] == "COLE_DEPTO_UBICACION":
        return str(nivel).title()
    return NIVELES_LEGIBLES.get(entrada["columna"], {}).get(nivel, legible(str(nivel)))


def campo(entrada: dict) -> object:
    """El control de una entrada de la ficha: lista cerrada, salvo la edad, que se escribe y se valida al estimar."""
    columna, defecto, clave = entrada["columna"], tablero.PERFIL_DE_EJEMPLO[entrada["columna"]], f"prediccion_{entrada['columna']}"
    if entrada["tipo"] == "numerica" and "respuestas" not in entrada:
        valor = st.number_input(entrada["etiqueta"], min_value=int(entrada["minimo"]), max_value=int(entrada["maximo"]), value=int(defecto), step=1, key=clave)
        sin_dato = entrada["sin_dato"] and st.checkbox(f"{entrada['etiqueta'].split(' (')[0]}: sin dato", key=f"{clave}_sin_dato")
        return None if sin_dato else float(valor)
    opciones = [*(entrada["valores"] if entrada["tipo"] == "numerica" else entrada["niveles"]), *([SIN_DATO] if entrada["sin_dato"] else [])]
    valor = st.selectbox(entrada["etiqueta"], opciones, index=opciones.index(defecto), format_func=lambda n: nivel_legible(entrada, n), key=clave)
    return None if valor == SIN_DATO else valor


def formulario(ficha: dict) -> dict[str, object]:
    with st.form("formulario_prediccion"):
        columnas = st.columns(len(GRUPOS_FORMULARIO))
        valores = {}
        for columna, (grupo, titulo) in zip(columnas, GRUPOS_FORMULARIO.items(), strict=True):
            with columna:
                st.markdown(f"**{titulo}**")
                for entrada in (e for e in ficha["entrada"] if e["grupo"] == grupo):
                    valores[entrada["columna"]] = campo(entrada)
        st.form_submit_button("Estimar", key="estimar")
    return valores


def grafico_estimacion(estimacion: float, inferior: float, superior: float, mediana: float) -> None:
    figura = go.Figure()
    figura.add_trace(go.Scatter(
        x=[inferior, superior], y=["Perfil", "Perfil"], mode="lines", line={"color": COLOR_APOYO, "width": 14}, name="Margen", hoverinfo="skip",
    ))
    figura.add_trace(go.Scatter(
        x=[estimacion], y=["Perfil"], mode="markers", marker={"color": COLOR_RESALTADO, "size": 14}, name="Estimación",
        hovertemplate="Estimación: %{x:.1f}<extra></extra>",
    ))
    figura.add_vline(x=mediana, line={"color": COLOR_PRINCIPAL, "dash": "dash", "width": 2}, annotation_text="Mediana nacional", annotation_position="top")
    figura.update_xaxes(range=[0, 500])
    st.plotly_chart(estilo(figura, 200, "Puntaje global"), key="grafico_prediccion")


def prediccion(ficha: dict, mediana_nacional: float) -> None:
    st.container(border=True).markdown(f"**Aviso.** {AVISO_PREDICCION}")
    st.caption(
        "Elija el perfil y pulse Estimar. Las opciones son las que el modelo vio en 10 o más estudiantes de entrenamiento; "
        "«Sin dato» aparece solo donde el modelo sabe imputar un faltante."
    )
    valores = formulario(ficha)
    try:
        estimacion = estimar(ficha, valores)
    except ValueError as error:
        st.error(f"{error}. El formulario solo admite lo que el modelo vio en 10 o más estudiantes de entrenamiento: corrija la entrada y pulse Estimar.")
        return
    margen = ficha["margen"]["puntos"]
    inferior, superior = max(0.0, estimacion - margen), min(500.0, estimacion + margen)
    columnas = st.columns(3)
    columnas[0].metric("Puntaje global esperado del perfil", numero(estimacion, 1))
    columnas[1].metric(f"Margen del {numero(100 * ficha['margen']['nivel'])} %", f"±{numero(margen, 1)} puntos")
    columnas[2].metric("Rango con el margen", f"{numero(inferior)} a {numero(superior)}")
    grafico_estimacion(estimacion, inferior, superior, mediana_nacional)
    completo, vacio = efecto_del_sin_dato()
    st.caption(
        f"{ficha['margen']['lectura']} Esa cobertura es aproximada: la validación también sirvió para elegir el modelo. "
        f"«Sin dato» no es neutro: el modelo imputa el faltante y la estimación cambia; con el perfil de ejemplo, dejar «Sin dato» "
        f"en todo lo que lo admite la lleva de {numero(completo, 1)} a {numero(vacio, 1)} puntos."
    )


def metodo_y_fuente(ficha: dict, serie: pd.DataFrame) -> None:
    st.subheader("Fuente")
    st.markdown(
        "ICFES, Resultados únicos Saber 11, publicados en datos.gov.co con licencia "
        "[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Todas las vistas usan el periodo 2022-2, salvo Serie, "
        f"que va de {nombre_periodo(serie['periodo'].min())} a {nombre_periodo(serie['periodo'].max())} sin {enumerar(huecos(serie), "ni")}. "
        "Los agregados conservan la licencia, con atribución al ICFES."
    )
    st.subheader("Privacidad")
    st.markdown(
        "En atención a la Ley 1581 de 2012 de protección de datos personales, la app solo lee agregados con 10 o más estudiantes "
        "por celda; si un cruce no cumple, no se publica y la app lo dice. El plano del PCA va por celdas, nunca con un punto por "
        "estudiante. El modelo de la pestaña Predicción no guarda observaciones: cada hoja de sus árboles cubre "
        f"{numero(ficha['minimo_por_hoja'])} o más estudiantes, y el formulario solo ofrece niveles vistos en 10 o más."
    )
    st.subheader("Método")
    st.markdown(
        "Cada filtro responde con el cruce más fino publicado que lo contiene. Las pruebas de hipótesis, la segmentación y los "
        "modelos son los de la fase 3; sus métricas de prueba se calcularon una sola vez por modelo. La pestaña Predicción usa el "
        f"XGBoost de la fase 3 reajustado en la CPU con hojas de {numero(ficha['minimo_por_hoja'])} o más estudiantes; su margen es "
        "el cuantil de los residuos absolutos de validación con corrección de muestra finita (predicción conformal, Lei et al., 2018), "
        "de cobertura aproximada porque la validación también eligió el modelo. Todo es asociación, no efecto causal."
    )
    st.subheader("Reproducción")
    st.markdown(
        f"Código, datos, cuadernos y pruebas: [{URL_REPOSITORIO.removeprefix('https://')}]({URL_REPOSITORIO}). "
        "El selector «Leer como» cambia el texto de lectura de cada pestaña para cuatro audiencias; los gráficos son los mismos."
    )
    st.dataframe(
        pd.DataFrame({"Paquete": ["python", *ficha["versiones"]], "Versión": [ficha["python"], *ficha["versiones"].values()]}),
        hide_index=True, key="tabla_versiones",
    )
    st.caption("Versiones con que se entrenó el modelo de la pestaña Predicción; dashboard/requirements.txt fija las mismas.")


def main() -> None:
    st.set_page_config(page_title=TITULO, layout="wide")
    st.title(TITULO)
    st.caption("Proyecto integrador de Ciencia de Datos, fases 2 y 3. Periodo 2022-2 salvo en la serie, que va de 2014-2 a 2022-2.")
    st.caption("La fase 4 suma el resumen, la jornada, los modelos de la fase 3, la predicción, el método y una lectura por audiencia.")
    cubo = leer("estrato-naturaleza-zona-20224.csv")
    departamento_zona = leer("departamento-zona-20224.csv")
    serie = leer("serie-estrato-naturaleza.csv")
    jornadas = leer_publicado(JORNADA)
    metricas = leer("metricas-modelos.csv")
    ficha = leer_ficha()
    audiencia = selector_audiencia()
    filtros = filtros_laterales(serie, sorted(departamento_zona["departamento"].unique().tolist()))
    jornada = filtro_jornada(jornadas)
    if filtros.publicado:
        tarjetas(filtros, cubo, departamento_zona)
    else:
        st.warning(NO_PUBLICADO)
    valores = valores_de_lectura(cubo, serie, metricas, ficha)
    pestanas = dict(zip(PESTANAS, st.tabs(list(PESTANAS)), strict=True))
    with pestanas["Resumen"]:
        lectura("resumen", audiencia, valores)
        resumen(cubo, departamento_zona, serie, metricas, ficha)
    with pestanas["Exploración"]:
        lectura("exploracion", audiencia, valores)
        panorama_, serie_, territorio_, jornada_ = st.tabs(list(SUBPESTANAS_EXPLORACION))
        with panorama_:
            panorama(filtros, cubo, leer("distribucion-puntaje-20224.csv"))
        with serie_:
            serie_temporal(filtros, serie)
            nota_de_huecos(serie)
        with territorio_:
            territorio(filtros, departamento_zona)
        with jornada_:
            vista_jornada(filtros, jornada, jornadas)
    with pestanas["Pruebas de hipótesis"]:
        lectura("hipotesis", audiencia, valores)
        pruebas_de_hipotesis(
            leer_publicado("hipotesis-medias-20224.csv"), leer_publicado("hipotesis-tukey-20224.csv"),
            leer_publicado("hipotesis-interaccion-20224.csv"), leer_publicado("hipotesis-residuos-20224.csv"),
        )
    with pestanas["Modelos"]:
        st.caption(AVISO_PERIODO)
        lectura("modelos", audiencia, valores)
        regresion_, con_sin_ia_, clasificacion_, interpretabilidad_ = st.tabs(list(SUBPESTANAS_MODELOS))
        with regresion_:
            regresion(filtros, leer("regresion-metricas.csv"), leer("regresion-coeficientes.csv"), leer("regresion-grupos.csv"))
        with con_sin_ia_:
            con_y_sin_ia(metricas)
        with clasificacion_:
            clasificacion(metricas)
        with interpretabilidad_:
            interpretabilidad()
    with pestanas["Predicción"]:
        st.caption(AVISO_PERIODO)
        lectura("prediccion", audiencia, valores)
        prediccion(ficha, celda(cubo)["mediana"])
    with pestanas["Segmentación"]:
        st.caption(AVISO_PERIODO)
        lectura("segmentacion", audiencia, valores)
        segmentacion(
            leer_publicado("cluster-perfiles-20224.csv"), leer_publicado("pca-celdas-20224.csv"), leer_publicado("pca-cargas-20224.csv"),
            {nombre: leer_publicado(f"cluster-{columna}-20224.csv") for nombre, columna in CRUCES_CLUSTER.items()},
        )
    with pestanas["Método y fuente"]:
        metodo_y_fuente(ficha, serie)
    st.caption(FUENTE)


main()
