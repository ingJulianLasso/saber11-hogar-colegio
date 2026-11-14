"""Dashboard del Saber 11: panorama, serie, territorio y regresión (parte 1, fase 2), y pruebas
de hipótesis y segmentación (parte 2, fase 3).

Lee solo los agregados de data/agregados/, donde cada archivo es un cruce completo con 10 o
más estudiantes por celda. Cada combinación de filtros responde con el cruce más fino que la
contiene; si ninguno la contiene, o si un cruce no se publicó, la app lo dice en lugar de
estimarlo.

Uso local: uv run streamlit run dashboard/app.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

AGREGADOS = Path(__file__).resolve().parent.parent / "data" / "agregados"
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


def main() -> None:
    st.set_page_config(page_title=TITULO, layout="wide")
    st.title(TITULO)
    st.caption("Proyecto integrador de Ciencia de Datos, fases 2 y 3. Periodo 2022-2 salvo en la serie, que va de 2014-2 a 2022-2.")
    cubo = leer("estrato-naturaleza-zona-20224.csv")
    departamento_zona = leer("departamento-zona-20224.csv")
    serie = leer("serie-estrato-naturaleza.csv")
    filtros = filtros_laterales(serie, sorted(departamento_zona["departamento"].unique().tolist()))
    if filtros.publicado:
        tarjetas(filtros, cubo, departamento_zona)
    else:
        st.warning(NO_PUBLICADO)
    pestanas = st.tabs(["Panorama", "Serie", "Territorio", "Regresión", "Pruebas de hipótesis", "Segmentación"])
    with pestanas[0]:
        panorama(filtros, cubo, leer("distribucion-puntaje-20224.csv"))
    with pestanas[1]:
        serie_temporal(filtros, serie)
    with pestanas[2]:
        territorio(filtros, departamento_zona)
    with pestanas[3]:
        regresion(filtros, leer("regresion-metricas.csv"), leer("regresion-coeficientes.csv"), leer("regresion-grupos.csv"))
    with pestanas[4]:
        pruebas_de_hipotesis(
            leer_publicado("hipotesis-medias-20224.csv"), leer_publicado("hipotesis-tukey-20224.csv"),
            leer_publicado("hipotesis-interaccion-20224.csv"), leer_publicado("hipotesis-residuos-20224.csv"),
        )
    with pestanas[5]:
        segmentacion(
            leer_publicado("cluster-perfiles-20224.csv"), leer_publicado("pca-celdas-20224.csv"), leer_publicado("pca-cargas-20224.csv"),
            {nombre: leer_publicado(f"cluster-{columna}-20224.csv") for nombre, columna in CRUCES_CLUSTER.items()},
        )
    st.caption(FUENTE)


main()
