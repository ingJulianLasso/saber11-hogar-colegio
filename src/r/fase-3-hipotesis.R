# Fase 3 en R: replica las pruebas de hipótesis H1 a H4 de notebooks/fase-3-modelado.ipynb
# con t.test, oneway.test, aov y TukeyHSD, kruskal.test, car::Anova (tipo II, clásica y con
# HC3) y chisq.test; calcula los tamaños de efecto con las mismas fórmulas que
# saber11.hipotesis; dibuja en ggplot2 las medias con su intervalo (H1 y H2) y la interacción
# de estrato y naturaleza (H3), mide esta última con la especificación de la comparación de
# herramientas del cuaderno, y escribe resultados/r/fase-3-hipotesis.json, que el cuaderno
# compara con sus valores (sección 8).
#
# Uso: Rscript src/r/fase-3-hipotesis.R, o Source en RStudio con el archivo abierto.
# Requisitos: los paquetes de src/r/instalar.R y los intermedios que escribe
# `uv run python -m saber11.intermedios --fase 3`: data/intermedio/fase-3-hipotesis.parquet
# (las variables de H1 a H4 con los valores observados) y data/intermedio/fase-3-hipotesis.json
# (el marco de cada hipótesis: variables, niveles, orden y etiquetas).
#
# Definiciones, las mismas del cuaderno: cada prueba usa los casos completos de sus variables;
# la d de Cohen divide la diferencia de medias por la desviación combinada con n - 1; η² es la
# suma de cuadrados entre grupos sobre la total; ε² = H / (n - 1); el η² parcial de un término
# es su suma de cuadrados tipo II sobre ella más la residual; la V de Cramér sale de la χ² sin
# corrección de Yates, sin corrección de sesgo. Los intervalos bootstrap de los efectos son solo
# de Python.

suppressPackageStartupMessages({
  library(car)
  library(ggplot2)
  library(jsonlite)
  library(nanoparquet)
})

# La raíz del repositorio se busca en tres lugares: el --file= de Rscript, el archivo
# que está ejecutando source() (lo que hace Source en RStudio) y la carpeta de trabajo,
# si es la raíz.
ruta_del_script <- function() {
  argumento <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(argumento) == 1) {
    return(sub("^--file=", "", argumento))
  }
  for (marco in rev(sys.frames())) {
    if (!is.null(marco$ofile)) {
      return(marco$ofile)
    }
  }
  NULL
}

resolver_raiz <- function() {
  script <- ruta_del_script()
  if (!is.null(script)) {
    return(normalizePath(file.path(dirname(script), "..", "..")))
  }
  if (file.exists("r-versiones.txt")) {
    return(normalizePath("."))
  }
  stop("No se encontró la raíz del repositorio: ejecute desde la raíz o con Rscript src/r/fase-3-hipotesis.R")
}

raiz <- resolver_raiz()
carpeta_figuras <- file.path(raiz, "figuras", "fase-3", "r")
carpeta_resultados <- file.path(raiz, "resultados", "r")
dir.create(carpeta_figuras, recursive = TRUE, showWarnings = FALSE)
dir.create(carpeta_resultados, recursive = TRUE, showWarnings = FALSE)

REPETICIONES <- 3
PUNTOS_POR_PULGADA <- 300
CONFIANZA <- 0.95
TAMANO_MEDIAS <- c(7.0, 3.0)
TAMANO_INTERACCION <- c(2.0, 1.75)  # panel de la comparación de herramientas; el mismo del cuaderno
COLOR_PRINCIPAL <- "#2C4A6E"
COLOR_APOYO <- "#7F7F7F"
COLOR_BANDA <- "#D9D9D9"
COLOR_OFICIAL <- "#0072B2"
COLOR_NO_OFICIAL <- "#D55E00"
decimal <- scales::label_number(accuracy = 1, decimal.mark = ",", big.mark = ".")
tema <- theme_bw(base_size = 8, base_family = "serif") +
  theme(panel.grid.minor = element_blank(), legend.position = "top", legend.title = element_blank(), axis.text = element_text(size = 7.5), legend.text = element_text(size = 7.5), legend.key.size = unit(8, "pt"))

marco <- fromJSON(file.path(raiz, "data", "intermedio", "fase-3-hipotesis.json"), simplifyVector = FALSE)
datos <- as.data.frame(read_parquet(file.path(raiz, "data", "intermedio", "fase-3-hipotesis.parquet")))
y <- as.numeric(datos[[marco$objetivo]])
SIN_DATO <- marco$sin_dato

# ---- Funciones comunes --------------------------------------------------------

d_cohen <- function(a, b) {
  combinada <- sqrt(((length(a) - 1) * var(a) + (length(b) - 1) * var(b)) / (length(a) + length(b) - 2))
  (mean(a) - mean(b)) / combinada
}

eta_cuadrado <- function(valores, grupos) {
  medias <- tapply(valores, grupos, mean)
  conteos <- tapply(valores, grupos, length)
  sum(conteos * (medias - mean(valores))^2) / sum((valores - mean(valores))^2)
}

v_cramer <- function(prueba) {
  sqrt(unname(prueba$statistic) / (sum(prueba$observed) * (min(dim(prueba$observed)) - 1)))
}

# Media, desviación y n por grupo, con el intervalo t de la media, en el orden de `niveles`.
medias_con_ic <- function(valores, grupos, niveles) {
  filas <- lapply(niveles, function(nivel) {
    x <- valores[!is.na(grupos) & grupos == nivel]
    margen <- qt(0.5 + CONFIANZA / 2, length(x) - 1) * sd(x) / sqrt(length(x))
    data.frame(grupo = nivel, n = length(x), media = mean(x), desviacion = sd(x), inferior = mean(x) - margen, superior = mean(x) + margen)
  })
  do.call(rbind, filas)
}

con_sin_dato <- function(x) {
  x <- as.character(x)
  x[is.na(x)] <- SIN_DATO
  x
}

filas_json <- function(tabla) {
  lapply(seq_len(nrow(tabla)), function(i) as.list(tabla[i, , drop = FALSE]))
}

# ---- H1: internet en el hogar ---------------------------------------------------

h1 <- marco$h1
internet <- datos[[h1$variable]]
con_internet <- y[!is.na(internet) & internet == h1$grupos[[1]]]
sin_internet <- y[!is.na(internet) & internet == h1$grupos[[2]]]
prueba_h1 <- t.test(con_internet, sin_internet, var.equal = FALSE, conf.level = CONFIANZA)
resultado_h1 <- list(
  t = unname(prueba_h1$statistic), gl = unname(prueba_h1$parameter), p = prueba_h1$p.value,
  diferencia = mean(con_internet) - mean(sin_internet), inferior = prueba_h1$conf.int[1], superior = prueba_h1$conf.int[2],
  d = d_cohen(con_internet, sin_internet), n_a = length(con_internet), n_b = length(sin_internet)
)
medias_h1 <- medias_con_ic(y, con_sin_dato(internet), c(unlist(h1$grupos), SIN_DATO))

# ---- H2: educación de la madre --------------------------------------------------

h2 <- marco$h2
educacion <- datos[[h2$variable]]
en_escala <- !is.na(educacion) & educacion %in% unlist(h2$niveles)
y_h2 <- y[en_escala]
g_h2 <- factor(educacion[en_escala], levels = unlist(h2$niveles))
welch_h2 <- oneway.test(y_h2 ~ g_h2, var.equal = FALSE)
kruskal_h2 <- kruskal.test(y_h2, g_h2)
tukey_h2 <- TukeyHSD(aov(y_h2 ~ g_h2), conf.level = CONFIANZA)$g_h2
pares <- t(combn(unlist(h2$niveles), 2))
tukey_pares <- data.frame(
  grupo_a = pares[, 1], grupo_b = pares[, 2],
  diferencia = tukey_h2[paste(pares[, 2], pares[, 1], sep = "-"), "diff"],
  inferior = tukey_h2[paste(pares[, 2], pares[, 1], sep = "-"), "lwr"],
  superior = tukey_h2[paste(pares[, 2], pares[, 1], sep = "-"), "upr"],
  p = tukey_h2[paste(pares[, 2], pares[, 1], sep = "-"), "p adj"],
  row.names = NULL
)
resultado_h2 <- list(
  F = unname(welch_h2$statistic), gl1 = unname(welch_h2$parameter[1]), gl2 = unname(welch_h2$parameter[2]), p = welch_h2$p.value,
  eta2 = eta_cuadrado(y_h2, g_h2),
  H = unname(kruskal_h2$statistic), gl_kruskal = unname(kruskal_h2$parameter), p_kruskal = kruskal_h2$p.value,
  epsilon2 = unname(kruskal_h2$statistic) / (length(y_h2) - 1),
  n = length(y_h2), tukey = filas_json(tukey_pares)
)
medias_h2 <- medias_con_ic(y, con_sin_dato(educacion), c(unlist(h2$niveles), unlist(h2$descriptivos), SIN_DATO))

# ---- H3: estrato por naturaleza -------------------------------------------------

h3 <- marco$h3
completos_h3 <- !is.na(datos[[h3$a]]) & !is.na(datos[[h3$b]])
datos_h3 <- data.frame(
  y = y[completos_h3],
  estrato = factor(datos[[h3$a]][completos_h3], levels = unlist(h3$niveles_a)),
  naturaleza = factor(datos[[h3$b]][completos_h3], levels = unlist(h3$niveles_b))
)
modelo_h3 <- lm(y ~ estrato * naturaleza, data = datos_h3)
clasica <- Anova(modelo_h3, type = 2)
robusta <- suppressMessages(Anova(modelo_h3, type = 2, white.adjust = "hc3"))
terminos <- c("estrato", "naturaleza", "estrato:naturaleza")
suma_residual <- clasica["Residuals", "Sum Sq"]
anova_h3 <- setNames(lapply(terminos, function(termino) {
  list(
    suma_cuadrados = clasica[termino, "Sum Sq"], gl = clasica[termino, "Df"], F = clasica[termino, "F value"], p = clasica[termino, "Pr(>F)"],
    F_hc3 = robusta[termino, "F"], p_hc3 = robusta[termino, "Pr(>F)"],
    eta_parcial = clasica[termino, "Sum Sq"] / (clasica[termino, "Sum Sq"] + suma_residual)
  )
}), terminos)
nivel <- h3$niveles_b[[2]]
referencia <- h3$niveles_b[[1]]
efectos_simples <- do.call(rbind, lapply(unlist(h3$niveles_a), function(e) {
  dentro <- datos_h3[datos_h3$estrato == e, ]
  a <- dentro$y[dentro$naturaleza == nivel]
  b <- dentro$y[dentro$naturaleza == referencia]
  prueba <- t.test(a, b, var.equal = FALSE, conf.level = CONFIANZA)
  data.frame(
    estrato = e, diferencia = mean(a) - mean(b), inferior = prueba$conf.int[1], superior = prueba$conf.int[2],
    t = unname(prueba$statistic), gl = unname(prueba$parameter), p = prueba$p.value, d = d_cohen(a, b)
  )
}))
medias_h3 <- do.call(rbind, lapply(unlist(h3$niveles_b), function(n) {
  dentro <- datos_h3[datos_h3$naturaleza == n, ]
  cbind(naturaleza = n, medias_con_ic(dentro$y, as.character(dentro$estrato), unlist(h3$niveles_a)))
}))
names(medias_h3)[names(medias_h3) == "grupo"] <- "estrato"
resultado_h3 <- list(anova = anova_h3, gl_residual = clasica["Residuals", "Df"], n = nrow(datos_h3), efectos_simples = filas_json(efectos_simples))

# ---- H4: estrato y naturaleza; internet y zona ----------------------------------

prueba_h4 <- function(definicion) {
  completos <- !is.na(datos[[definicion$filas]]) & !is.na(datos[[definicion$columnas]])
  tabla <- table(
    factor(datos[[definicion$filas]][completos], levels = unlist(definicion$niveles_filas)),
    factor(datos[[definicion$columnas]][completos], levels = unlist(definicion$niveles_columnas))
  )
  prueba <- chisq.test(tabla, correct = FALSE)
  residuos <- prueba$stdres
  list(
    prueba = prueba,
    resultado = list(
      chi2 = unname(prueba$statistic), gl = unname(prueba$parameter), p = prueba$p.value, v = v_cramer(prueba), n = sum(tabla),
      residuos = setNames(lapply(rownames(residuos), function(f) as.list(residuos[f, ])), rownames(residuos))
    )
  )
}
pruebas_h4 <- lapply(marco$h4, prueba_h4)

# ---- Gráficos en ggplot2 --------------------------------------------------------

panel_medias <- function(medias, etiquetas, probados, panel) {
  data.frame(
    medias,
    clave = paste(panel, medias$grupo),
    etiqueta = vapply(medias$grupo, function(g) if (g == SIN_DATO) SIN_DATO else etiquetas[[g]], character(1)),
    probado = medias$grupo %in% probados,
    panel = panel
  )
}

tabla_medias <- rbind(
  panel_medias(medias_h1, h1$etiquetas, unlist(h1$grupos), "(a) Internet en el hogar (H1)"),
  panel_medias(medias_h2, h2$etiquetas, unlist(h2$niveles), "(b) Educación de la madre (H2)")
)
tabla_medias$clave <- factor(tabla_medias$clave, levels = rev(tabla_medias$clave))

medias_ggplot2 <- function(tabla) {
  ggplot(tabla, aes(media, clave)) +
    geom_segment(aes(x = media - desviacion, xend = media + desviacion, yend = clave), colour = COLOR_BANDA, linewidth = 2.2) +
    geom_errorbar(aes(xmin = inferior, xmax = superior, colour = probado), orientation = "y", width = 0, linewidth = 0.5) +
    geom_point(aes(colour = probado, fill = probado), shape = 21, size = 1.8) +
    scale_colour_manual(values = c(`TRUE` = COLOR_PRINCIPAL, `FALSE` = COLOR_APOYO), guide = "none") +
    scale_fill_manual(values = c(`TRUE` = COLOR_PRINCIPAL, `FALSE` = "white"), guide = "none") +
    scale_y_discrete(labels = setNames(tabla$etiqueta, tabla$clave)) +
    scale_x_continuous(labels = decimal) +
    facet_wrap(~panel, scales = "free") +
    labs(x = "Puntaje global (media e IC del 95 %)", y = NULL) +
    tema +
    theme(strip.background = element_blank(), strip.text = element_text(size = 9))
}

tabla_interaccion <- medias_h3
tabla_interaccion$estrato <- factor(tabla_interaccion$estrato, levels = unlist(h3$niveles_a), labels = sub(" ", "\n", unlist(h3$etiquetas_a[unlist(h3$niveles_a)])))
tabla_interaccion$naturaleza <- factor(tabla_interaccion$naturaleza, levels = unlist(h3$niveles_b), labels = unlist(h3$etiquetas_b[unlist(h3$niveles_b)]))

interaccion_ggplot2 <- function(tabla) {
  ggplot(tabla, aes(estrato, media, colour = naturaleza, shape = naturaleza, group = naturaleza)) +
    geom_line(linewidth = 0.5) +
    geom_errorbar(aes(ymin = inferior, ymax = superior), width = 0.15, linewidth = 0.4) +
    geom_point(size = 1.6) +
    scale_colour_manual(values = setNames(c(COLOR_OFICIAL, COLOR_NO_OFICIAL), levels(tabla$naturaleza))) +
    scale_y_continuous(labels = decimal) +
    labs(x = "Estrato", y = "Puntaje global medio") +
    tema
}

# Componentes de una expresión de ggplot2 encadenada con `+`: la función tiene una sola
# instrucción, así que esto mide su tamaño.
componentes <- function(expresion) {
  if (is.call(expresion) && identical(expresion[[1]], as.name("+"))) {
    return(componentes(expresion[[2]]) + componentes(expresion[[3]]))
  }
  1
}

# Mide lo mismo que la comparación de herramientas del cuaderno: instrucciones del cuerpo
# de la función, mediana de tres tiempos de construir y exportar a PNG, y peso del PNG y del PDF.
medir <- function(nombre, construir, tabla, tamano) {
  ruta_png <- file.path(carpeta_figuras, sprintf("%s-ggplot2.png", nombre))
  tiempos <- vapply(seq_len(REPETICIONES), function(i) {
    inicio <- Sys.time()
    ggsave(ruta_png, construir(tabla), width = tamano[1], height = tamano[2], dpi = PUNTOS_POR_PULGADA)
    as.numeric(difftime(Sys.time(), inicio, units = "secs"))
  }, numeric(1))
  ruta_pdf <- tempfile(fileext = ".pdf")
  ggsave(ruta_pdf, construir(tabla), width = tamano[1], height = tamano[2])
  medida <- list(
    instrucciones = length(body(construir)) - 1,
    componentes = componentes(body(construir)[[length(body(construir))]]),
    tiempo_s = median(tiempos),
    png_kb = file.size(ruta_png) / 1024,
    pdf_kb = file.size(ruta_pdf) / 1024
  )
  unlink(ruta_pdf)
  medida
}

ggsave(file.path(carpeta_figuras, "medias-ic-ggplot2.png"), medias_ggplot2(tabla_medias), width = TAMANO_MEDIAS[1], height = TAMANO_MEDIAS[2], dpi = PUNTOS_POR_PULGADA)
medicion <- list(interaccion = medir("interaccion", interaccion_ggplot2, tabla_interaccion, TAMANO_INTERACCION))

# ---- Resultados para la paridad -------------------------------------------------

resultados <- list(
  version_r = paste(R.version$major, R.version$minor, sep = "."),
  version_ggplot2 = as.character(packageVersion("ggplot2")),
  version_car = as.character(packageVersion("car")),
  h1 = resultado_h1,
  h2 = resultado_h2,
  h3 = resultado_h3,
  h4 = lapply(pruebas_h4, `[[`, "resultado"),
  medias = list(h1 = filas_json(medias_h1), h2 = filas_json(medias_h2), h3 = filas_json(medias_h3)),
  medicion = medicion
)
ruta_json <- file.path(carpeta_resultados, "fase-3-hipotesis.json")
write_json(resultados, ruta_json, auto_unbox = TRUE, digits = NA, pretty = TRUE)

cat(sprintf("%s, car %s y ggplot2 %s\n", R.version.string, resultados$version_car, resultados$version_ggplot2))
cat("\nH1: puntaje global con internet en el hogar frente a sin internet\n")
print(prueba_h1)
cat(sprintf("d de Cohen = %s\n", formatC(resultado_h1$d, format = "f", digits = 3, decimal.mark = ",")))
cat("\nH2: educación de la madre\n")
print(welch_h2)
cat(sprintf(
  "η² = %s; Kruskal-Wallis H = %s, ε² = %s; Tukey HSD: %d pares, %d con p < 0,05\n",
  formatC(resultado_h2$eta2, format = "f", digits = 4, decimal.mark = ","),
  formatC(resultado_h2$H, format = "f", digits = 1, decimal.mark = ",", big.mark = "."),
  formatC(resultado_h2$epsilon2, format = "f", digits = 4, decimal.mark = ","),
  nrow(tukey_pares), sum(tukey_pares$p < 0.05)
))
cat("\nH3: ANOVA de dos vías, tipo II, clásica y con HC3\n")
print(clasica)
print(robusta)
cat("\nH4: χ² de independencia sin corrección de Yates\n")
for (nombre in names(pruebas_h4)) {
  cat(sprintf("%s: χ² = %s con %d gl, V de Cramér = %s\n", nombre,
    formatC(pruebas_h4[[nombre]]$resultado$chi2, format = "f", digits = 1, decimal.mark = ",", big.mark = "."),
    as.integer(pruebas_h4[[nombre]]$resultado$gl),
    formatC(pruebas_h4[[nombre]]$resultado$v, format = "f", digits = 4, decimal.mark = ",")))
}
cat(sprintf("\nEscrito %s\n", sub(paste0(raiz, "/"), "", ruta_json, fixed = TRUE)))
