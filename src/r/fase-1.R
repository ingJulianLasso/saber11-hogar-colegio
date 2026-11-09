# Fase 1 en R: replica los descriptivos, las frecuencias, las correlaciones y las
# pruebas χ² de H4 que calcula notebooks/fase-1-eda.ipynb sobre el recorte del
# 2022-2, dibuja en ggplot2 el histograma del puntaje global, el mapa de
# correlación y la caja por estrato, mide los dos primeros con la misma
# especificación de la sección 8 del cuaderno y escribe resultados/r/fase-1.json,
# que el cuaderno compara con sus propios valores (sección 9).
#
# Uso: Rscript src/r/fase-1.R
# Requisitos: los paquetes de src/r/instalar.R y el recorte data/saber11-20224.parquet
# (python -m saber11.datos --materializar).
#
# Definiciones, las mismas del cuaderno: varianza muestral; cuartiles de tipo 7;
# asimetría g1 sesgada (moments::skewness); curtosis en exceso (moments::kurtosis,
# que es la de Pearson, menos 3); χ² sin corrección de Yates; V de Cramér sin
# corrección de sesgo.

suppressPackageStartupMessages({
  library(ggplot2)
  library(jsonlite)
  library(moments)
  library(nanoparquet)
})

argumento <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
raiz <- normalizePath(file.path(dirname(sub("^--file=", "", argumento)), "..", ".."))
carpeta_figuras <- file.path(raiz, "figuras", "fase-1", "r")
carpeta_resultados <- file.path(raiz, "resultados", "r")
dir.create(carpeta_figuras, recursive = TRUE, showWarnings = FALSE)
dir.create(carpeta_resultados, recursive = TRUE, showWarnings = FALSE)

PUNTAJES <- c(
  "PUNT_GLOBAL", "PUNT_LECTURA_CRITICA", "PUNT_MATEMATICAS",
  "PUNT_C_NATURALES", "PUNT_SOCIALES_CIUDADANAS", "PUNT_INGLES"
)
ETIQUETAS_CORTAS <- c("Global", "Lectura", "Matemát.", "Naturales", "Sociales", "Inglés")
ORDEN_ESTRATO <- c(paste("Estrato", 1:6), "Sin Estrato")
ETIQUETA_ESTRATO <- c(as.character(1:6), "Sin\nestrato")
NATURALEZAS <- c("OFICIAL", "NO OFICIAL")
FRASE_RURAL <- "Vive en una zona rural donde no hay estratificación socioeconómica"
MINIMO_POR_CELDA <- 10
MINIMO_ESCALA_CORRELACION <- 0.5
REPETICIONES <- 3
COLOR_PRINCIPAL <- "#2C4A6E"
COLORES_NATURALEZA <- c(OFICIAL = "#0072B2", `NO OFICIAL` = "#D55E00")
TAMANO_HISTOGRAMA <- c(2.6, 2.0)
TAMANO_CORRELACION <- c(2.8, 2.4)
PUNTOS_POR_PULGADA <- 300
miles <- scales::label_number(big.mark = ".", decimal.mark = ",")
decimal <- scales::label_number(accuracy = 0.1, decimal.mark = ",")
tema <- theme_bw(base_size = 9, base_family = "serif") +
  theme(panel.grid.minor = element_blank(), legend.position = "top", legend.title = element_blank())
# Las figuras de herramientas van a un cuarto del ancho de página en el informe: se dibujan
# con letra de 11 puntos, como en el cuaderno, para que impresas queden en unos 6; el mapa
# va sin cifras en las celdas, igual que en Python.
tema_herramientas <- theme_bw(base_size = 11, base_family = "serif") +
  theme(panel.grid.minor = element_blank())

saber <- read_parquet(file.path(raiz, "data", "saber11-20224.parquet"))
saber$FAMI_ESTRATOVIVIENDA[saber$FAMI_ESTRATOVIVIENDA %in% FRASE_RURAL] <- "Sin Estrato"

# ---- Estadística descriptiva -------------------------------------------------

descriptivos <- function(x) {
  x <- x[!is.na(x)]
  cuartiles <- unname(quantile(x, c(0.25, 0.75), type = 7))
  conteos <- table(x)
  list(
    n = length(x), media = mean(x), mediana = median(x),
    moda = as.numeric(names(conteos)[which.max(conteos)]),
    varianza = var(x), desviacion = sd(x), minimo = min(x), maximo = max(x),
    rango = max(x) - min(x), q1 = cuartiles[1], q3 = cuartiles[2],
    iqr = cuartiles[2] - cuartiles[1],
    asimetria = skewness(x), curtosis = kurtosis(x) - 3
  )
}

frecuencias <- function(x, orden) {
  conteos <- table(factor(x, levels = orden))
  c(as.list(setNames(as.integer(conteos), orden)), list(`(sin respuesta)` = sum(is.na(x))))
}

chi_cuadrado <- function(filas, columnas, orden_filas, orden_columnas) {
  tabla <- table(factor(filas, levels = orden_filas), factor(columnas, levels = orden_columnas))
  stopifnot(min(tabla) >= MINIMO_POR_CELDA)
  prueba <- chisq.test(tabla, correct = FALSE)
  n <- sum(tabla)
  estadistico <- unname(prueba$statistic)
  list(
    n = n, estadistico = estadistico, gl = unname(prueba$parameter), p = prueba$p.value,
    v_cramer = sqrt(estadistico / (n * (min(dim(tabla)) - 1)))
  )
}

matriz_a_lista <- function(m) setNames(lapply(rownames(m), function(f) as.list(m[f, ])), rownames(m))

completos <- saber[stats::complete.cases(saber[, PUNTAJES]), PUNTAJES]
pearson <- cor(completos, method = "pearson")
spearman <- cor(completos, method = "spearman")

# ---- Gráficos en ggplot2 -----------------------------------------------------

histograma_ggplot2 <- function(puntajes) {
  ggplot(data.frame(puntaje = puntajes), aes(puntaje)) +
    geom_histogram(breaks = seq(0, 500, 10), fill = COLOR_PRINCIPAL) +
    scale_y_continuous(labels = miles) +
    labs(x = "Puntaje global (0 a 500)", y = "Estudiantes") +
    tema_herramientas
}

correlacion_ggplot2 <- function(matriz) {
  largo <- expand.grid(fila = ETIQUETAS_CORTAS, columna = ETIQUETAS_CORTAS)
  largo$valor <- as.vector(matriz)
  ggplot(largo, aes(columna, fila, fill = valor)) +
    geom_tile() +
    scale_fill_viridis_c(option = "cividis", limits = c(MINIMO_ESCALA_CORRELACION, 1), labels = decimal, name = NULL) +
    scale_y_discrete(limits = rev) +
    labs(x = NULL, y = NULL) +
    tema_herramientas +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

caja_ggplot2 <- function(datos) {
  datos <- datos[!is.na(datos$FAMI_ESTRATOVIVIENDA) & !is.na(datos$COLE_NATURALEZA), ]
  datos$estrato <- factor(datos$FAMI_ESTRATOVIVIENDA, levels = ORDEN_ESTRATO, labels = ETIQUETA_ESTRATO)
  datos$naturaleza <- factor(datos$COLE_NATURALEZA, levels = NATURALEZAS)
  ggplot(datos, aes(estrato, PUNT_GLOBAL, fill = naturaleza)) +
    geom_boxplot(outlier.shape = NA, coef = 1.5, linewidth = 0.3, width = 0.7) +
    scale_fill_manual(values = COLORES_NATURALEZA, labels = c("Colegio oficial", "Colegio no oficial")) +
    coord_cartesian(ylim = c(80, 530)) +
    labs(x = "Estrato de la vivienda", y = "Puntaje global (0 a 500)") +
    tema
}

# Componentes de una expresión de ggplot2 encadenada con `+` (ggplot, capas, escalas,
# etiquetas y tema): la función tiene una sola instrucción, así que esto mide su tamaño.
componentes <- function(expresion) {
  if (is.call(expresion) && identical(expresion[[1]], as.name("+"))) {
    return(componentes(expresion[[2]]) + componentes(expresion[[3]]))
  }
  1
}

# Mide lo mismo que la sección 8 del cuaderno: instrucciones del cuerpo de la función,
# mediana de tres tiempos de construir y exportar a PNG, y peso del PNG y del PDF.
medir <- function(nombre, construir, datos, tamano) {
  ruta_png <- file.path(carpeta_figuras, sprintf("herramientas-%s-ggplot2.png", nombre))
  tiempos <- vapply(seq_len(REPETICIONES), function(i) {
    inicio <- Sys.time()
    ggsave(ruta_png, construir(datos), width = tamano[1], height = tamano[2], dpi = PUNTOS_POR_PULGADA)
    as.numeric(difftime(Sys.time(), inicio, units = "secs"))
  }, numeric(1))
  ruta_pdf <- tempfile(fileext = ".pdf")
  ggsave(ruta_pdf, construir(datos), width = tamano[1], height = tamano[2])
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

medicion <- list(
  histograma = medir("histograma", histograma_ggplot2, saber$PUNT_GLOBAL, TAMANO_HISTOGRAMA),
  correlacion = medir("correlacion", correlacion_ggplot2, pearson, TAMANO_CORRELACION)
)
ggsave(
  file.path(carpeta_figuras, "caja-estrato-naturaleza-ggplot2.png"), caja_ggplot2(saber),
  width = 2.9, height = 2.3, dpi = PUNTOS_POR_PULGADA
)

# ---- Resultados para la paridad ----------------------------------------------

resultados <- list(
  version_r = paste(R.version$major, R.version$minor, sep = "."),
  version_ggplot2 = as.character(packageVersion("ggplot2")),
  licencia_ggplot2 = packageDescription("ggplot2")$License,
  descriptivos = setNames(lapply(PUNTAJES, function(p) descriptivos(saber[[p]])), PUNTAJES),
  frecuencias = list(
    FAMI_ESTRATOVIVIENDA = frecuencias(saber$FAMI_ESTRATOVIVIENDA, ORDEN_ESTRATO),
    COLE_NATURALEZA = frecuencias(saber$COLE_NATURALEZA, NATURALEZAS)
  ),
  correlacion = list(n = nrow(completos), pearson = matriz_a_lista(pearson), spearman = matriz_a_lista(spearman)),
  chi2 = list(
    estrato_naturaleza = chi_cuadrado(saber$FAMI_ESTRATOVIVIENDA, saber$COLE_NATURALEZA, ORDEN_ESTRATO, NATURALEZAS),
    internet_zona = chi_cuadrado(saber$FAMI_TIENEINTERNET, saber$COLE_AREA_UBICACION, c("Si", "No"), c("URBANO", "RURAL"))
  ),
  medicion = medicion
)
ruta_json <- file.path(carpeta_resultados, "fase-1.json")
write_json(resultados, ruta_json, auto_unbox = TRUE, digits = NA, pretty = TRUE)

cat(sprintf("%s y ggplot2 %s\n", R.version.string, resultados$version_ggplot2))
cat(sprintf("Estudiantes leídos: %s\n", miles(nrow(saber))))
print(round(sapply(resultados$descriptivos, unlist), 3))
for (prueba in names(resultados$chi2)) {
  r <- resultados$chi2[[prueba]]
  cat(sprintf(
    "%s: chi2 = %s, gl = %d, V de Cramér = %s\n",
    prueba, miles(round(r$estadistico)), r$gl, formatC(r$v_cramer, format = "f", digits = 3, decimal.mark = ",")
  ))
}
cat(sprintf("Escrito %s\n", sub(paste0(raiz, "/"), "", ruta_json, fixed = TRUE)))
