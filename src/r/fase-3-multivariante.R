# Fase 3 en R: replica el análisis multivariante de notebooks/fase-3-modelado.ipynb con prcomp
# (valores propios y cargas del PCA de las cinco áreas estandarizadas), las medias de cada
# clúster con las asignaciones de K-Means y de Ward que hizo Python, y chisq.test del clúster
# de K-Means por estrato, zona y naturaleza; dibuja en ggplot2 el plano PCA por celdas y los
# perfiles de los clústeres, mide el plano con la especificación de la comparación de
# herramientas del cuaderno, y escribe resultados/r/fase-3-multivariante.json, que el cuaderno
# compara con sus valores (sección 19).
#
# Uso: Rscript src/r/fase-3-multivariante.R, o Source en RStudio con el archivo abierto.
# Requisitos: los paquetes de src/r/instalar.R y los intermedios que escribe
# `uv run python -m saber11.intermedios --fase 3` cuando segmentacion.K_KMEANS está fijada:
# data/intermedio/fase-3-multivariante.parquet (las áreas crudas y estandarizadas de cada
# estudiante de entrenamiento, su clúster y sus variables de perfil) y
# data/intermedio/fase-3-pca-celdas.parquet (las celdas del plano PCA, cada una con 10 o más).
#
# K-Means no se vuelve a ajustar en R: kmeans() usa otro algoritmo (Hartigan-Wong) y otra
# inicialización, así que no daría las mismas etiquetas. La paridad va sobre el PCA y sobre lo
# que se calcula con las asignaciones de Python. La V de Cramér sale de la χ² sin corrección,
# con la misma fórmula del cuaderno.

suppressPackageStartupMessages({
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
  stop("No se encontró la raíz del repositorio: ejecute desde la raíz o con Rscript src/r/fase-3-multivariante.R")
}

raiz <- resolver_raiz()
carpeta_figuras <- file.path(raiz, "figuras", "fase-3", "r")
carpeta_resultados <- file.path(raiz, "resultados", "r")
dir.create(carpeta_figuras, recursive = TRUE, showWarnings = FALSE)
dir.create(carpeta_resultados, recursive = TRUE, showWarnings = FALSE)

REPETICIONES <- 3
PUNTOS_POR_PULGADA <- 300
TAMANO_PLANO <- c(2.0, 1.75)  # panel de la comparación de herramientas; el mismo del cuaderno
TAMANO_PERFILES <- c(3.4, 2.6)
# Los mismos colores de los clústeres del cuaderno (COLORES_CLUSTER), distintos de los de la naturaleza.
COLORES_CLUSTER <- c("#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#4C4C4C")
ETIQUETAS_AREAS <- c(
  PUNT_LECTURA_CRITICA = "Lectura\ncrítica", PUNT_MATEMATICAS = "Matemáticas", PUNT_C_NATURALES = "Ciencias\nnaturales",
  PUNT_SOCIALES_CIUDADANAS = "Sociales y\nciudadanas", PUNT_INGLES = "Inglés"
)
PERFILES <- c(estrato = "FAMI_ESTRATOVIVIENDA", zona = "COLE_AREA_UBICACION", naturaleza = "COLE_NATURALEZA")
decimal <- scales::label_number(accuracy = 0.1, decimal.mark = ",", big.mark = ".", style_negative = "minus")
tema <- theme_bw(base_size = 8, base_family = "serif") +
  theme(panel.grid.minor = element_blank(), legend.position = "top", legend.title = element_blank(), axis.text = element_text(size = 7.5), legend.text = element_text(size = 7.5), legend.key.size = unit(8, "pt"))

datos <- as.data.frame(read_parquet(file.path(raiz, "data", "intermedio", "fase-3-multivariante.parquet")))
celdas <- as.data.frame(read_parquet(file.path(raiz, "data", "intermedio", "fase-3-pca-celdas.parquet")))
columnas_z <- grep("^Z_", names(datos), value = TRUE)
areas <- sub("^Z_", "", columnas_z)
clusteres <- sort(unique(datos$kmeans))

v_cramer <- function(prueba) {
  sqrt(unname(prueba$statistic) / (sum(prueba$observed) * (min(dim(prueba$observed)) - 1)))
}

filas_json <- function(tabla) {
  lapply(seq_len(nrow(tabla)), function(i) as.list(tabla[i, , drop = FALSE]))
}

# ---- PCA de las cinco áreas estandarizadas --------------------------------------

pca <- prcomp(as.matrix(datos[columnas_z]), center = TRUE, scale. = FALSE)
valores_propios <- pca$sdev^2
cargas <- unname(pca$rotation)

# ---- Medias por clúster con las asignaciones de Python --------------------------

medias_por <- function(asignacion) {
  do.call(rbind, lapply(sort(unique(asignacion)), function(k) {
    filas <- datos[asignacion == k, c(areas, columnas_z)]
    data.frame(cluster = k, n = nrow(filas), t(colMeans(filas)), check.names = FALSE)
  }))
}
medias <- list(kmeans = medias_por(datos$kmeans), ward = medias_por(datos$ward))

# ---- χ² del clúster por estrato, zona y naturaleza ------------------------------

pruebas <- lapply(PERFILES, function(variable) {
  completos <- !is.na(datos[[variable]])
  prueba <- chisq.test(table(datos$kmeans[completos], datos[[variable]][completos]), correct = FALSE)
  list(prueba = prueba, resultado = list(n = sum(prueba$observed), gl = unname(prueba$parameter), chi2 = unname(prueba$statistic), p = prueba$p.value, v = v_cramer(prueba)))
})

# ---- Gráficos en ggplot2 --------------------------------------------------------

varianza <- valores_propios / sum(valores_propios)
titulo_componente <- function(j) sprintf("Componente %d (%s %%)", j, decimal(100 * varianza[j]))
celdas$cluster <- factor(celdas$cluster, levels = clusteres)

plano_ggplot2 <- function(tabla) {
  ggplot(tabla) +
    geom_rect(aes(xmin = x_inferior, xmax = x_superior, ymin = y_inferior, ymax = y_superior, fill = cluster, alpha = proporcion), colour = "white", linewidth = 0.15) +
    scale_fill_manual(values = setNames(COLORES_CLUSTER[seq_along(clusteres)], clusteres), labels = paste("Clúster", clusteres)) +
    scale_alpha_continuous(range = c(0.25, 1), guide = "none") +
    scale_x_continuous(labels = decimal) +
    scale_y_continuous(labels = decimal) +
    labs(x = titulo_componente(1), y = titulo_componente(2)) +
    tema
}

perfiles <- do.call(rbind, lapply(seq_len(nrow(medias$kmeans)), function(i) {
  data.frame(cluster = factor(medias$kmeans$cluster[i], levels = clusteres), area = factor(ETIQUETAS_AREAS[areas], levels = ETIQUETAS_AREAS[areas]), media = unlist(medias$kmeans[i, columnas_z]))
}))

perfiles_ggplot2 <- function(tabla) {
  ggplot(tabla, aes(area, media, colour = cluster, group = cluster)) +
    geom_hline(yintercept = 0, colour = "#7F7F7F", linewidth = 0.3) +
    geom_line(linewidth = 0.5) +
    geom_point(size = 1.4) +
    scale_colour_manual(values = setNames(COLORES_CLUSTER[seq_along(clusteres)], clusteres), labels = paste("Clúster", clusteres)) +
    scale_y_continuous(labels = decimal) +
    labs(x = NULL, y = "Media estandarizada (z)") +
    tema +
    theme(axis.text.x = element_text(size = 7))
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

ggsave(file.path(carpeta_figuras, "perfiles-ggplot2.png"), perfiles_ggplot2(perfiles), width = TAMANO_PERFILES[1], height = TAMANO_PERFILES[2], dpi = PUNTOS_POR_PULGADA)
medicion <- list(plano = medir("plano-pca", plano_ggplot2, celdas, TAMANO_PLANO))

# ---- Resultados para la paridad -------------------------------------------------

resultados <- list(
  version_r = paste(R.version$major, R.version$minor, sep = "."),
  version_ggplot2 = as.character(packageVersion("ggplot2")),
  n = nrow(datos),
  areas = areas,
  pca = list(valores_propios = valores_propios, cargas = lapply(seq_len(ncol(cargas)), function(j) cargas[, j])),
  medias = lapply(medias, filas_json),
  chi2 = lapply(pruebas, `[[`, "resultado"),
  celdas = list(n = nrow(celdas), estudiantes = sum(celdas$n), menor = min(celdas$n)),
  medicion = medicion
)
ruta_json <- file.path(carpeta_resultados, "fase-3-multivariante.json")
write_json(resultados, ruta_json, auto_unbox = TRUE, digits = NA, pretty = TRUE)

cat(sprintf("%s y ggplot2 %s\n", R.version.string, resultados$version_ggplot2))
cat(sprintf("\nPCA de las cinco áreas estandarizadas (%d estudiantes de entrenamiento)\n", nrow(datos)))
print(summary(pca)$importance)
print(round(pca$rotation, 4))
cat("\nMedias de las áreas por clúster de K-Means (asignaciones de Python)\n")
print(medias$kmeans[c("cluster", "n", areas)], digits = 5, row.names = FALSE)
for (nombre in names(pruebas)) {
  cat(sprintf("\nClúster de K-Means por %s\n", nombre))
  print(pruebas[[nombre]]$prueba)
  cat(sprintf("V de Cramér: %.4f\n", pruebas[[nombre]]$resultado$v))
}
cat(sprintf("\nPlano PCA: %d celdas con %d estudiantes; la menor tiene %d\n", resultados$celdas$n, resultados$celdas$estudiantes, resultados$celdas$menor))
cat(sprintf("Escrito %s\n", sub(paste0(raiz, "/"), "", ruta_json, fixed = TRUE)))
