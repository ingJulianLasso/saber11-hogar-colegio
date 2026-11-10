# Fase 2 en R: replica con lm la regresión simple y la múltiple que ajusta
# notebooks/fase-2-preparacion.ipynb sobre el mismo entrenamiento preparado, calcula
# el laboratorio descriptivo antes y después de preparar, dibuja en ggplot2 los
# coeficientes con su intervalo, los residuos agrupados y las frecuencias del estrato,
# mide las dos primeras figuras con la especificación de la sección 16 del cuaderno y
# escribe resultados/r/fase-2.json, que el cuaderno compara con sus valores (sección 15).
#
# Uso: Rscript src/r/fase-2.R, o Source en RStudio con el archivo abierto.
# Requisitos: los paquetes de src/r/instalar.R y los intermedios que escribe
# `uv run python -m saber11.intermedios` (data/intermedio/fase-2-entrenamiento.parquet
# y resultados/fase-2-niveles.json).
#
# Definiciones, las mismas del cuaderno: varianza y desviación muestrales; moda, el
# valor más frecuente y el menor si hay empate; rango, máximo menos mínimo.

suppressPackageStartupMessages({
  library(broom)
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
  stop("No se encontró la raíz del repositorio: ejecute desde la raíz o con Rscript src/r/fase-2.R")
}

raiz <- resolver_raiz()
carpeta_figuras <- file.path(raiz, "figuras", "fase-2", "r")
carpeta_resultados <- file.path(raiz, "resultados", "r")
dir.create(carpeta_figuras, recursive = TRUE, showWarnings = FALSE)
dir.create(carpeta_resultados, recursive = TRUE, showWarnings = FALSE)

REPETICIONES <- 3
PUNTOS_POR_PULGADA <- 300
MINIMO_POR_CELDA <- 10
TAMANO_COEFICIENTES <- c(3.6, 3.0)
TAMANO_RESIDUOS <- c(2.6, 2.2)
COLOR_PRINCIPAL <- "#2C4A6E"
COLOR_APOYO <- "#7F7F7F"
SIN_DATO <- "Sin dato"
miles <- scales::label_number(big.mark = ".", decimal.mark = ",")
entero <- scales::label_number(accuracy = 1, decimal.mark = ",", big.mark = ".")
tema <- theme_bw(base_size = 9, base_family = "serif") +
  theme(panel.grid.minor = element_blank(), legend.position = "top", legend.title = element_blank())
tema_herramientas <- theme_bw(base_size = 10, base_family = "serif") +
  theme(panel.grid.minor = element_blank())

niveles <- fromJSON(file.path(raiz, "resultados", "fase-2-niveles.json"), simplifyVector = FALSE)
datos <- read_parquet(file.path(raiz, "data", "intermedio", "fase-2-entrenamiento.parquet"))
datos <- as.data.frame(datos)
for (variable in names(niveles$categoricas)) {
  categoria <- niveles$categoricas[[variable]]
  datos[[variable]] <- relevel(factor(datos[[variable]], levels = unlist(categoria$niveles)), ref = categoria$referencia)
}

# ---- Regresión -----------------------------------------------------------------

terminos <- c(unlist(niveles$numericas), names(niveles$categoricas), unlist(niveles$indicadores))
formula_multiple <- reformulate(terminos, response = niveles$objetivo)
formula_simple <- reformulate(niveles$predictor_simple, response = niveles$objetivo)
simple <- lm(formula_simple, data = datos)
multiple <- lm(formula_multiple, data = datos)

# Nombre común de un término, el mismo de saber11.modelos.clave_termino: VARIABLE=nivel
# en las categóricas, el nombre de la columna en el resto e "Intercept" para la constante.
clave_termino <- function(nombre) {
  if (nombre == "(Intercept)") {
    return("Intercept")
  }
  for (variable in names(niveles$categoricas)[order(-nchar(names(niveles$categoricas)))]) {
    if (startsWith(nombre, variable)) {
      return(paste0(variable, "=", substring(nombre, nchar(variable) + 1)))
    }
  }
  nombre
}

coeficientes <- function(modelo) {
  tabla <- tidy(modelo, conf.int = TRUE)
  setNames(
    lapply(seq_len(nrow(tabla)), function(i) {
      list(estimado = tabla$estimate[i], error = tabla$std.error[i], inferior = tabla$conf.low[i], superior = tabla$conf.high[i])
    }),
    vapply(tabla$term, clave_termino, character(1))
  )
}

resumen <- function(modelo) {
  ajuste <- glance(modelo)
  list(r2 = ajuste$r.squared, r2_ajustado = ajuste$adj.r.squared, n = nobs(modelo), coeficientes = coeficientes(modelo))
}

# ---- Laboratorio descriptivo ---------------------------------------------------

descriptivos <- function(x) {
  x <- x[!is.na(x)]
  conteos <- table(x)
  list(
    n = length(x), media = mean(x), mediana = median(x),
    moda = as.numeric(names(conteos)[which.max(conteos)]),
    varianza = var(x), desviacion = sd(x), rango = max(x) - min(x)
  )
}

frecuencias <- function(x) {
  x <- as.character(x)
  x[is.na(x)] <- SIN_DATO
  as.list(table(x))
}

prefijo <- niveles$prefijo_antes
laboratorio <- list(
  numericas = setNames(lapply(unlist(niveles$descriptivas_numericas), function(variable) {
    list(antes = descriptivos(datos[[paste0(prefijo, variable)]]), despues = descriptivos(datos[[variable]]))
  }), unlist(niveles$descriptivas_numericas)),
  categoricas = setNames(lapply(unlist(niveles$descriptivas_categoricas), function(variable) {
    list(antes = frecuencias(datos[[paste0(prefijo, variable)]]), despues = frecuencias(datos[[variable]]))
  }), unlist(niveles$descriptivas_categoricas))
)

# ---- Gráficos en ggplot2 -------------------------------------------------------

etiquetas_figura <- unlist(niveles$terminos_figura)
tabla_multiple <- tidy(multiple, conf.int = TRUE)
tabla_multiple$clave <- vapply(tabla_multiple$term, clave_termino, character(1))
coeficientes_figura <- tabla_multiple[tabla_multiple$clave %in% names(etiquetas_figura), ]
coeficientes_figura$etiqueta <- etiquetas_figura[coeficientes_figura$clave]
residuos_figura <- data.frame(ajustado = fitted(multiple), residuo = residuals(multiple))

coeficientes_ggplot2 <- function(tabla) {
  ggplot(tabla, aes(estimate, reorder(etiqueta, estimate))) +
    geom_vline(xintercept = 0, colour = COLOR_APOYO, linewidth = 0.4) +
    geom_pointrange(aes(xmin = conf.low, xmax = conf.high), colour = COLOR_PRINCIPAL, size = 0.25) +
    scale_x_continuous(labels = entero) +
    labs(x = "Puntos frente a la referencia (IC del 95 %)", y = NULL) +
    tema_herramientas
}

residuos_ggplot2 <- function(tabla) {
  ggplot(tabla, aes(ajustado, residuo)) +
    geom_bin_2d(aes(fill = after_stat(ifelse(count >= MINIMO_POR_CELDA, count, NA))), binwidth = c(5, 8)) +
    geom_hline(yintercept = 0, colour = COLOR_APOYO, linetype = "dashed", linewidth = 0.4) +
    scale_fill_viridis_c(option = "cividis", trans = "log10", labels = miles, na.value = NA, name = "Estudiantes") +
    scale_x_continuous(labels = miles) +
    labs(x = "Valor ajustado", y = "Residuo (puntos)") +
    tema_herramientas
}

estrato_ggplot2 <- function(tabla) {
  ggplot(tabla, aes(nivel, porcentaje, fill = momento)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.75) +
    scale_fill_manual(values = c(Antes = COLOR_APOYO, `Después` = COLOR_PRINCIPAL)) +
    scale_y_continuous(labels = entero) +
    labs(x = "Estrato", y = "% del entrenamiento") +
    tema +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# Componentes de una expresión de ggplot2 encadenada con `+`: la función tiene una sola
# instrucción, así que esto mide su tamaño.
componentes <- function(expresion) {
  if (is.call(expresion) && identical(expresion[[1]], as.name("+"))) {
    return(componentes(expresion[[2]]) + componentes(expresion[[3]]))
  }
  1
}

# Mide lo mismo que la sección 16 del cuaderno: instrucciones del cuerpo de la función,
# mediana de tres tiempos de construir y exportar a PNG, y peso del PNG y del PDF.
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

medicion <- list(
  coeficientes = medir("coeficientes", coeficientes_ggplot2, coeficientes_figura, TAMANO_COEFICIENTES),
  residuos = medir("residuos", residuos_ggplot2, residuos_figura, TAMANO_RESIDUOS)
)

estrato <- laboratorio$categoricas$FAMI_ESTRATOVIVIENDA
orden_estrato <- c(paste("Estrato", 1:6), "Sin Estrato", SIN_DATO)
porcentajes <- function(conteos, momento) {
  total <- sum(unlist(conteos))
  data.frame(
    nivel = factor(orden_estrato, levels = orden_estrato),
    porcentaje = vapply(orden_estrato, function(n) if (is.null(conteos[[n]])) 0 else 100 * conteos[[n]] / total, numeric(1)),
    momento = momento
  )
}
ggsave(
  file.path(carpeta_figuras, "laboratorio-estrato-ggplot2.png"),
  estrato_ggplot2(rbind(porcentajes(estrato$antes, "Antes"), porcentajes(estrato$despues, "Después"))),
  width = 3.4, height = 2.6, dpi = PUNTOS_POR_PULGADA
)

# ---- Resultados para la paridad ------------------------------------------------

resultados <- list(
  version_r = paste(R.version$major, R.version$minor, sep = "."),
  version_ggplot2 = as.character(packageVersion("ggplot2")),
  simple = resumen(simple),
  multiple = resumen(multiple),
  laboratorio = laboratorio,
  medicion = medicion
)
ruta_json <- file.path(carpeta_resultados, "fase-2.json")
write_json(resultados, ruta_json, auto_unbox = TRUE, digits = NA, pretty = TRUE)

cat(sprintf("%s y ggplot2 %s\n", R.version.string, resultados$version_ggplot2))
cat(sprintf("Estudiantes de entrenamiento: %s\n", miles(nrow(datos))))
print(summary(simple))
cat(sprintf(
  "Regresión múltiple: %d coeficientes, R² = %s\n",
  length(coef(multiple)), formatC(resultados$multiple$r2, format = "f", digits = 4, decimal.mark = ",")
))
cat(sprintf("Escrito %s\n", sub(paste0(raiz, "/"), "", ruta_json, fixed = TRUE)))
