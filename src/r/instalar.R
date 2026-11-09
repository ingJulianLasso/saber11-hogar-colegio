# Instala los paquetes de R del proyecto en la biblioteca de usuario y registra
# sus versiones (con las de sus dependencias) en r-versiones.txt, en la raíz del
# repositorio.
#
# Uso: Rscript src/r/instalar.R
# Requisitos del sistema (Arch): r, gcc-fortran y nlopt.

paquetes <- c("ggplot2", "dplyr", "broom", "moments", "car", "nanoparquet", "jsonlite")
espejo <- "https://cloud.r-project.org"

argumento <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
raiz <- normalizePath(file.path(dirname(sub("^--file=", "", argumento)), "..", ".."))

# Sin la carpeta creada, R no la agrega a .libPaths() e install.packages()
# intenta escribir en la biblioteca del sistema.
biblioteca <- Sys.getenv("R_LIBS_USER")
dir.create(biblioteca, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(biblioteca, .libPaths()))

faltantes <- paquetes[!vapply(paquetes, requireNamespace, logical(1), quietly = TRUE)]
if (length(faltantes) > 0) {
  install.packages(faltantes, lib = biblioteca, repos = espejo, Ncpus = parallel::detectCores())
}
for (paquete in paquetes) library(paquete, character.only = TRUE)

instalados <- installed.packages()
dependencias <- tools::package_dependencies(
  paquetes, db = instalados, which = c("Depends", "Imports", "LinkingTo"), recursive = TRUE
)
base <- rownames(instalados)[instalados[, "Priority"] %in% "base"]
todos <- sort(setdiff(unique(c(paquetes, unlist(dependencias))), base))
versiones <- vapply(todos, function(p) as.character(packageVersion(p)), character(1))

writeLines(
  c(sprintf("R==%s", getRversion()), sprintf("%s==%s", todos, versiones)),
  file.path(raiz, "r-versiones.txt")
)
cat(sprintf("%d paquetes registrados en %s\n", length(todos), file.path(raiz, "r-versiones.txt")))
