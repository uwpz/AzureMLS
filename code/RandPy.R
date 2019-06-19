
.libPaths()
.libPaths("C:/tmp")
.libPaths()

packageurl <- "http://cran.r-project.org/src/contrib/Archive/ggplot2/ggplot2_0.9.1.tar.gz"
install.packages(packageurl, repos=NULL, type="source")

library("splines", lib.loc="C:/Program Files/R/R-3.4.4/library")

remove.packages("corrplot")
install.packages("http://cran.r-project.org/src/contrib/Archive/corrplot/corrplot_0.73.tar.gz", repos=NULL, type="source")


library(reticulate)
getwd()
use_virtualenv("venv")


Sys.getenv("RETICULATE_PYTHON")
Sys.setenv("RETICULATE_PYTHON" = "C:/My/AzureMLS/venv/Scripts/python.exe")
reticulate::py_config()

#ubuntu: "usr/local/lib/R/site-library"
