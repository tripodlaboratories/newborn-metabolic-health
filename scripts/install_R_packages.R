# Install additional packages if they don't exist
PACKAGES <- c(
  "RMTL",
  "caret",
  "glmnet",
  "pROC",
  "tibble",
  "dplyr",
  "readr",
  "optparse",
  "doParallel",
  "parallel",
  "testthat"
)

check_installed_packages <- function(...) {
  all_packages <- unlist(list(...))
  is_installed <- unlist(lapply(all_packages, require, character.only=TRUE))
  absent_packages <- all_packages[is_installed == FALSE]
  if (length(absent_packages) > 0) { 
    install.packages(absent_packages)
    lapply(absent_packages, require, character.only=TRUE)
  }
}

check_installed_packages(PACKAGES)