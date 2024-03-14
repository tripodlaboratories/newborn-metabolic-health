# Deep learning-based risk stratification in preterm infants using NBS metabolites
This repository contains code related to the manuscript entitled _Quantitative
assessment of neonatal health using dried blood spot metabolite profiles and
deep learning_.

# Data Availability
Pre-existing data policies outlines by the California State Biobank, the California Office of
Statewide Health Planning and Development (OSHPD), and the California Perinatal
Quality Care Collaborative (CPQCC) govern data access requests. Requests will be
reviewed by the steering committees from each organization prior to providing
access.

# Requirements
For the manuscript, model implementation was done using Python 3.7.6 and
`pytorch` 1.6.0. Model training handlers used `pandas` 1.0.1 and `scikit-learn` 0.22.1.  
Subgroup discovery results depend on `pysubgroup`.
For data visualization and plotting in the manuscript, R 3.6.1 was used along
with the `dplyr`, `tidyr`, `yardstick`, `pROC`, `ggthemes`, `ggplot2`,
`tidygraph`, and `ggraph` packages.

## Installation
With conda: `conda env create -f environment.yml`  
Python module code should be installed using `pip install -e .`

We expect installation of python packages and R packages to take minutes on a
standard desktop computer or server.

## Tests
Tests depend on `pytest` 5.3.0.  
Use `make test` or `pytest -v ./tests/ `to run all tests.

# Usage
## Reproducing Manuscript Results
Individual manuscript results can be reproduced with various `make` commands.
Obtain a list of make rules with `make help`

## Using Models in New Analyses
See model training scripts in `./scripts/` for examples of model initialization
and repeated K-Fold training handlers.

