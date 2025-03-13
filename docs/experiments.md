# Experiment Documentation
Details the scripts and outputs required to generate the main results.  

We generated the input data of newborn screen metabolites paired with outcomes
of interest as one CSV file `data/processed/neonatal_conditions.csv` with paired
metadata as `data/processed/metadata.csv`. The `row_id` identifiers are an
arbitrary sequence of unique identifiers to this dataset.  

## Main Bottleneck Modeling Results
```bash
python scripts/experiments/run_deep_mtl_bottleneck.py \
--input data/processed/neonatal_conditions.csv \
--output results/deep_mtl/neonatal_bottleneck_validation/ \
--tasks config/neonatal_covariates.txt \
--drop_sparse \
--validate
```

Ultimately, we used the `ensemble` family of models.  
Subgroup discovery was subsequently performed with the following:
```bash
python scripts/subgroup_discovery/analysis/bottleneck_subgroup_results.py \
  --input results/deep_mtl/neonatal_bottleneck_validation/ensemble_bottle_1/ \
  --output results/subgroup_discovery/metabolic_health_index/ \
  --tasks config/neonatal_covariates.txt
```

## Model Variants
### Model Without IVH
```bash
python scripts/experiments/run_deep_mtl_bottleneck.py \
  --input data/processed/neonatal_conditions.csv \
  --output results/deep_mtl/supplementary_variants/without_ivh/modeling/ \
  --column_specification config/without_ivh/colspec.yml \
  --drop_sparse \
  --bottleneck_sequence 1 \
  --validate
```

Subsequent subgroup discovery
```bash
python scripts/subgroup_discovery/analysis/bottleneck_subgroup_results.py \
  --input results/deep_mtl/supplementary_variants/without_ivh/modeling/ensemble_bottle_1/ \
  --output results/deep_mtl/supplementary_variants/without_ivh/subgroup_discovery/ \
  --tasks config/without_ivh/neonatal_covariates.txt
```