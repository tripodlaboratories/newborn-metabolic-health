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

# Model Variants in Supplementary Analyses: Adding Covariates
First, we created a input CSV variant with additional metadata features:
```bash
python scripts/data_processing/create_metadata_features.py \
  --input data/processed/neonatal_conditions.csv \
  --metadata data/processed/metadata.csv \
  --output data/processed/variants/neonatal_conditions_meta.csv
```

## Model Variants: Adding Infant Sex as Feature
```bash
python scripts/experiments/run_deep_mtl_bottleneck.py \
  --input data/processed/variants/neonatal_conditions_meta.csv \
  --output results/deep_mtl/supplementary_variants/with_infant_sex/modeling/ \
  --column_specification config/with_infant_sex/colspec.yml \
  --drop_sparse \
  --bottleneck_sequence 1 \
  --validate
```
Subsequent subgroup discovery
```bash
python scripts/subgroup_discovery/analysis/bottleneck_subgroup_results.py \
  --input results/deep_mtl/supplementary_variants/with_infant_sex/modeling/ensemble_bottle_1/ \
  --output results/deep_mtl/supplementary_variants/with_infant_sex/subgroup_discovery/ \
  --config config/with_infant_sex/subgroup_discovery.yml
```

### Feature Importance Analysis
Create a checkpointed model for downstream feature importance analysis.
```bash
python scripts/experiments/interpretability/train_bottleneck_model_checkpoint.py \
  --input data/processed/variants/neonatal_conditions_meta.csv \
  --output results/interpretability/with_infant_sex/model_outputs/ \
  --column_specification config/with_infant_sex/colspec.yml \
  --drop_sparse \
  --bottleneck 1
```
Run feature analysis script.
```bash
python scripts/experiments/interpretability/calculate_feature_importance.py \
  -i results/interpretability/with_infant_sex/model_outputs/ \
  -o results/interpretability/with_infant_sex/scores/
```
Then analyze within notebooks.

## Model Variants: Gestational Age, Birthweight, and Infant Sex as Features
```bash
python scripts/experiments/run_deep_mtl_bottleneck.py \
  --input data/processed/variants/neonatal_conditions_meta.csv \
  --output results/deep_mtl/supplementary_variants/with_ga_bwt_sex/modeling/ \
  --column_specification config/with_ga_bwt_sex/colspec.yml \
  --drop_sparse \
  --bottleneck_sequence 1 \
  --validate
```

Subsequent Subgroup Discovery
```bash
python scripts/subgroup_discovery/analysis/bottleneck_subgroup_results.py \
  --input results/deep_mtl/supplementary_variants/with_ga_bwt_sex/modeling/ensemble_bottle_1/ \
  --output results/deep_mtl/supplementary_variants/with_ga_bwt_sex/subgroup_discovery/ \
  --config config/with_ga_bwt_sex/subgroup_discovery.yml
```

### Feature Importance
```bash
python scripts/experiments/interpretability/train_bottleneck_model_checkpoint.py \
  --input data/processed/variants/neonatal_conditions_meta.csv \
  --output results/interpretability/with_ga_bwt_sex/model_outputs/ \
  --column_specification config/with_ga_bwt_sex/colspec.yml \
  --drop_sparse \
  --bottleneck 1
```
Followed by feature analysis script:
```bash
python scripts/experiments/interpretability/calculate_feature_importance.py \
  -i results/interpretability/with_ga_bwt_sex/model_outputs/ \
  -o results/interpretability/with_ga_bwt_sex/scores/
```

## Model Variants: Minimal MLP with Gestational Age and Birthweight

# Model Variants in Supplementary Analyses: Ablation Studies
## Model Variants: Ablation By Removing One of the Outcomes
### Model Without IVH
```bash
python scripts/experiments/run_deep_mtl_bottleneck.py \
  --input data/processed/neonatal_conditions.csv \
  --output results/deep_mtl/supplementary_variants/without_ivh/modeling/ \
  --column_specification config/without_ivh/colspec.yml \
  --drop_sparse \
  --bottleneck_sequence 1 \
  --validate \
  --lr_scheduler "ReduceLROnPlateau"
```

Subsequent subgroup discovery
```bash
python scripts/subgroup_discovery/analysis/bottleneck_subgroup_results.py \
  --input results/deep_mtl/supplementary_variants/without_ivh/modeling/ensemble_bottle_1/ \
  --output results/deep_mtl/supplementary_variants/without_ivh/subgroup_discovery/ \
  --config config/without_ivh/subgroup_discovery.yml
```

### Model Without NEC
```bash
python scripts/experiments/run_deep_mtl_bottleneck.py \
  --input data/processed/neonatal_conditions.csv \
  --output results/deep_mtl/supplementary_variants/without_nec/modeling/ \
  --column_specification config/without_nec/colspec.yml \
  --drop_sparse \
  --bottleneck_sequence 1 \
  --validate \
  --lr_scheduler "ReduceLROnPlateau"
```
Subsequent subgroup discovery
```bash
python scripts/subgroup_discovery/analysis/bottleneck_subgroup_results.py \
  --input results/deep_mtl/supplementary_variants/without_nec/modeling/ensemble_bottle_1/ \
  --output results/deep_mtl/supplementary_variants/without_nec/subgroup_discovery/ \
  --config config/without_nec/subgroup_discovery.yml
```
### Model Without BPD
```bash
python scripts/experiments/run_deep_mtl_bottleneck.py \
  --input data/processed/neonatal_conditions.csv \
  --output results/deep_mtl/supplementary_variants/without_bpd/modeling/ \
  --column_specification config/without_bpd/colspec.yml \
  --drop_sparse \
  --bottleneck_sequence 1 \
  --validate \
  --lr_scheduler "ReduceLROnPlateau"
```
Subsequent subgroup discovery
```bash
python scripts/subgroup_discovery/analysis/bottleneck_subgroup_results.py \
  --input results/deep_mtl/supplementary_variants/without_bpd/modeling/ensemble_bottle_1/ \
  --output results/deep_mtl/supplementary_variants/without_bpd/subgroup_discovery/ \
  --config config/without_bpd/subgroup_discovery.yml
```

### Model Without ROP
```bash
python scripts/experiments/run_deep_mtl_bottleneck.py \
  --input data/processed/neonatal_conditions.csv \
  --output results/deep_mtl/supplementary_variants/without_rop/modeling/ \
  --column_specification config/without_rop/colspec.yml \
  --drop_sparse \
  --bottleneck_sequence 1 \
  --validate \
  --lr_scheduler "ReduceLROnPlateau"
```
Subsequent subgroup discovery
```bash
python scripts/subgroup_discovery/analysis/bottleneck_subgroup_results.py \
  --input results/deep_mtl/supplementary_variants/without_rop/modeling/ensemble_bottle_1/ \
  --output results/deep_mtl/supplementary_variants/without_rop/subgroup_discovery/ \
  --config config/without_rop/subgroup_discovery.yml
```
