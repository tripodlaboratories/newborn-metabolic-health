.PHONY: environment tests integration_tests
.PHONY: correlation_networks
.PHONY: performance_curves subgroup_discovery single_unit_analysis
.PHONY: feature_removal_results

#############
# VARIABLES #
#############
# Helper Functions
# An empty newline function for pretty printing during `foreach` calls
define \n


endef

############
# COMMANDS #
############
# Comment command with '##' for the self-documenting feature

## Create python environment
environment:
	conda env create -f environment.yml --force
	source activate newborn-metabolic-health || conda activate newborn-metabolic-health
	pip install -e .

## Run tests
test:
	pytest -v ./tests/

## Create correlation network plots
correlation_networks:
	Rscript ./scripts/plotting/correlation_networks/plot_metabolites_only_corrnet.R
	Rscript ./scripts/plotting/correlation_networks/metabolite_correlation_differences.R
	Rscript ./scripts/plotting/correlation_networks/plots_metabolite_only_corrnet_bottleneck_unit.R

## Create plots for AUROC/AUPR curves
performance_curves:
	Rscript ./scripts/plotting/deep_mtl_results/plot_deep_mtl_rocpr_curves.R \
		-i ./results/deep_mtl/neonatal/validation/ensemble/ \
		-o ./results/deep_mtl/neonatal/validation/plots/ensemble_test.pdf \
		--compare_prc_vs_random
	Rscript ./scripts/plotting/deep_mtl_results/plot_deep_mtl_rocpr_curves.R \
		-i ./results/deep_mtl/neonatal/validation/ensemble/ \
		--preds_filename valid_preds.csv \
		--true_vals_file ./data/processed/neonatal_conditions.csv \
		-o ./results/deep_mtl/neonatal/validation/plots/ensemble_valid.pdf \
		--compare_prc_vs_random
	Rscript ./scripts/plotting/deep_mtl_results/plot_deep_mtl_rocpr_curves.R \
		-i ./results/deep_mtl/neonatal_bottleneck_validation/ensemble_bottle_1/ \
		--preds_filename bottleneck.csv \
		-o ./results/deep_mtl/neonatal_bottleneck_validation/ensemble_1unit_bottleaspreds_test.pdf \
		--controls_as_positive \
		--compare_prc_vs_random
	Rscript ./scripts/plotting/deep_mtl_results/plot_deep_mtl_rocpr_curves.R \
		-i ./results/deep_mtl/neonatal_bottleneck_validation/ensemble_bottle_1/ \
		--preds_filename valid_bottleneck.csv \
		--true_vals_file ./data/processed/neonatal_conditions.csv \
		-o ./results/deep_mtl/neonatal_bottleneck_validation/ensemble_1unit_bottleaspreds_valid.pdf \
		--controls_as_positive \
		--compare_prc_vs_random

## Subgroup discovery results
subgroup_discovery:
	Rscript ./scripts/subgroup_discovery/prediction_vis.R
	Rscript ./scripts/subgroup_discovery/bottleneck_vis.R

## Iterative feature removal experiments
feature_removal_results:
	Rscript ./scripts/plotting/deep_mtl_results/feature_removal_results.R \
		-i ./results/deep_mtl/neonatal/feature_removal/ \
		-o ./results/deep_mtl/plots/feature_removal_by_total_aupr.pdf \
		--feature_file ./results/deep_mtl/neonatal/feature_removal/ensemble_46_feat_total_aupr_ranks/features.txt \
		--total_aupr_ranks

## Further analysis of single unit bottleneck
single_unit_analysis:
	Rscript ./scripts/plotting/deep_mtl_results/plot_singleunit_bottleneck_outcome_distro.R \
		-i ./results/deep_mtl/neonatal_bottleneck_validation/ensemble_bottle_1/ \
		--preds_filename bottleneck.csv \
		-o ./results/deep_mtl/neonatal_bottleneck_validation/ensemble_1unit_bottleaspreds_outcome_distro.pdf
	Rscript ./scripts/plotting/deep_mtl_results/get_bottleneck_outputs.R \
		-i ./results/deep_mtl/neonatal_bottleneck_validation/
	Rscript ./scripts/plotting/deep_mtl_results/plot_bottleneck_outputs_projection.R \
		-i ./results/deep_mtl/neonatal_bottleneck_validation/bottleneck_layer_outputs/ \
		-o projection_by_gestage.pdf --models "ensemble" --by_gest_age

############################
# Self Documenting Commands #
#############################
.DEFAULT_GOAL := help

# From https://github.com/drivendata/cookiecutter-data-science Makefile
# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
#		* save line in hold space
#		* purge line
#		* Loop:
#			* append newline + line to hold space
#			* go to next line
#			* if line starts with doc comment, strip comment character off and loop
#		* remove target prerequisites
#		* append hold space (+ newline) to line
#		* replace newline plus comments by `---`
#		* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
		}" ${MAKEFILE_LIST} \
		| LC_ALL='C' sort --ignore-case \
		| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=25 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
		'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
		line_length -= length(words[i]) + 1; \
		if (line_length <= 0) { \
		line_length = ncol - indent - length(words[i]) - 1; \
		printf "\n%*s ", -indent, " "; \
		} \
		printf "%s ", words[i]; \
		} \
		printf "\n"; \
		}' \
		| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
