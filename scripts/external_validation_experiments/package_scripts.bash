#! /bin/bash
package_dir='pkg/to_package/'
health_index_dir="${package_dir}health_index/"
sg_dir="${package_dir}subgroup_discovery/"

# Package config
config_dir="${package_dir}/config"; mkdir -p $config_dir
cp -r config/ontario/ $config_dir

# Package biobank project scripts
module_source="./biobank_project/"
module_dir="${package_dir}/src/"; mkdir -p $module_dir
cp -r $module_source $module_dir
cp setup.py $module_dir
cp environment.yml $module_dir

# Packaging metabolic health index artifacts and scripts for external validation
health_index_checkpoints_dir="${health_index_dir}/checkpoints/"; mkdir -p $health_index_checkpoints_dir
cp -r checkpoints/for_ontario_nbs/metabolic_health_index/ $health_index_checkpoints_dir

validation_scripts_source="./scripts/external_validation_experiments/"
scripts_dir="${package_dir}scripts/"; mkdir -p $scripts_dir
cp "${validation_scripts_source}/infer_with_health_index_model.py" $scripts_dir

# Packaging subgroup discovery scripts for external validation
subgroup_discovery_artifacts_dir="checkpoints/for_ontario_nbs/subgroup_discovery/"
subgroup_discovery_artifacts=(${subgroup_discovery_artifacts_dir}*.joblib)
mkdir -p $sg_dir
for f in "${subgroup_discovery_artifacts[@]}"; do
    cp "$f" "$sg_dir"
done

cp "${validation_scripts_source}/evaluate_external_subgroups.py" $scripts_dir

# Package additional health index results for combined subgroup discovery
# experiment
health_index_results_dir="${health_index_dir}/results/"; mkdir -p $health_index_results_dir
cp -r results/deep_mtl/neonatal_bottleneck_validation/ensemble_bottle_1/. $health_index_results_dir
cp data/processed/neonatal_conditions.csv $health_index_results_dir

# Zip final packaging command for rclone
package_outdir="pkg/packaged/"; mkdir -p $package_outdir
artifact_file="${package_outdir}metabolic_health.tar.gz"
tar -czvf $artifact_file $package_dir
rclone copy --no-traverse $artifact_file nbs-gdrive:pkg/

