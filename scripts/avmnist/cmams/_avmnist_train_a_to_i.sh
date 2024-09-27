#!/bin/bash

set -e

dataset="avmnist"
# Default values
runs=3
cmam_configs_root="configs/${dataset}/cmams"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --runs)
        runs="$2"
        shift 2
        ;;
    --cmam_configs_root)
        cmam_configs_root="$2"
        shift 2
        ;;
    *)
        echo "Unknown argument: $1"
        exit 1
        ;;
    esac
done
for run_idx in $(seq 1 $runs); do
    python3 train_cmams.py --config "${cmam_configs_root}/avmnist_a_to_i.yaml" --run_id "$run_idx"

done
python3 metric_output/process_cmam_test_metrics.py --metrics_path "experiments/avmnist/metrics/AVMNIST (CMAMs A-I): Baseline Training" --inner_dir_name "cmam_A_to_I" --output_path "experiments/avmnist/metrics/AVMNIST (CMAMs A-I): Baseline Training/a_to_i.tex"
