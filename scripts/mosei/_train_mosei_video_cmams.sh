#!/bin/bash
set -e

dataset="mosei"
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
        --configs_root)
            configs_root="$2"
            shift 2
        ;;
        *)
            echo "Unknown argument: $1"
            exit 1
        ;;
    esac
done

# Run the training script
for run_idx in $(seq 1 $runs); do
    python3 train_cmams.py --config "${configs_root}/mosei_cmams_v_to_a_${run_idx}.yaml"
    python3 train_cmams.py --config "${configs_root}/mosei_cmams_v_to_l_${run_idx}.yaml"
done

python3 metric_output/process_cmam_test_metrics.py --metrics_path "experiments/mosei/metrics/MOSEI (CMAMs V-A): Baseline Training" --inner_dir_name "cmam_V_to_A" --output_path "experiments/mosei/metrics/MOSEI (CMAMs V-A): Baseline Training/v_to_a.tex"
python3 metric_output/process_cmam_test_metrics.py --metrics_path "experiments/mosei/metrics/MOSEI (CMAMs V-T): Baseline Training" --inner_dir_name "cmam_V_to_T" --output_path "experiments/mosei/metrics/MOSEI (CMAMs V-T): Baseline Training/v_to_t.tex"