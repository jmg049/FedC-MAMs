#!/bin/bash

set -e

dataset="mosei"
# Default values
runs=3
configs_root="configs/${dataset}"

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

for run_id in $(seq 1 "$runs"); do
    python train_multimodal.py \
        --config "${configs_root}/train_mosei_multimodal.yaml" \
        --run_id "$run_id"
done
