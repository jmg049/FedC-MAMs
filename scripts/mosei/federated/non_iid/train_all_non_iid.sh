#!/bin/bash
set -e
RUNS=3
CONFIGS=("configs/mosei/federated/mosei_federated_multimodal_non_iid_01.yaml" "configs/mosei/federated/mosei_federated_multimodal_non_iid_03.yaml" "configs/mosei/federated/mosei_federated_multimodal_non_iid_05.yaml" "configs/mosei/federated/mosei_federated_multimodal_non_iid_05.yaml")

for run_id in $(seq 1 $RUNS); do

    for config in "${CONFIGS[@]}"; do
        echo "Running" $config "with run_id" $run_id
        if [ ! -f $config ]; then
            echo "File not found!"
        fi

        python train_federated.py --config $config --run_id $run_id
    done

done
