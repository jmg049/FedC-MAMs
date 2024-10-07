#!/bin/bash
set -e
RUNS=3

for run_id in $(seq 2 $RUNS); do
    python train_federated.py --config "configs/avmnist/federated/avmnist_federated_multimodal_iid.yaml" --run_id $run_id
done
