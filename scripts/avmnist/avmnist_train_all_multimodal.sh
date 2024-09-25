#!/bin/bash
set -e
runs=3
for i in $(seq 1 $runs); do
    python3 train_multimodal.py --config configs/avmnist/avmnist_multimodal.yaml --run_id $i
done
