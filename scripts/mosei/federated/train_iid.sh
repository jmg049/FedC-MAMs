#!/bin/bash

set -e
RUNS=3
CONFIG="configs/mosei/federated/mosei_federated_multimodal.yaml"

for run_id in $(seq 1 $RUNS); do

    echo "Running" $CONFIG "with run_id" $run_id
    if [ ! -f $CONFIG ]; then
        echo "File not found!"
    fi

    python train_federated.py --config $CONFIG --run_id $run_id

done
