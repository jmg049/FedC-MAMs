#!/bin/bash

set -e

RUNS=3

for i in $(seq 1 "$RUNS"); do
    python train_cmams.py --config configs/mosei/cmams/mosei_a_to_vl.yaml --run_id "$i"
    python train_cmams.py --config configs/mosei/cmams/mosei_v_to_al.yaml --run_id "$i"
    python train_cmams.py --config configs/mosei/cmams/mosei_l_to_av.yaml --run_id "$i"
done
