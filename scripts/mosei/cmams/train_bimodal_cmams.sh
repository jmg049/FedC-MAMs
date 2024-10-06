#!/bin/bash

set -e

RUNS=3

for i in $(seq 1 "$RUNS"); do
#    python train_cmams.py --config configs/mosei/cmams/mosei_av_to_l.yaml --run_id "$i"
    python train_cmams.py --config configs/mosei/cmams/mosei_al_to_v.yaml --run_id "$i"
    python train_cmams.py --config configs/mosei/cmams/mosei_vl_to_a.yaml --run_id "$i"
done
