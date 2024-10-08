#!/bin/bash

# set -e

# dataset="avmnist"
# # Default values
runs=3
# cmam_configs_root="configs/${dataset}/cmams"

# # Parse command-line arguments
# while [[ $# -gt 0 ]]; do
#     case $1 in
#     --runs)
#         runs="$2"
#         shift 2
#         ;;
#     --configs_root)
#         configs_root="$2"
#         shift 2
#         ;;
#     *)
#         echo "Unknown argument: $1"
#         exit 1
#         ;;
#     esac
# done
## Need to finish some of the IID-C-MAMs

for run in $(seq 1 $runs); do
    python train_federated_cmams.py --config configs/avmnist/federated/cmams/avmnist_federated_i_to_a_iid.yaml --run_id $run
done

python train_federated_cmams.py --config configs/avmnist/federated/cmams/avmnist_federated_a_to_i_iid.yaml --run_id 2
python train_federated_cmams.py --config configs/avmnist/federated/cmams/avmnist_federated_a_to_i_iid.yaml --run_id 3

## Do non-iid for alpha=0.1
for run in $(seq 1 $runs); do
    python train_federated.py --config configs/avmnist/federated/avmnist_federated_multimodal_non_iid_01.yaml --run_id $run
done

## Non-IID C-MAMs
python train_federated_cmams.py --config configs/avmnist/federated/cmams/avmnist_federated_a_to_i_non_iid_05.yaml --run_id 3
python train_federated_cmams.py --config configs/avmnist/federated/cmams/avmnist_federated_i_to_a_non_iid_05.yaml --run_id 2
python train_federated_cmams.py --config configs/avmnist/federated/cmams/avmnist_federated_i_to_a_non_iid_05.yaml --run_id 3

for run in $(seq 1 $runs); do
    python train_federated_cmams.py --config configs/avmnist/federated/cmams/avmnist_federated_a_to_i_non_iid_03.yaml --run_id $run
    python train_federated_cmams.py --config configs/avmnist/federated/cmams/avmnist_federated_i_to_a_non_iid_03.yaml --run_id $run
    python train_federated_cmams.py --config configs/avmnist/federated/cmams/avmnist_federated_a_to_i_non_iid_01.yaml --run_id $run
    python train_federated_cmams.py --config configs/avmnist/federated/cmams/avmnist_federated_i_to_a_non_iid_01.yaml --run_id $run
done
