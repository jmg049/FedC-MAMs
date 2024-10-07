#!/bin/bash

set -e

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

for i in $(seq 1 $runs); do
    python train_federated.py --config configs/avmnist/federated/avmnist_federated_multimodal_non_iid_05.yaml --run_id $i
    python train_federated_cmams.py --config configs/avmnist/federated/cmams/avmnist_federated_a_to_i_non_iid_05.yaml --run_id $i
    python train_federated_cmams.py --config configs/avmnist/federated/cmams/avmnist_federated_i_to_a_non_iid_05.yaml --run_id $i

    python train_federated.py --config configs/avmnist/federated/avmnist_federated_multimodal_non_iid_03.yaml --run_id $i
    python train_federated_cmams.py --config configs/avmnist/federated/cmams/avmnist_federated_a_to_i_non_iid_03.yaml --run_id $i
    python train_federated_cmams.py --config configs/avmnist/federated/cmams/avmnist_federated_i_to_a_non_iid_03.yaml --run_id $i

    python train_federated.py --config configs/avmnist/federated/avmnist_federated_multimodal_non_iid_01.yaml --run_id $i
    python train_federated_cmams.py --config configs/avmnist/federated/cmams/avmnist_federated_a_to_i_non_iid_01.yaml --run_id $i
    python train_federated_cmams.py --config configs/avmnist/federated/cmams/avmnist_federated_i_to_a_non_iid_01.yaml --run_id $i
done

## already have trained the 1st run

python train_federated_cmams.py --config configs/avmnist/federated/cmams/avmnist_federated_i_to_a_iid.yaml --run_id 1
python train_federated_cmams.py --config configs/avmnist/federated/cmams/avmnist_federated_a_to_i_iid.yaml --run_id 1

python train_federated.py --config configs/avmnist/federated/avmnist_federated_multimodal_iid.yaml --run_id 2
python train_federated_cmams.py --config configs/avmnist/federated/cmams/avmnist_federated_i_to_a_iid.yaml --run_id 2
python train_federated_cmams.py --config configs/avmnist/federated/cmams/avmnist_federated_a_to_i_iid.yaml --run_id 2

python train_federated.py --config configs/avmnist/federated/avmnist_federated_multimodal_iid.yaml --run_id 3
python train_federated_cmams.py --config configs/avmnist/federated/cmams/avmnist_federated_i_to_a_iid.yaml --run_id 3
python train_federated_cmams.py --config configs/avmnist/federated/cmams/avmnist_federated_a_to_i_iid.yaml --run_id 3
