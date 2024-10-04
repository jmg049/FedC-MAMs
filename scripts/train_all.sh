#!/bin/bash

set -e

bash scripts/avmnist/federated/non_iid/train_all_non_iid.sh

bash scripts/mosei/federated/train_iid.sh
bash scripts/mosei/federated/non_iid/train_all_non_iid.sh
