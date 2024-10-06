#!/bin/bash

set -e

bash scripts/mosei/federated/train_iid.sh
bash scripts/mosei/federated/train_all_non_iid.sh
