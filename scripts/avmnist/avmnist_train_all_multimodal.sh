#!/bin/bash

set -e

dataset="avmnist"
# Default values
runs=3
cmam_configs_root="configs/${dataset}/cmams"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --runs)
        runs="$2"
        shift 2
        ;;
    --configs_root)
        configs_root="$2"
        shift 2
        ;;
    *)
        echo "Unknown argument: $1"
        exit 1
        ;;
    esac
done

bash scripts/avmnist
