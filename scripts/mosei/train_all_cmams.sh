#!/bin/bash

set -e

dataset="mosei"
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

bash scripts/mosei/train_mosei_unimodal_cmams.sh --runs $runs --configs_root $cmam_configs_root
bash scripts/mosei/train_mosei_bimodal_cmams.sh --runs $runs --configs_root $cmam_configs_root