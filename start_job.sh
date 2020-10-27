#!/usr/bin/env bash

# gradient jobs create \
#   --name "dfl-training" \
#   --machineType "P6000" \
#   --container "ufoym/deepo:all-py36-cu100" \
#   --command "bash scripts/run.sh --target-iteration 250000" \
#   --workspace $(pwd) \
#   --projectId "pr6b7rwbn" \
#   --createOptionsFile $(pwd)/config.yaml

gradient jobs create --optionsFile ./config.yaml