#!/bin/bash

GSPAN_BIN=$1
FSG_BIN=$2
GASTON_BIN=$3
DATASET=$4
OUTPUT_DIR=$5

mkdir -p "$OUTPUT_DIR"

python3 runner.py \
    --gspan "$GSPAN_BIN" \
    --fsg "$FSG_BIN" \
    --gaston "$GASTON_BIN" \
    --dataset "$DATASET" \
    --outdir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo "Mining complete. Generating plot..."
    python3 plotter.py --outdir "$OUTPUT_DIR"
else
    echo "Error occurred during mining phase."
    exit 1
fi