#!/bin/bash

APRIORI=$1
FPGROWTH=$2
DATASET=$3
OUTDIR=$4

mkdir -p "$OUTDIR"

python3 run_experiments.py "$APRIORI" "$FPGROWTH" "$DATASET" "$OUTDIR"
python3 plot.py "$OUTDIR"