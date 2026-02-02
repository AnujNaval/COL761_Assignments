#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <path_graphs> <path_discriminative_subgraphs> <path_features>"
    exit 1
fi

python3 convert.py "$1" "$2" "$3"