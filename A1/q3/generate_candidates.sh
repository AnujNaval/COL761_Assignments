#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <path_database_graph_features> <path_query_graph_features> <path_out_file>"
    exit 1
fi

python3 generate_candidates.py "$1" "$2" "$3"