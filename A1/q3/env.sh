#!/bin/bash

if ! command -v python3.10 &> /dev/null; then
    echo "Error: Python 3.10 is not installed or not in the PATH."
    echo "If you have sudo access, try: sudo apt-get install python3.10"
    exit 1
fi

python3.10 -m pip install --upgrade pip

python3.10 -m pip install --user numpy networkx

echo "Compiling C++ mining tools..."

if [ -d "mining_tools" ]; then
    cd mining_tools
    
    if [ -f "Makefile" ] || [ -f "makefile" ]; then
        make clean  
        make
        if [ $? -eq 0 ]; then
            echo "Compilation successful."
        else
            echo "Error: Compilation failed."
            exit 1
        fi
    else
        echo "Error: No Makefile found in mining_tools."
        exit 1
    fi
    
    cd ..
else
    echo "Error: Directory 'mining_tools' not found."
    exit 1
fi

echo "Environment setup complete."