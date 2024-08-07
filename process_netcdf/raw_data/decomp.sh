#!/bin/bash

for tf in ./*.tar; do
    # Check if the file exists
    if [ -e "$tf" ]; then
        echo "Extracting $tf"
        tar -xf "$tf"
    else
        echo "No .tar files found."
    fi
done
