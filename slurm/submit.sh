#!/bin/bash

if [ -z "$1" ]; then
    echo "====================================================" >&2
    echo "ERROR: NO IP ADDRESS!" >&2
    echo "USAGE: ./submit.sh <IP-ADDRESS>" >&2
    echo "====================================================" >&2
    exit 1
fi

echo "Submitting training job with MLFlow IP: $1"
sbatch train.slurm "$1"
