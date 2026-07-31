#!/bin/bash

if [ -z "$1" ]; then
    echo "====================================================" >&2
    echo "ERROR: NO IP ADDRESS!" >&2
    echo "USAGE: ./submit.sh <IP-ADDRESS>" >&2
    echo "====================================================" >&2
    exit 1
fi

echo "Checking if MLflow is reachable at http://$1:5050..."
if ! curl -s --connect-timeout 3 "http://$1:5050" > /dev/null; then
    echo "====================================================" >&2
    echo "ERROR: MLflow server not reachable at http://$1:5050" >&2
    echo "Please check the IP address and ensure MLflow is running." >&2
    echo "====================================================" >&2
    exit 1
fi
echo "MLflow is reachable."

echo "Submitting training job with MLFlow IP: $1"
sbatch train.slurm "$1"
