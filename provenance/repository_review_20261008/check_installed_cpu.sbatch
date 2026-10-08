#!/bin/bash
#SBATCH --job-name=geouno_wheel_review
#SBATCH --partition=cpu
#SBATCH --qos=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=provenance/repository_review_20261008/wheel_%j.log

set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=2
export GEOUNO_REVIEW_ROOT="${SLURM_SUBMIT_DIR:?submit from the repository root}"
export PYTHONPATH="${GEOUNO_REVIEW_INSTALLED:?installed wheel directory}"
review_python="${GEOUNO_REVIEW_PYTHON:-python}"
cd "${GEOUNO_REVIEW_INSTALLED:?}/.."
/usr/bin/time -v "$review_python" "$GEOUNO_REVIEW_ROOT/provenance/repository_review_20261008/check_installed.py"
