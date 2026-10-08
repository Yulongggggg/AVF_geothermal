#!/bin/bash
#SBATCH --job-name=geouno_repo_review
#SBATCH --partition=cpu
#SBATCH --qos=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=provenance/repository_review_20261008/tests_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=2
export GEOUNO_TEST_RECOMPUTE=1
export PYTHONPATH="${GEOUNO_REVIEW_DEPS:?shared test dependency directory}:$PWD"
review_python="${GEOUNO_REVIEW_PYTHON:-python}"
/usr/bin/time -v "$review_python" provenance/repository_review_20261008/run_checks.py
