#!/bin/bash
#SBATCH --job-name=geouno_review
#SBATCH --partition=cpu
#SBATCH --qos=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=provenance/review_20261007/equivalence_%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MALLOC_ARENA_MAX=2
export PYTHONPATH=.
review_python="${GEOUNO_REVIEW_PYTHON:-python}"
hostname
/usr/bin/time -v "$review_python" provenance/verify_equivalence.py --low-memory --threads 1 \
  --report provenance/review_20261007/equivalence.json
