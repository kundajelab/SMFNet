#!/bin/bash
#SBATCH -J fiberbam_to_calls
#SBATCH --requeue
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --array=0-22

# Fiber-seq BAM -> calls TSV (<chrom>.tsv.zst), one chromosome per array task (chr1-22, chrX).
#
# Submit from this directory, adding your cluster's partition or account options:
#   BAM=... REGIONS=... FASTA=... OUT=... sbatch -o "$OUT/logs/%A_%a.log" fiberbam_to_calls.sbatch
#   BAM=... REGIONS=... FASTA=... OUT=... sbatch --array=20 ...   # chr21 only, to check
#
# Each step runs in the repository's pixi environment, which provides samtools and zstd.
# Re-running a task overwrites that chromosome, so a preempted task just restarts. chr1
# needs about 16 GB of memory.
set -euo pipefail
# If your cluster provides pixi as a module, load it here, for example:
# module load devel pixi

: "${BAM:?set BAM to an indexed Fiber-seq BAM}"
: "${REGIONS:?set REGIONS to a region BED from select_regions.py}"
: "${FASTA:?set FASTA to the reference FASTA}"
: "${OUT:?set OUT to the output directory}"
# sbatch runs a copy of this script, so the repository is found from the submit directory.
HERE=${HERE:-$SLURM_SUBMIT_DIR}
REPO=${REPO:-$HERE/../..}
PIXI_ENV=${PIXI_ENV:-default}
# The default environment includes CUDA builds of PyTorch, which pixi only runs where a
# CUDA driver is present. Data preparation doesn't use the GPU, so allow CPU nodes too.
export CONDA_OVERRIDE_CUDA=${CONDA_OVERRIDE_CUDA:-12}

CHROMS=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 \
        chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22 chrX)
CHROM=${CHROMS[$SLURM_ARRAY_TASK_ID]}

run() { pixi run --frozen --manifest-path "$REPO/pyproject.toml" -e "$PIXI_ENV" "$@"; }

for f in "$BAM" "$BAM.bai" "$REGIONS" "$FASTA" "$HERE/fiberbam_to_calls.py"; do
    [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
done
# Catch a truncated file before spending hours streaming it.
run samtools quickcheck -v "$BAM"
mkdir -p "$OUT"

echo "=== $CHROM $(date) bam=$BAM regions=$REGIONS ==="
run python "$HERE/fiberbam_to_calls.py" \
    --bam "$BAM" --regions "$REGIONS" --fasta "$FASTA" --chrom "$CHROM" \
    --threads "$SLURM_CPUS_PER_TASK" --out "$OUT/$CHROM.tsv.zst"
echo "DONE $CHROM $(date)"
