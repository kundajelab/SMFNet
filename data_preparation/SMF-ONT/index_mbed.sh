#!/usr/bin/env bash
# Sort per-molecule calls, compress them with bgzip, and create a tabix index.
set -euo pipefail

if [[ $# -eq 0 || ${1:-} == --help ]]; then
    echo 'Usage: bash data_extraction/index_mbed.sh INPUT OUTPUT.gz [THREADS=4]'
    echo 'Sort, bgzip and tabix-index five-column delta-string mBED; preserves additional columns.'
    exit 0
fi

# Check arguments before creating any output files.
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo 'Expected INPUT OUTPUT.gz [THREADS]' >&2
    exit 2
fi
input=$1
output=$2
threads=${3:-4}

if [[ ! $threads =~ ^[1-9][0-9]*$ ]]; then
    echo 'THREADS must be positive' >&2
    exit 2
fi
if [[ ! -f "$input" ]]; then
    echo 'Input file is missing' >&2
    exit 1
fi
if [[ -e "$output" || -e "$output.tbi" ]]; then
    echo 'Output or index already exists' >&2
    exit 1
fi
for tool in sort bgzip tabix awk; do
    if ! command -v "$tool" >/dev/null; then
        echo "Missing tool: $tool" >&2
        exit 1
    fi
done

# Keep temporary files beside the destination and clean them up on exit.
mkdir -p -- "$(dirname -- "$output")"
temporary=$(mktemp -d "$(dirname -- "$output")/.index_mbed.XXXXXX")
trap 'rm -rf -- "$temporary"' EXIT

read_input() {
    # Accept both plain text and gzip-compatible compressed input.
    case "$input" in
        *.gz|*.bgz) gzip -cd -- "$input" ;;
        *) cat -- "$input" ;;
    esac
}

# Validate required columns, then sort by chromosome, start and end.
# A stable sort preserves the order of records with identical coordinates.
read_input |
    awk '
        BEGIN { FS = "\t" }
        /^#/ || NF == 0 { next }
        NF < 5 || $2 !~ /^[0-9]+$/ || $3 !~ /^[0-9]+$/ ||
        $3 < $2 || $4 == "" || $5 !~ /^([0-9]+[mux])+$/ {
            print "Invalid mBED row " NR > "/dev/stderr"
            exit 1
        }
        { print }
    ' |
    LC_ALL=C sort -s -T "$temporary" -S 1G --parallel="$threads" -k1,1 -k2,2n -k3,3n |
    bgzip -@ "$threads" -c > "$temporary/calls.gz"

# Publish the compressed calls only after their index has been built.
tabix -p bed "$temporary/calls.gz"
mv -- "$temporary/calls.gz" "$output"
mv -- "$temporary/calls.gz.tbi" "$output.tbi"
echo "Wrote $output and $output.tbi"
