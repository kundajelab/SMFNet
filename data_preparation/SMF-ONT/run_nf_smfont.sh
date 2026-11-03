#!/usr/bin/env bash
# Launch the bundled workflow with settings appropriate to the input format.
set -euo pipefail

if [[ $# -eq 0 || $1 == --help ]]; then
    cat <<'HELP'
Usage: bash data_extraction/run_nf_smfont.sh bam|raw [--dry-run] NEXTFLOW_ARGUMENTS...

bam: pre-called BAM/CRAM with MM/ML tags; raw: POD5/FAST5 with a custom Remora model.
Provide --input_folder DIR --sample_name NAME --ref reference.fa --outdir DIR.
Raw input also needs --remora_model_path model.pt.
NF_PROFILE defaults to apptainer,multi_gpus. NF_CONFIG adds a cluster config.
--dry-run prints the shell-quoted command without starting Nextflow.
HELP
    exit 0
fi

mode=$1
shift
case "$mode" in
    bam|raw) ;;
    *) echo 'Mode must be bam or raw' >&2; exit 2 ;;
esac

# Resolve bundled files relative to this script, not the working directory.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
config="$script_dir/config/nf_smfont_${mode}.config"
if [[ -n ${NF_CONFIG:-} ]]; then
    config="$config,$NF_CONFIG"
fi
workflow="$script_dir/vendor/nf-smfont/main.nf"
if [[ ! -f "$workflow" ]]; then
    echo 'Bundled nf-smfont workflow is missing' >&2
    exit 1
fi

# An array preserves argument boundaries, including paths containing spaces.
nextflow_command=(
    nextflow run "$workflow"
    -c "$config"
    -profile "${NF_PROFILE:-apptainer,multi_gpus}"
)
if [[ ${1:-} == --dry-run ]]; then
    shift
    printf '%q ' "${nextflow_command[@]}" "$@"
    printf '\n'
    exit 0
fi

if ! command -v nextflow >/dev/null; then
    echo 'Activate an environment providing Nextflow first' >&2
    exit 1
fi
exec "${nextflow_command[@]}" "$@"
