# SMF data extraction

Prepares the `smf_mat`/`smf_pos` H5 files used by SMFNet from ONT per-molecule methylation calls. The file is created from mBED files following data processing with the [nf-smfont pipeline](https://git.embl.org/grp-krebs/nf-smfont/) (see [Boulanger et al. Molecular Cell 2026](https://www.cell.com/molecular-cell/fulltext/S1097-2765(26)00506-X)). 

Proocessing can start from raw POD5, methyl-called BAM, or mBED files.

## Steps

| Input | Preprocessing | H5 extraction |
| --- | --- | --- |
| ONT POD5/FAST5 | `run_nf_smfont.sh raw` → called, aligned reads → indexed mBED | `extract_mbed.py` |
| ONT BAM/CRAM with methylation calls | `run_nf_smfont.sh bam` → indexed mBED | `extract_mbed.py` |
| Existing delta-string mBED | `index_mbed.sh` only if sorting/compression/indexing is needed | `extract_mbed.py` |


## Inputs

| Input | Requirement |
| --- | --- |
| Regions | Three-column BED, 0-based half-open by default; chromosome names must match the reference |
| ONT reads | Raw POD5/FAST5 plus a suitable custom methylation model, or BAM/CRAM already containing the required MM/ML calls |
| ONT reference | Matching FASTA; `extract_mbed.py` requires its `.fai` index |
| mBED | BGZF-compressed, tabix-indexed per-molecule delta strings from nf-smfont's custom mBED route |

An ordinary alignment BAM without methylation calls is not sufficient for the `bam` route. The raw route requires the appropriate pore/basecalling and Remora models. The custom model used in the historical runs is not bundled here.


## ONT: nf-smfont preprocessing

The complete, unmodified [nf-SMFONT v1.0.0 Zenodo release](https://doi.org/10.5281/zenodo.21919078) is included under `vendor/nf-smfont`, with its MIT license and upstream docs. The launcher uses this local copy, so it does not fetch the workflow from GitLab. Nextflow plugins, institutional configuration and containers may still require network access.

Activate Nextflow and Apptainer before running. The default profile is `apptainer,multi_gpus`, matching the recovered launch commands. This profile does **not** select a Slurm partition: upstream defaults to the local executor.
Run inside an appropriate allocation, or supply your site's executor/resourcesettings using `NF_CONFIG=path/to/cluster.config`. `NF_PROFILE` overrides the profiles. Raw basecalling needs suitable GPU resources.

### From already-called BAMs

```bash
bash data_extraction/run_nf_smfont.sh bam \
  --input_folder input/ont_bams --sample_name SAMPLE \
  --ref input/reference.fa --outdir output/ont
```

The directory should contain that sample's BAMs directly. For CRAM input add `--input_ext cram`. The workflow inspects input status and retains the mapping stage while disabling basecalling and methylation calling for this route.

### From raw signal

```bash
bash data_extraction/run_nf_smfont.sh raw \
  --input_folder input/pod5 --sample_name SAMPLE \
  --ref input/reference.fa --outdir output/ont \
  --remora_model_path input/models/custom_CG_GC_model.pt
```

For FAST5 add `--input_ext fast5`. The raw template uses Bonito with `dna_r10.4.1_e8.2_400bps_sup@v4.2.0` and a custom Remora model, as recorded in the original runs. These settings must match the actual chemistry and model. The bundled release has newer container versions than the historical runs; this is a portable example, not a claim of bit-identical historical basecalls.
Add `-resume` to resume a run with its work directory and Nextflow cache intact. Insert `--dry-run` immediately after `bam` or `raw` to print the command only.

Both templates use these explicit mBED settings:

| Parameter | Value |
| --- | --- |
| Read quality threshold | 7 |
| Methylation LLR threshold | 2.0 |
| Motifs | `cpg,gpc` |
| Custom mBED route | `modkit=false` |
| Filter by read methylation rate | Enabled |
| Minimum / maximum methylation rate | 0.05 / 0.95 |

Bulk SMF analysis, read-statistic reports, sorting/cooperativity and nucleosome analysis are disabled in these focused examples. The mBED subworkflow can still generate auxiliary frequency and filtering files.

The workflow writes sorted, bgzip-compressed, tabix-indexed files into `output/ont/mBED/`. The filtered output for these settings is:

```text
SAMPLE_methcalls_llr_2.0_mot_CG-GC_methFiltered_min_0.05_max_0.95.bed.gz
```

### Extract the H5

```bash
samtools faidx input/reference.fa

python data_extraction/extract_mbed.py \
  --bed input/regions.bed --fasta input/reference.fa \
  --mbed output/ont/mBED/SAMPLE_methcalls_llr_2.0_mot_CG-GC_methFiltered_min_0.05_max_0.95.bed.gz \
  --out output/ont/SAMPLE_reads.h5 --workers 4
```

For an existing unsorted/unindexed mBED, prepare a separate indexed copy first:

```bash
bash data_extraction/index_mbed.sh \
  input/calls.mbed.gz output/ont/calls.mbed.gz 4
```

This helper preserves extra columns, drops comment/blank lines, sorts by chromosome/start/end, and uses bgzip plus `tabix -p bed`. It refuses to overwriteexisting output and publishes files only after successful compression/indexing.
The nf-smfont output is already indexed and does not need this extra step.

mBED's first five fields must be chromosome, read start, read end, read ID and a delta string such as `5m3u2x`. Event positions follow the supplied parser: `read_start + 1 + cumulative_delta`, with `m=0`, `u=1`, `x=-1` in the H5.
This is not generic modkit bedMethyl. First-five-event strand inference, CGC/GCG collapsing, duplicate-read-ID overwrite behavior and read ordering follow the original extraction implementation. 

## H5 layout, coordinates and encoding

| Group | Per-region data |
| --- | --- |
| `smf_mat` | Hard calls, `[reads, positions]`, signed int8 |
| `smf_pos` | Positions, 0-based offsets within the region |
| `read_id` | Molecule IDs |

Matrices have reads on the first axis when opened with Python/h5py. H5 compression defaults to gzip level 1 and is
configurable with `--compression`.

Default BED input is **0-based, half-open**. H5 names are **1-based, closed**: `chr1 100 110` becomes `chr1:101-110`. A call at 1-based coordinate 103 is stored at offset 2. T

The extractors defaults to `-1` missing, `0` methylated and `1` unmethylated. 


## Dependencies and validation

The mBED extractor needs Python, numpy, h5py and pysam.
The shell helper needs Bash, GNU sort/awk, gzip, bgzip and tabix. ONT
preprocessing uses Nextflow, Java and Apptainer plus the upstream containers.

Versions used for the local preprocessing/extraction checks (not minimum-version
constraints):

| Package/tool | Version |
| --- | --- |
| Python | 3.12.14 |
| numpy | 2.2.6 |
| h5py | 3.16.0 |
| pysam | 0.23.1 |
| bgzip / tabix (HTSlib) | 1.23.1 |
| Nextflow | 24.04.2 |
| nf-smfont (bundled source) | 1.0.0 `|
