# SMF data extraction

Prepares the `smf_mat`/`smf_pos` H5 files used by SMFNet from Illumina
bisulfite reads or ONT per-molecule methylation calls. There are three H5
extractors: two R entry points for GC+CG and GC-only, and one Python entry
point for mBED. The R entry points share an engine; keep both files together.

Run every command below from `example_scripts/general_example_scripts`
(relative to the repository root). Paths under `input/`
are examples to replace with your own files. The scripts create their output
directories. Region selection, train/test splits and reference preparation are
upstream inputs; these scripts do not choose new training regions.

## Steps

| Input | Preprocessing | H5 extraction |
| --- | --- | --- |
| Paired Illumina FASTQ | `preprocess_illumina_quasr.R` → saved QuasR project | `extract_illumina_gc_cg.R` or `extract_illumina_gc.R` |
| Existing QuasR BAMs | Save a project with the same preprocessing script, or supply an alignment sample table directly | Same two R extractors |
| ONT POD5/FAST5 | `run_nf_smfont.sh raw` → called, aligned reads → indexed mBED | `extract_mbed.py` |
| ONT BAM/CRAM with methylation calls | `run_nf_smfont.sh bam` → indexed mBED | `extract_mbed.py` |
| Existing delta-string mBED | `index_mbed.sh` only if sorting/compression/indexing is needed | `extract_mbed.py` |

State calling is a separate downstream step, documented in the
[parent README](../README.md#state-calling). It accepts both existing training
H5 and the newer indexed samples layout. Extraction here writes the training
layout described below.

## Inputs

| Input | Requirement |
| --- | --- |
| Regions | Three-column BED, 0-based half-open by default; chromosome names must match the reference |
| Illumina reads | Paired bisulfite FASTQ, existing compatible BAMs, or a saved QuasR project |
| R reference | The BSgenome package matching those reads and regions; default is mm10 |
| ONT reads | Raw POD5/FAST5 plus a suitable custom methylation model, or BAM/CRAM already containing the required MM/ML calls |
| ONT reference | Matching FASTA; `extract_mbed.py` requires its `.fai` index |
| mBED | BGZF-compressed, tabix-indexed per-molecule delta strings from nf-smfont's custom mBED route |

An ordinary alignment BAM without methylation calls is not sufficient for the
`bam` route. The raw route requires the appropriate pore/basecalling and Remora
models. The custom model used in the historical runs is not bundled here.

## Illumina: QuasR preprocessing

Create a tab-separated `input/samples.tsv`:

```text
FileName1	FileName2	SampleName
fastq/sample_1.fastq.gz	fastq/sample_2.fastq.gz	SAMPLE
```

Paths in this table are relative to the table's directory. For existing BAMs,
use columns `FileName` and `SampleName` instead. Distinct input files may share
a sample name; repeated input-file rows are rejected.

```bash
Rscript data_extraction/preprocess_illumina_quasr.R \
  --sample-file input/samples.tsv \
  --genome BSgenome.Mmusculus.UCSC.mm10 \
  --out-dir output/illumina/quasr --workers 8
```

Alternatively, replace `--sample-file input/samples.tsv` with
`--fastq-dir input/fastq` to pair `*_1.fastq.gz` with `*_2.fastq.gz`.
Missing mates fail before alignment starts.

The cleaned script retains the original Kreibich alignment settings:
`aligner=Rbowtie`, `paired=fr`, `bisulfite=undir`, and
`alignmentParameter="-e 70 -X 1000 -k 2 --best -strata"`. These are explicit
CLI options where applicable. Worker count is configurable, and the cluster
is stopped on exit. Use one mutually compatible R/Bioconductor environment;
`--lib-paths DIR1:DIR2` can prepend additional R libraries.

Outputs are `project.rds`, normalized `samples.tsv`, `alignments.tsv`,
`alignments/`, `cache/`, `package_versions.tsv` and `sessionInfo.txt`.
The saved project refers to its alignment files; retain those files when using
it later. An existing `project.rds` requires `--overwrite true` to replace.

### Extract GC+CG or GC-only

```bash
Rscript data_extraction/extract_illumina_gc_cg.R \
  --project-rds output/illumina/quasr/project.rds --sample SAMPLE \
  --bed input/regions.bed --genome BSgenome.Mmusculus.UCSC.mm10 \
  --out output/illumina/SAMPLE_gc_cg.h5 --workers 8

Rscript data_extraction/extract_illumina_gc.R \
  --project-rds output/illumina/quasr/project.rds --sample SAMPLE \
  --bed input/regions.bed --genome BSgenome.Mmusculus.UCSC.mm10 \
  --out output/illumina/SAMPLE_gc.h5 --workers 8
```

These are alternative outputs, not sequential processing stages. Extract one
sample per H5. Both can instead take `--sample-file input/aligned_samples.tsv`.

| Default | GC+CG | GC-only |
| --- | --- | --- |
| Coverage threshold | 1 | 50 |
| Minimum GC sites | 15 | 15 |
| GC context | `DGCHN` | `GC` |
| CG context | `NWCGW` | Not extracted |
| Bulk coverage calculation | From the extracted reads | From the extracted reads |

For a strict NOMe GC-only comparison, set `--gc-context DGCHN` and match the
coverage threshold explicitly. `--bulk-mode qmeth` restores the separate
QuasR bulk scan; the default `reads` mode avoids that additional scan. Contexts,
coverage and minimum sites are configurable. Empty or insufficient-coverage
channels are skipped.

## ONT: nf-smfont preprocessing

The complete, unmodified [nf-SMFONT v1.0.0 Zenodo release](https://doi.org/10.5281/zenodo.21919078)
is included under `vendor/nf-smfont`, with its MIT license and upstream docs.
The launcher uses this local copy, so it does not fetch the workflow from
GitLab. Nextflow plugins, institutional configuration and containers may still
require network access. The ZIP checksum and release metadata are recorded in
[provenance/nf_smfont_release.json](provenance/nf_smfont_release.json).

Activate Nextflow and Apptainer before running. The default profile is
`apptainer,multi_gpus`, matching the recovered launch commands. This profile
does **not** select a Slurm partition: upstream defaults to the local executor.
Run inside an appropriate allocation, or supply your site's executor/resource
settings using `NF_CONFIG=path/to/cluster.config`. `NF_PROFILE` overrides the
profiles. Raw basecalling needs suitable GPU resources.

### From already-called BAMs

```bash
bash data_extraction/run_nf_smfont.sh bam \
  --input_folder input/ont_bams --sample_name SAMPLE \
  --ref input/reference.fa --outdir output/ont
```

The directory should contain that sample's BAMs directly. For CRAM input add
`--input_ext cram`. The workflow inspects input status and retains the mapping
stage while disabling basecalling and methylation calling for this route.

### From raw signal

```bash
bash data_extraction/run_nf_smfont.sh raw \
  --input_folder input/pod5 --sample_name SAMPLE \
  --ref input/reference.fa --outdir output/ont \
  --remora_model_path input/models/custom_CG_GC_model.pt
```

For FAST5 add `--input_ext fast5`. The raw template uses Bonito with
`dna_r10.4.1_e8.2_400bps_sup@v4.2.0` and a custom Remora model, as recorded in
the original runs. These settings must match the actual chemistry and model.
The bundled release has newer container versions than the historical runs;
this is a portable example, not a claim of bit-identical historical basecalls.
Add `-resume` to resume a run with its work directory and Nextflow cache intact.
Insert `--dry-run` immediately after `bam` or `raw` to print the command only.

Both templates use these explicit mBED settings:

| Parameter | Value |
| --- | --- |
| Read quality threshold | 7 |
| Methylation LLR threshold | 2.0 |
| Motifs | `cpg,gpc` |
| Custom mBED route | `modkit=false` |
| Filter by read methylation rate | Enabled |
| Minimum / maximum methylation rate | 0.075 / 0.95 |

Bulk SMF analysis, read-statistic reports, sorting/cooperativity and nucleosome
analysis are disabled in these focused examples. The mBED subworkflow can
still generate auxiliary frequency and filtering files. These settings differ
from a complete historical analysis run; mBED thresholds above are retained.

The workflow writes sorted, bgzip-compressed, tabix-indexed files into
`output/ont/mBED/`. The filtered output for these settings is:

```text
SAMPLE_methcalls_llr_2.0_mot_CG-GC_methFiltered_min_0.075_max_0.95.bed.gz
```

### Extract the H5

```bash
samtools faidx input/reference.fa

python data_extraction/extract_mbed.py \
  --bed input/regions.bed --fasta input/reference.fa \
  --mbed output/ont/mBED/SAMPLE_methcalls_llr_2.0_mot_CG-GC_methFiltered_min_0.075_max_0.95.bed.gz \
  --out output/ont/SAMPLE_reads.h5 --workers 4
```

For an existing unsorted/unindexed mBED, prepare a separate indexed copy first:

```bash
bash data_extraction/index_mbed.sh \
  input/calls.mbed.gz output/ont/calls.mbed.gz 4
```

This helper preserves extra columns, drops comment/blank lines, sorts by
chromosome/start/end, and uses bgzip plus `tabix -p bed`. It refuses to overwrite
existing output and publishes files only after successful compression/indexing.
The nf-smfont output is already indexed and does not need this extra step.

mBED's first five fields must be chromosome, read start, read end, read ID and
a delta string such as `5m3u2x`. Event positions follow the supplied parser:
`read_start + 1 + cumulative_delta`, with `m=0`, `u=1`, `x=-1` in the H5.
This is not generic modkit bedMethyl. First-five-event strand inference,
CGC/GCG collapsing, duplicate-read-ID overwrite behavior and read ordering
follow the original extraction implementation. Unlike the R extractors,
mBED extraction retains empty regions.

## H5 layout, coordinates and encoding

| Group | Per-region data |
| --- | --- |
| `smf_mat` | Hard calls, `[reads, positions]`, signed int8 |
| `smf_pos` | Positions, 0-based offsets within the region |
| `read_id` | Molecule IDs |
| `methyl_mat`, `methyl_pos`, `methyl_read_id` | Corresponding CG channel, only for GC+CG extraction |

For R extraction, `smf_mat` is the GC channel and `methyl_mat` is the CG
channel. These have independent rows and IDs: do not pair molecules by row
number. The mBED extractor preserves the original combined CG/GC anchoring
in `smf_mat`; it does not create a separate CG channel or perform a GC-only
context filter. Apply the appropriate assay-site selection downstream. Matrices have reads on the first axis when opened with Python/h5py,
including files written by R. H5 compression defaults to gzip level 1 and is
configurable with `--compression`.

Default BED input is **0-based, half-open**. H5 names are **1-based, closed**:
`chr1 100 110` becomes `chr1:101-110`. A call at 1-based coordinate 103 is
stored at offset 2. Use `--bed-coordinates one-based` only to reproduce inputs
that already used the original ACT script's 1-based origin. These naming
rules are specific to these extractors and are not the exclusive-end naming
convention in the separate Fiber-seq example.

All three extractors default to `-1` missing, `0` methylated and `1`
unmethylated. The GC accessibility label is therefore `0`; endogenous CG
methylation should not be interpreted as the same accessibility assay.

| Raw QuasR | Standard | `legacy-gc-cg` | `legacy-gc` |
| --- | --- | --- | --- |
| 0, missing | -1 | -1 | 0 |
| 1, unmethylated | 1 | -1 | 1 |
| 2, methylated | 0 | 0 | -1 |

The explicit `--encoding legacy-gc-cg` and `--encoding legacy-gc` modes preserve
the inconsistent conversions in the original examples. They are for exact
legacy reproduction, not recommended encodings for new training data.

## Recovered run provenance

Exact successful commands were found in `.nextflow/history` for
`R1_haplo1`, `R1_haplo2`, `R2_haplo1` and `R2_haplo2` under
`ont_results/20260211_F1_Spret/20260409_fixed_mbed`. For example:

```text
nextflow run /g/krebs/boulanger/Scripts/dev/nf-smfont/main.nf -c 20260409_R1_haplo1.config -profile apptainer,multi_gpus
```

This is historical evidence, not a portable command to copy into the examples.
All four commands are preserved in
[provenance/recovered_commands.tsv](provenance/recovered_commands.tsv).
[provenance/historical_runs.json](provenance/historical_runs.json) records the
original input folders, reference/model paths, selected parameters and source
files. Whole-sample R1 and R2 parameter JSONs were recovered, but their exact
launch commands were not. The latest R2 whole-sample parameters use a minimum
read methylation rate of **0.10**, whereas these haplotype templates use
**0.075**. Do not treat all runs as having identical filtering.

Historical successful runs report **nf-smfont v0.9.0 and Nextflow 24.04.2**.
The bundled Zenodo workflow is **v1.0.0**. Historical container tags include
Bonito `0.7.3custom` and Dorado `1.3.0`; the release defaults use Bonito `0.8.1`
and Dorado `1.4.0`. Both specify `smfont_utils:1.0.2`. The custom CG/GC Remora
model and the nondefault read filters are part of the original setup.
Recorded workflow software versions are preserved in
[provenance/historical_nf_software_versions.yml](provenance/historical_nf_software_versions.yml).

## Dependencies and validation

The Python extractor needs Python, numpy, h5py and pysam. R extraction requires
SingleMoleculeFootprinting, QuasR, Matrix, GenomicRanges, Biostrings, rhdf5 and
the chosen BSgenome package; QuasR preprocessing additionally uses Rbowtie.
The shell helper needs Bash, GNU sort/awk, gzip, bgzip and tabix. ONT
preprocessing uses Nextflow, Java and Apptainer plus the upstream containers.

Versions used for the local preprocessing/extraction checks (not minimum-version
constraints):

| Package/tool | Version |
| --- | --- |
| R | 4.4.1 |
| QuasR | 1.46.0 |
| Rbowtie | 1.46.0 |
| SingleMoleculeFootprinting | 2.2.0 |
| BSgenome | 1.72.0 |
| BSgenome.Mmusculus.UCSC.mm10 | 1.4.3 |
| Matrix | 1.7.0 |
| GenomicRanges | 1.56.1 |
| Biostrings | 2.72.1 |
| rhdf5 | 2.48.0 |
| Python | 3.12.14 |
| numpy | 2.2.6 |
| h5py | 3.16.0 |
| pysam | 0.23.1 |
| bgzip / tabix (HTSlib) | 1.23.1 |
| Nextflow | 24.04.2 |
| nf-smfont (bundled source) | 1.0.0 |

R's `parallel` package is included with R. Transitive dependencies are installed
by the R/Bioconductor package manager. Runtime `package_versions.tsv` and
`sessionInfo.txt` record the actual environment used for each QuasR project.
Container dependency versions are defined by the upstream workflow and are
separate from this host-side table.

Version and validation details are recorded in
[../validation/preprocessing_validation.txt](../validation/preprocessing_validation.txt).
Existing extraction/state comparisons are described in the parent README.
The source checksums of those four scripts are unchanged. All 182 bundled
workflow files match the Zenodo ZIP byte-for-byte, including its license.
Full FASTQ alignment and ONT signal/basecalling workflows were not rerun as
part of preparing this package.
