# Short-read SMF data extraction

Replace the example paths, sample name and reference genome with your own inputs. This data extraction relies on [QuasR](https://bioconductor.org/packages/release/bioc/html/QuasR.html) for sequence alignment and the [SingleMoleculeFootprinting](https://bioconductor.org/packages/release/bioc/html/SingleMoleculeFootprinting.html) bioconductor package for data extraction. Preprocessing follows the [SingleMoleculeFootprinting workflow](https://github.com/Krebslabrep/SingleMoleculeFootprinting).

## Choose the assay mode

| Entry point | SMF output | Endogenous methylation output | Data source | Cell type |
| --- | --- | --- | --- | --- |
| `extract_illumina_gc_cg.R` | GC **and** CG calls combined in `smf_mat`, `smf_pos`, `read_id` | None | [Sönmezer et al., Molecular Cell 2021](https://doi.org/10.1016/j.molcel.2020.11.015) | mESC |
| `extract_illumina_gc.R` | GC calls in `smf_mat`, `smf_pos`, `read_id` | CG calls in `methyl_mat`, `methyl_pos`, `methyl_read_id` | [Kreibich et al., Molecular Cell 2023](10.1016/j.molcel.2023.01.017) | mESC, NPC, C2CL12, MEL |

Combined mode is for assays where both GC and CG methylation report SMF. GC-only mode is for assays where GC reports SMF and CG methylation is endogenous. "GC-only" describes the SMF channel, not the complete H5.
The scripts do not infer the assay mode from the sample name.

Both scripts extract and filter both contexts. Combined mode joins calls by molecule ID and sorts their union of positions. Missing entries stay missing. A position shared by both contexts appears once. Conflicting observed calls at the same molecule/position cause an error rather than an arbitrary overwrite. The minimum-site threshold is applied after combining the SMF calls.
In GC-only mode, the two channels have their own position arrays and molecule ID. Match IDs rather than assuming row order is identical.

Keep the two R entry points together because they share the extraction engine. These commands extract raw methylation calls. They do not call chromatin states.

## Dependencies

Use a compatible R/Bioconductor installation with `SingleMoleculeFootprinting`,
`QuasR`, `Rbowtie`, `Matrix`, `GenomicRanges`, `Biostrings`, `rhdf5`, and the
BSgenome package matching the alignments. Every entry point supports `--help`.

## Prepare a QuasR project

Create a tab-separated `input/samples.tsv`:

```text
FileName1	FileName2	SampleName
fastq/sample_1.fastq.gz	fastq/sample_2.fastq.gz	SAMPLE
```

Paths in this table are relative to the table's directory. For existing BAMs, use columns `FileName` and `SampleName` instead. Distinct input files may share a sample name. Repeated input-file rows are rejected.

```bash
Rscript preprocess_illumina_quasr.R \
  --sample-file input/samples.tsv \
  --genome BSgenome.Mmusculus.UCSC.mm10 \
  --out-dir output/illumina/quasr --workers 8
```

Alternatively, replace `--sample-file input/samples.tsv` with `--fastq-dir input/fastq` to pair `*_1.fastq.gz` with `*_2.fastq.gz`.
Missing mates fail before alignment starts.

The default alignment settings are: 
`aligner=Rbowtie`, `paired=fr`, `bisulfite=undir`, and
`alignmentParameter="-e 70 -X 1000 -k 2 --best -strata"`. These are explicit CLI options where applicable. Worker count is configurable, and the cluster is stopped on exit. Use one mutually compatible R/Bioconductor environment.
`--lib-paths DIR1:DIR2` can prepend additional R libraries.

Outputs are `project.rds`, normalized `samples.tsv`, `alignments.tsv`, `alignments/`, `cache/`, `package_versions.tsv` and `sessionInfo.txt`. The saved project refers to its alignment files. Retain those files when using it later. An existing `project.rds` requires `--overwrite true` to replace.

### Extract GC+CG or GC-only

```bash
Rscript extract_illumina_gc_cg.R \
  --project-rds output/illumina/quasr/project.rds --sample SAMPLE \
  --bed input/regions.bed --genome BSgenome.Mmusculus.UCSC.mm10 \
  --out output/illumina/SAMPLE_gc_cg.h5 --workers 8

Rscript extract_illumina_gc.R \
  --project-rds output/illumina/quasr/project.rds --sample SAMPLE \
  --bed input/regions.bed --genome BSgenome.Mmusculus.UCSC.mm10 \
  --out output/illumina/SAMPLE_gc.h5 --workers 8
```

These are alternative outputs, not sequential processing stages. Extract one sample per H5. Both can instead take `--sample-file input/aligned_samples.tsv`.

| Default | GC+CG | GC-only |
| --- | --- | --- |
| Coverage threshold | 1 | 1 |
| Minimum SMF sites (`--min-sites`) | 15 across GC + CG | 15 GC |
| GC context | `GC` | `DGCHN` |
| CG context | `CG` | `HCGD` |
| Bulk coverage calculation | From the extracted reads | From the extracted reads |

Contexts, coverage and minimum sites are configurable. Empty or insufficient-coverage channels are skipped.

## H5 layout and encoding

Each group contains one dataset per region. Matrices opened with Python/h5py have shape **reads × positions** and signed 8-bit values. Position arrays contain 0-based offsets from the first region base. Compression defaults to gzip level 1.
The root attributes record `assay_mode` and `accessibility_contexts`.

| QuasR value | Meaning | Standard H5 value |
| --- | --- | --- |
| 0 or NA | Missing | -1 |
| 1 | Unmethylated | 1 |
| 2 | Methylated | 0 |

In `smf_mat`, methylated calls report SMF signal. In GC-only mode, `methyl_mat` uses the same numerical encoding but represents **endogenous CG methylation**, not SMF signal. The endogenous methylation channel requires at least one retained site and is written independently of the SMF channel's minimum-site threshold.

## Coordinates and output handling

BED input defaults to **0-based, half-open** intervals. H5 region names use **1-based, closed** coordinates: BED `chr1 100 110` becomes `chr1:101-110`.
A call at 1-based position 103 has relative position 2. Use `--bed-coordinates one-based` for input intervals already expressed as
1-based, closed coordinates.

Only the parent process writes H5/ workers extract bounded batches of regions. Use `--workers 1` on non-Unix systems. Output is written to a temporary sibling file and renamed on success. Existing output requires `--overwrite true`.

