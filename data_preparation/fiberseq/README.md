# Fiber-seq data preparation

Turns released Fiber-seq BAMs into the `smf_mat`/`smf_pos` H5 files SMFNet trains on.
Fiber-seq marks accessible DNA with m6A (Hia5 methyltransferase) and reads whole
molecules with PacBio HiFi, so every molecule reports both strands: m6A on top-strand
adenines appears at reference A bases, and on bottom-strand adenines at reference T bases.

The examples use **GM12878** (Vollger et al.; see Citations), which has deep read-level
coverage, haplotype tags and extensive ENCODE data.

## Steps

| Step | Script |
| --- | --- |
| 1. Select regions: cCRE windows split into active and background by ATAC | `select_regions.py` |
| 2. Extract per-molecule m6A calls from the BAM over those regions | `fiberbam_to_calls.py` (+ `.sbatch`) |
| 3. Crop molecules to fixed windows and write the training H5 | `calls_to_h5.py` |

The region BED only decides which molecules step 2 keeps, so running all three steps
reproduces the training data from the released BAM alone.

Run the scripts in the repository's pixi environment, which provides `samtools` and
`zstd` (`pixi shell`, or prefix each command with `pixi run`).

## Inputs

| Input | Source |
| --- | --- |
| GM12878 FIRE BAM (326 GB) | `s3://stergachis-manuscript-data/2024/Vollger_et_al/FIRE/` |
| Reference | UCSC hg38 |
| cCRE registry, GM12878 classification | ENCODE SCREEN v3, `ENCFF733BFV` |
| ATAC peaks, IDR thresholded | ENCODE `ENCSR095QNB`, `ENCFF614SMH` |
| ATAC peaks, pseudoreplicated (relaxed) | ENCODE `ENCSR095QNB`, `ENCFF158ORB` |
| Blacklist | ENCODE hg38 v2 ([Boyle-Lab/Blacklist](https://github.com/Boyle-Lab/Blacklist)) |

## Step 1: `select_regions.py`

The design follows the mouse ONT data, where regions are the Sönmezer et al. 2021 bait
library split by overlap with ATAC peaks. The human equivalent of the bait library is the
ENCODE cCRE registry (1.06M elements, active in at least one ENCODE biosample).

1. Put a 2114 bp window on the centre of every registry cCRE. Drop windows that overlap the
   blacklist or are more than 1% N.
2. Thin overlapping windows so centres are at least 1057 bp apart (at most 50% overlap).
3. **Active**: the window overlaps an IDR ATAC peak. **Background**: it overlaps no ATAC
   peak, relaxed calls included. Windows touching only a relaxed peak are dropped.
4. Write one BED of candidates with their class and GC. Step 2 extracts molecules over all
   of them; step 3 applies the depth filter and then downsamples the background to 1:1 with
   the active set, matched on GC in 1% bins (optional, `--no-gc-match`). Bins short of
   background keep what they have.

```bash
python select_regions.py \
    --ccre ENCFF733BFV_GM12878_cCRE_v3.bed.gz \
    --active-peaks ENCFF614SMH.bed.gz \
    --exclude-peaks ENCFF158ORB.bed.gz \
    --blacklist hg38-blacklist.v2.bed.gz \
    --fasta hg38.fa \
    --out GM12878/candidates.bed
```

Output columns: `chrom start end name class gc ccre_id ccre_class`, 0-based half-open, sorted
chr1-22 then chrX. `name` is the window's H5 key (see step 3). It takes about 80 s.

For GM12878: 1,062,844 cCREs → 1,058,400 windows after the blacklist, N and
chromosome-edge filters → 583,549 after thinning → **84,241 active and 455,183 background**,
with 44,125 dropped as ambiguous. That is before any coverage filter.

## Step 2: `fiberbam_to_calls.py`

```bash
python fiberbam_to_calls.py --bam GM12878.fire.bam --regions GM12878/candidates.bed \
    --fasta hg38.fa --chrom chr21 --out GM12878/calls/chr21.tsv.zst

# All chromosomes on Slurm, one array task each; submit from this directory and add
# your cluster's partition or account options.
BAM=... REGIONS=... FASTA=... OUT=... sbatch -o "$OUT/logs/%A_%a.log" fiberbam_to_calls.sbatch
```

The Slurm script runs in the pixi environment through `pixi run`, so `pixi` must be on
`PATH` in the job; it has a commented line for loading it as a module.

### Output format

A zstd-compressed TSV, `<chrom>.tsv.zst`, one file per chromosome and one row per molecule.
Lines starting with `#` are comments; the first two give the column names and the
parameters used.

Compression is `zstd -12 --long=27` through the `zstd` command-line tool (`calls_io.py`).
Long mode is what makes it pay off: overlapping molecules repeat the no-call pattern that
the reference fixes at C/G bases, and on GM12878 chr21 it compresses 15× (725 MB to 48 MB)
in 3.5 s, against 6× for gzip or zstd without it. Files must be read with the same flag:

```bash
zstd -dc --long=27 chr21.tsv.zst | head
```

Output goes to `<out>.tmp` and is renamed only when both samtools and zstd succeed, so a
failed or preempted task never leaves a truncated file under the final name.

| # | Column | Meaning |
| --- | --- | --- |
| 1 | `chrom` | |
| 2 | `start` | 0-based reference coordinate of `calls[0]` |
| 3 | `end` | exclusive; `end - start == len(calls)` |
| 4 | `read_id` | BAM `QNAME` |
| 5 | `strand` | alignment strand, `+` / `-` |
| 6 | `haplotype` | `HP` tag (`1` / `2`), `.` if untagged |
| 7 | `calls` | one character per reference base: `0` no call, `1` accessible (m6A), `2` protected |

Rows span the molecule's full aligned extent. A molecule is kept if it overlaps at least one
region by `--min-span` bp (default 500), and written once even when it overlaps several.

### Filters

- MAPQ ≥ 20. Unmapped, secondary, supplementary, duplicate and QC-fail alignments are
  excluded (`-F 0xF04`).
- A base is called m6A when its `ML` probability is at least 230/255 (about 0.9).
- A reference position is **assayable** only if the molecule aligns a base there, the base
  matches the reference, and both are A or T. Deletions, reference skips, mismatches and soft
  clips are no-calls (`0`), so a molecule carrying the non-A/T allele of a heterozygous SNP
  never gets a "protected" call it could not have made.
- `MM` segments in explicit mode (`?`) make unlisted bases no-calls instead of protected.

## Step 3: `calls_to_h5.py`

```bash
python calls_to_h5.py --calls-dir GM12878/calls --candidates GM12878/candidates.bed \
    --fasta hg38.fa --out-dir GM12878/h5 --workers 8
```

Writes two files, one for active and one for background windows, laid out like the mouse
ONT files, so training takes both as `h5_path: [active.h5, background.h5]`. For each window,
every overlapping molecule is cropped to it and stored as:

| Group | Per window | Notes |
| --- | --- | --- |
| `smf_mat` | `[molecules, positions]` int8 | `1` unmethylated (protected), `0` methylated (m6A, accessible), `-1` no call (TSV `2`, `1`, `0`) |
| `smf_pos` | `[positions]` | window offsets of A/T columns with at least one call |
| `read_id` | `[molecules]` str | |
| `strand`, `haplotype` | `[molecules]` | extra to the ONT layout; the loader ignores them |

A molecule is stored in a window if it overlaps it by at least 500 bp (`--min-overlap`). A
window is kept if at least 32 molecules span all of it (`--min-molecules`). Only then is the
background GC-matched and downsampled, so the 1:1 ratio holds for the windows actually
written. `manifest.tsv` lists every candidate window with its class, GC, molecule count,
full-span count, number of A/T columns, and whether it was kept (`kept`, `low_depth`,
`not_selected`).

Each chromosome is cropped in its own worker, into a temporary file; the kept windows are
then copied into `active.h5.tmp` / `background.h5.tmp`, which are renamed only once complete.

For GM12878 (16.5 min on 8 CPUs, 1.6 GB peak memory):

| | Windows | Size |
| --- | --- | --- |
| Active | 84,125 kept, 116 low depth | 2.9 GB |
| Background | 83,343 kept (99.1% of a 1:1 match), 371,250 not selected, 590 low depth | 2.8 GB |

On chr21 the two classes have similar depth (median 132 and 129 stored molecules, about 1,100
A/T columns per window) and GC (0.490 and 0.486), and the active windows a higher m6A rate
(0.137 against 0.110).

**Encoding.** `smf_mat` uses the mouse ONT files' encoding, `1` unmethylated (protected)
and `0` methylated (m6A, accessible), which is the reverse of the calls TSVs. Each file
records this in its `smf_encoding` attribute.

**Window names.** A 0-based half-open window `[a, a + 2114)` is named `chr:{a+1}-{a+2115}`,
a 1-based start with an exclusive end, as in the ONT files. `smf_net.data.load_dna` reads
it given the window width (`h5_region_length` in training). The converter loads each
window this way and checks that every observed call is on a reference A or T. The assay
filter in `smf_net.data` drops ineligible columns silently, and A/T runs mean a coordinate
off by one still lands on A/T about 60% of the time, so an off-by-one would otherwise go
unnoticed.

## Training notes

Train with `assay_type="fiberseq_m6a"` (`src/smf_net/data.py`):

- **Valid sites** are A and T bases, about 58% of positions, against a few percent for GCH.
  Expect about 10× more stored calls per region than GCH.
- **Reverse complement** flips the calls and reverse-complements the DNA, with no shift: the
  m6A sits on a single base pair, unlike a CpG/GpC call, which moves to the other cytosine of
  its dinucleotide. `reverse_complement_dna_and_smf` takes the assay type for this.
- **RC augmentation defaults to on**, because GM12878's m6A rates at A sites and at T sites
  in the reverse-complement context agree to within split-half noise.
  `SparseDataModuleConfig.rc_augmentation` can override it per dataset.
- **Flank baseline** (`smf_net.evaluation.flank_baselines`) fits separate A and T site
  models, since read on the plus strand a T site's context runs the other way.
- **Human chromosome splits** need setting in the training config. The defaults are
  mouse-based and put chr20-22 in no split.

## Citations

Data:

- **Fiber-seq samples** (GM12878 and others) come from one bioRxiv preprint, doi
  [10.1101/2024.06.14.599122](https://doi.org/10.1101/2024.06.14.599122): Vollger MR,
  Swanson EG, Neph SJ, Mallory BJ, Dubocanin D, …, Ray JP, Stergachis AB. *Somatic
  epimutations cap genetic determinism in the human diploid chromatin epigenome.* (v3,
  2026; earlier versions were titled *A haplotype-resolved view of human gene
  regulation*.) Its data are listed on the
  [Fiber-seq data portal](https://stergachislab.github.io/Fiber-seq-publication-data/), with
  processed FIRE results on [Zenodo 14511246](https://doi.org/10.5281/zenodo.14511246) and
  raw GM12878 reads under BioProject `PRJNA1233341`. The BAMs used here are in
  `s3://stergachis-manuscript-data/2024/Vollger_et_al/FIRE/`.
- **ENCODE inputs** (step 1), cited with the ENCODE 2020 paper below:
  - cCREs: `ENCFF733BFV`, from annotation `ENCSR450FYE` (GM12878, "ENCODE v3"). This is
    Registry of cCREs **V3** (about 1.06M human elements), a later release than the V2
    registry the 2020 paper describes, with no paper of its own; name the version and link
    the [SCREEN versions page](https://screen.encodeproject.org/index/cversions).
  - ATAC peaks: experiment `ENCSR095QNB`, ENCODE pipeline outputs `ENCFF614SMH` (IDR
    thresholded) and `ENCFF158ORB` (pseudoreplicated).

Methods and resources:

- **Fiber-seq**: Stergachis AB, Debo BM, Haugen E, Churchman LS, Stamatoyannopoulos JA.
  *Single-molecule regulatory architectures captured by chromatin fiber sequencing.*
  Science 368(6498):1449–1454 (2020). doi
  [10.1126/science.aaz1646](https://doi.org/10.1126/science.aaz1646).
- **m6A calling (fibertools)**: Jha A, Bohaczuk SC, Mao Y, …, Stergachis AB, Vollger MR.
  *DNA-m6A calling and integrated long-read epigenetic and genetic analysis with fibertools.*
  Genome Research 34(11):1976–1986 (2024). doi
  [10.1101/gr.279095.124](https://doi.org/10.1101/gr.279095.124). The released BAMs carry
  its `MM`/`ML` calls.
- **cCRE registry**: ENCODE Project Consortium, Moore JE, Purcaro MJ, …, Weng Z. *Expanded
  encyclopaedias of DNA elements in the human and mouse genomes.* Nature 583(7818):699–710
  (2020). doi [10.1038/s41586-020-2493-4](https://doi.org/10.1038/s41586-020-2493-4).
  SCREEN's requested citation for the registry; also covers the ENCODE ATAC data.
- **IDR peaks**: Li Q, Brown JB, Huang H, Bickel PJ. *Measuring reproducibility of
  high-throughput experiments.* Annals of Applied Statistics 5(3):1752–1779 (2011). doi
  [10.1214/11-AOAS466](https://doi.org/10.1214/11-AOAS466). Defines the IDR threshold behind
  the active set.
- **Blacklist**: Amemiya HM, Kundaje A, Boyle AP. *The ENCODE Blacklist: identification of
  problematic regions of the genome.* Scientific Reports 9:9354 (2019). doi
  [10.1038/s41598-019-45839-z](https://doi.org/10.1038/s41598-019-45839-z).
- **Chromosome split**: ChromBPNet fold 0. Pampari A, Shcherbina A, …, Kundaje A.
  *ChromBPNet: bias factorized, base-resolution deep learning models of chromatin
  accessibility reveal cis-regulatory sequence syntax, transcription factor footprints and
  regulatory variants.* bioRxiv (2024). doi
  [10.1101/2024.12.25.630221](https://doi.org/10.1101/2024.12.25.630221).
- **Mouse region design mirrored here**: Sönmezer C, Kleinendorst R, …, Krebs AR. *Molecular
  co-occupancy identifies transcription factor binding cooperativity in vivo.* Molecular
  Cell 81(2):255–267.e6 (2021). doi
  [10.1016/j.molcel.2020.11.015](https://doi.org/10.1016/j.molcel.2020.11.015). It designed
  the bait library (about 297k mouse CREs, `SingleMoleculeFootprintingData::EnrichmentRegions_mm10`).

## Tests

`tests/test_fiberbam_to_calls.py` builds a small BAM from SAM text and checks
hand-derived rows: forward and reverse strands, a soft clip, a deletion over an A, both
directions of SNP, the 5mC segment being skipped, explicit-mode `MM`, and each excluded
flag, plus atomic output on failure. `tests/test_calls_to_h5.py` and
`tests/test_select_regions.py` cover steps 1 and 3. They need samtools and zstd, which the
pixi environment provides, and skip without them:

```bash
pixi run pytest tests/test_fiberbam_to_calls.py tests/test_calls_to_h5.py tests/test_select_regions.py
```
