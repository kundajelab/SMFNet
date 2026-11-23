# SMFNet

SMFNet is a discrete diffusion model that predicts the distribution of single-molecule
footprinting (SMF) reads given DNA sequence as input. From 2048 bp of DNA, it generates
individual reads over the central 1024 bp, each a string of per-site calls: 1 where the
molecule was occupied (protected from the methylase) and 0 where it was accessible.
Sampling many reads gives the range of molecular states a sequence encodes, not only its
average accessibility.

Models can be trained on three assays (`assay_type`; see [Assay types](#assay-types)):

| Assay type | Assay | Call sites |
| --- | --- | --- |
| `cg_gc_accessibility` | Nanopore SMF with CpG and GpC methylases | CpG and GpC |
| `gch_accessibility` | GpC methylase SMF | GpC in a DGCH context (no CpG overlap) |
| `fiberseq_m6a` | Fiber-seq (m6A) | A and T |

The repository also implements the paper's analyses of trained models: metrics comparing
generated with observed reads, motif marginalization, in-silico mutagenesis of predicted
profiles (ISM), and per-read in-silico mutagenesis (PRISM), which explains the model's
likelihood of a single observed read.

A preprint describing SMFNet in more detail is available at [biorxiv](https://www.biorxiv.org/content/10.64898/2026.10.30.762954v1) (see [Citation](#citation)).

## Installation

### System requirements

- **Operating system:** Linux x86-64 with glibc 2.17 or newer for the GPU environment;
  Linux or Apple Silicon macOS for the CPU environment.
- **Software:** Python 3.12, PyTorch 2.5.1, PyTorch Lightning 2.6.0 and the other
  dependencies in `pyproject.toml`, all installed by pixi from `pixi.lock`. The GPU
  environment needs an NVIDIA driver supporting CUDA 12.
- **Hardware:** training and sampling need an NVIDIA GPU. With the default settings,
  training needs about 64 GiB of GPU memory and sampling about 25 GiB; smaller batches fit
  GPUs with 24 GB or more (see [Demo](#demo)), though the released models were trained at
  the default batch size. Training uses about 12 GB of RAM (with the default 4 data-loading
  workers; each adds about 1 GB) and about one CPU core, so 16 GB of RAM and 2–4 cores are
  enough.
- **Disk:** about 10 GB for the GPU environment, 44 MB for the [demo](#demo), and 3.6 GB
  for the ONT mESC training data plus 2.7 GB for the mm10 genome.
- **Tested configuration:** the run times and resource use in this README were measured on
  nodes of Stanford's Sherlock cluster, running CentOS Linux 7.9 (kernel 3.10) with NVIDIA
  driver 550 (CUDA 12.4), with one NVIDIA H100 80GB SXM5 GPU and 8 cores of an Intel Xeon
  Platinum 8462Y+ (2.8 GHz) allocated to each job.

### Install

SMFNet's environment is managed with [pixi](https://pixi.sh), which installs Python,
PyTorch, the CUDA libraries and every other dependency from the lockfile `pixi.lock`. No
system CUDA toolkit or compilers are needed.

```bash
git clone https://github.com/kundajelab/SMFNet.git
cd SMFNet
pixi install            # GPU environment (Linux with an NVIDIA driver for CUDA 12)
pixi install -e cpu     # CPU-only environment (Linux or Apple Silicon macOS)
```

`pixi install` of the GPU environment took about 3 minutes, downloads included, on a Linux
server on Stanford's network; it takes longer on slower connections.

Run commands in the environment with `pixi run` (add `-e cpu` for the CPU environment), or
open a shell in it with `pixi shell`:

```bash
pixi run python -c "import torch; print(torch.cuda.is_available())"
pixi run -e cpu test
```

The `smf_net` package is installed in editable mode, so changes under `src/` take effect
without reinstalling. Training, sampling and PRISM need a GPU in practice; the CPU
environment is enough to load models, run the tests and explore saved results.

To install the GPU environment on a machine without an NVIDIA driver, such as a cluster
login node, set `CONDA_OVERRIDE_CUDA=12` for `pixi install`.

### Clusters with older operating systems

The lockfile is solved for glibc 2.17, so it also works on clusters running older Linux
distributions (e.g. CentOS 7), where many current PyPI wheels cannot be installed. The GPU
environment takes about 10 GB. If the repository lives on a filesystem with a small quota,
put the environments and the download cache elsewhere:

```bash
pixi config set --local detached-environments /path/with/space/pixi-envs
export PIXI_CACHE_DIR=/path/with/space/pixi-cache
```

### Without pixi

`pip install -e .` installs SMFNet and its Python dependencies from PyPI, with the optional
extras `plotting`, `training` and `notebook`. It needs a platform with wheels for every
dependency, and it does not install `samtools` or `zstd`, which `data_preparation/` uses.

## Quick start

```python
from smf_net.data import dna_str_to_tensor
from smf_net.model import load_model

model = load_model("kundajelab/smfnet-ont-mesc").cuda()
dna = dna_str_to_tensor(sequence).unsqueeze(0)  # sequence: a 2048 bp string
reads = model.sample(dna, reads_per_region=32)  # [1, 32, 1024]
```

In the generated reads, 1 is occupied, 0 accessible, and 2 marks positions that are not
call sites for the model's assay. `load_model` accepts a Hugging Face Hub ID, a directory
written by `save_pretrained`, or a Lightning `.ckpt` file.

[`examples/prism.ipynb`](examples/prism.ipynb) runs PRISM on two reads of one region, using
data shipped in `examples/data/`, and compares it with profile ISM.

## Demo

The demo trains a model on a small subset of the ONT mESC data, then generates reads for
the subset's validation regions with the released model and scores them against the
observed reads. It takes about 12 minutes on one NVIDIA H100.

The demo data (44 MB) holds 1,500 training and 300 validation regions sampled from the
full data set (none from the test chromosomes), and `demo.fa`, a small reference holding
only their sequence, so the mm10 genome is not needed.
[`data_preparation/demo/make_demo_data.py`](data_preparation/demo/make_demo_data.py) made
it; `regions.tsv` maps each demo region to its mm10 coordinates.

```bash
pixi run hf download kundajelab/smfnet-data-ont-mesc --repo-type dataset \
    --include "demo/*" --local-dir data/ont_mesc
export WANDB_MODE=offline
DATA=(data.cfg.fasta_path=data/ont_mesc/demo/demo.fa
    "data.cfg.h5_path=[data/ont_mesc/demo/ONT_mESC_acRegions.h5,data/ont_mesc/demo/ONT_mESC_bgRegions.h5]")

# 1. Train for 50 epochs (about 7 minutes).
pixi run python scripts/train.py assay_type=cg_gc_accessibility "${DATA[@]}" \
    trainer.max_epochs=50 hydra.run.dir=demo/train

# 2. Generate 1,024 reads for each validation region with the released model (about 4 minutes).
pixi run python scripts/sample.py checkpoint=kundajelab/smfnet-ont-mesc "${DATA[@]}" \
    regions.source=split regions.split=val hydra.run.dir=demo/sample

# 3. Score them against the observed reads (under a minute).
pixi run python scripts/metrics.py samples_path=demo/sample/samples.h5 \
    plot.enabled=true hydra.run.dir=demo/metrics
```

These commands need an 80 GB GPU. On a smaller one, add these settings:

| GPU memory | Add to `train.py` | Add to `sample.py` | Peak use, GiB (train, sample) |
| --- | --- | --- | --- |
| 80 GB | | | 64, 25 |
| 40 GB | `data.cfg.batch_size=128` | | 32, 25 |
| 24 GB | `data.cfg.batch_size=64` | `sampling.reads_per_chunk=64` | 16, 13 |

Run times are about the same for all three.

Expected outputs:

- `demo/train/`: the checkpoint with the lowest validation loss and `last.ckpt`. In
  `csv/metrics.csv`, the validation loss falls from 0.6–0.7 after the first epoch to about
  0.34 (about 0.33 with batch size 128, 0.32 with 64).
- `demo/metrics/`: `metrics.tsv` (one row per region), `site_correlations.tsv`,
  `metrics_summary.tsv` and `metrics.pdf`. Pooled over the 300 regions, the Pearson
  correlation of generated with observed profiles is about 0.77 on matched reads, against
  0.86 for a pseudo-replicate of the observed reads and 0.38 for a baseline from flanking
  sequence alone (`site_correlations.tsv`). In `metrics_summary.tsv`, the model closes about
  90% of the gap between a shuffled baseline and the pseudo-replicate (`pearson_matched`,
  `fraction`).

[`examples/demo_expected/`](examples/demo_expected/) holds these outputs from a run with the
default settings on the tested configuration: `site_correlations.tsv`,
`metrics_summary.tsv`, `metrics.pdf`, and the training run's `csv/metrics.csv` as
`train_metrics.csv`. Sampling is seeded, so on the same GPU model with the default settings
`demo/sample/samples.h5` and the scores match these exactly. Other GPUs may give slightly
different values, as do the 24 GB settings (0.7707 against 0.7711 on matched reads).
Training is not bit-for-bit reproducible, so its losses differ in the third decimal place
between runs.

The times are for a first run; later runs on the same machine are a minute or two faster,
as `torch.compile` reuses its cache.

## Pretrained models and data

These models are on the [Hugging Face Hub](https://huggingface.co/collections/kundajelab/smfnet-6ad52511a150e48c973e1ea6):

| Model | Data | Reference |
| --- | --- | --- |
| [`kundajelab/smfnet-ont-mesc`](https://huggingface.co/kundajelab/smfnet-ont-mesc) | Nanopore SMF (CpG and GpC), mouse ES cells | mm10 |
| [`kundajelab/smfnet-fiberseq-gm12878`](https://huggingface.co/kundajelab/smfnet-fiberseq-gm12878) | Fiber-seq (m6A), GM12878 | hg38 |

The ONT mESC model's training data is a Hub dataset,
[`kundajelab/smfnet-data-ont-mesc`](https://huggingface.co/datasets/kundajelab/smfnet-data-ont-mesc)
(see [Train a model](#train-a-model)).

Every model in the paper (including the GpC cell-type models and the read-depth series), the
processed training data, and the results behind the figures are on Zenodo:
[10.5281/zenodo.23407172](https://doi.org/10.5281/zenodo.23407172). The code that makes the
paper's figures from them is in
[kundajelab/SMFNet_paper_analysis_code](https://github.com/kundajelab/SMFNet_paper_analysis_code).

## Workflows

Each step is a [Hydra](https://hydra.cc) app in `scripts/`. Settings are given as
`key=value` arguments; `--help` lists them all with their defaults. Every run writes its
outputs and its full configuration (`.hydra/`) to a new directory, by default
`outputs/<date>/<time>/` under the directory you run it from; set `hydra.run.dir=<dir>` to
choose it.

A few things to know about the arguments:

- `checkpoint=` accepts a Hub ID, a `save_pretrained` directory or a `.ckpt` file.
- Relative paths are resolved against the directory you run the command from.
- Quote arguments whose values contain brackets, such as lists
  (`"data.cfg.h5_path=[active.h5,background.h5]"`), or `=`.

### Generate reads

`scripts/sample.py` writes `samples.h5` with generated reads for each region, and the
observed reads too when the regions come from a data split. Regions can come from a FASTA
file of 2048 bp sequences, from a BED file and a reference genome, or from the validation or
test split of a training data set:

```bash
pixi run python scripts/sample.py checkpoint=kundajelab/smfnet-ont-mesc \
    regions.source=fasta regions.fasta_path=sequences.fa

pixi run python scripts/sample.py checkpoint=kundajelab/smfnet-ont-mesc \
    regions.source=bed regions.bed_path=regions.bed regions.reference_fasta=mm10.fa

pixi run python scripts/sample.py checkpoint=kundajelab/smfnet-ont-mesc \
    data.cfg.fasta_path=mm10.fa "data.cfg.h5_path=[ONT_mESC_acRegions.h5]" \
    regions.split=val sampling.reads_per_region=1024
```

BED intervals are resized to 2048 bp about their centers, and minus-strand intervals are
reverse complemented.

### Compare generated and observed reads

`scripts/metrics.py` scores the generated reads of a `samples.h5` from a data split against
the observed reads: average profiles, per-site and pairwise statistics, run lengths and
covariation, against shuffled and positional baselines fitted on the training split. It
writes `metrics.tsv` (one row per region), a summary table and, with `plot.enabled=true`,
a PDF of plots.

```bash
pixi run python scripts/metrics.py samples_path=outputs/.../samples.h5 plot.enabled=true
```

### Marginalize motifs

`scripts/marginalize.py` inserts motifs into random background sequences, generates reads
with and without them, and reports each motif's effect on mean SMF around it. The packaged
panel holds JASPAR consensus motifs; `motifs.path=` takes your own TSV of motifs instead.

```bash
pixi run python scripts/marginalize.py checkpoint=kundajelab/smfnet-ont-mesc \
    "motifs.names=[CTCF,NRF1]" plot.enabled=true
```

### Profile ISM

`scripts/ism.py` mutates every base in a window and measures the change in the predicted
average profile. It takes regions the same ways as `sample.py`.

```bash
pixi run python scripts/ism.py checkpoint=kundajelab/smfnet-ont-mesc \
    regions.source=fasta regions.fasta_path=sequences.fa profile_ism.length=200 plot.enabled=true
```

### PRISM

PRISM scores how each substitution changes the model's likelihood of one observed read. It
is used from Python (`smf_net.evaluation.prism`), since you choose which reads to interpret;
see [`examples/prism.ipynb`](examples/prism.ipynb).

### Train a model

`scripts/train.py` trains on one or more H5 files of observed reads and logs to
[Weights & Biases](https://wandb.ai) (set `WANDB_MODE=offline` to log locally only) and to a
CSV file in the run directory.

To train on the ONT mESC data, download it (3.6 GB) from the Hub and the mm10 genome from
UCSC:

```bash
pixi run hf download kundajelab/smfnet-data-ont-mesc --repo-type dataset --local-dir data/ont_mesc
curl -L https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz | gunzip > data/mm10.fa

pixi run python scripts/train.py assay_type=cg_gc_accessibility \
    data.cfg.fasta_path=data/mm10.fa \
    "data.cfg.h5_path=[data/ont_mesc/ONT_mESC_acRegions.h5,data/ont_mesc/ONT_mESC_bgRegions.h5]"
```

The run directory then holds the checkpoint with the lowest validation loss
(`epoch<N>-step<M>.ckpt`) and the last one (`last.ckpt`), either of which `checkpoint=`
accepts in the other scripts.

Chromosomes are split into training, validation (chr8, chr16) and test (chr1, chr3, chr6)
sets by default; override `data.cfg.train_chrm`, `val_chrm` and `test_chrm` for other
genomes. Each released model's training configuration is on Zenodo.

## Using your own data

Training reads are stored in H5 files with, for each region named `chr:start-end`, a read
matrix `smf_mat/<region>` (`[reads, sites]` integers: 1 occupied, 0 accessible, -1 no call) and
the sites' offsets within the region, `smf_pos/<region>`. Regions are 2114 bp
(`data.cfg.h5_region_length`) by default, and training crops 2048 bp windows from them.
[`data_preparation/`](data_preparation/README.md) builds these files from SMF-ONT, short-read SMF
and Fiber-seq data and documents the format in full. Choose the `assay_type` that matches
your assay's call sites (see [Assay types](#assay-types)).

## Assay types

`assay_type` says which positions an assay can call. It decides:

- **Call sites.** When data is loaded, `smf_mat` columns at other positions are dropped, and
  regions left without calls are skipped. The model learns and generates reads only at call
  sites, and marks every other position 2.
- **Reverse-complement augmentation.** Whether training flips regions to the other strand at
  random by default (`data.cfg.rc_augmentation` overrides it), and how calls move when it
  does: a CpG or GpC call moves to the other cytosine of its dinucleotide, while an m6A call
  stays on its base pair.
- **Flank baseline sites** in `metrics.py`, which fits one baseline per site type.

| `assay_type` | Call sites | RC augmentation by default | Data preparation |
| --- | --- | --- | --- |
| `cg_gc_accessibility` | CpG and GpC cytosines | on | [SMF-ONT](data_preparation/SMF-ONT/README.md); [short-read SMF](data_preparation/Short-read%20SMF/README.md), combined mode |
| `gch_accessibility` | GpC in a DGCH context (no CpG overlap) | off, as the bait-capture data it was made for is strand-biased | [short-read SMF](data_preparation/Short-read%20SMF/README.md), GC-only mode |
| `fiberseq_m6a` | every A and T | on | [Fiber-seq](data_preparation/fiberseq/README.md) |

`train.py` takes `assay_type=` for both the data and the model, and saved models keep it,
so `sample.py`, `marginalize.py` and `ism.py` use the model's assay. When `sample.py` or
`ism.py` reads regions from a data split, set `data.cfg.assay_type` (and the splits) to match
the model as well; they stop with an error if the assays differ.

### Train on Fiber-seq data

[`data_preparation/fiberseq/`](data_preparation/fiberseq/README.md) builds `active.h5` and
`background.h5` from Fiber-seq BAMs. The default chromosome splits are for mouse, so set
splits for human data; the released GM12878 model validated on chr8 and chr20, tested on
chr1, chr3 and chr6, and trained on the other chromosomes:

```bash
SPLITS=("data.cfg.train_chrm=[chr2,chr4,chr5,chr7,chr9,chr10,chr11,chr12,chr13,chr14,chr15,chr16,chr17,chr18,chr19,chr21,chr22,chrX,chrY]"
    "data.cfg.val_chrm=[chr8,chr20]")

pixi run python scripts/train.py assay_type=fiberseq_m6a "${SPLITS[@]}" \
    data.cfg.fasta_path=data/hg38.fa \
    "data.cfg.h5_path=[GM12878/h5/active.h5,GM12878/h5/background.h5]"

pixi run python scripts/sample.py checkpoint=kundajelab/smfnet-fiberseq-gm12878 \
    data.cfg.assay_type=fiberseq_m6a "${SPLITS[@]}" data.cfg.fasta_path=data/hg38.fa \
    "data.cfg.h5_path=[GM12878/h5/active.h5,GM12878/h5/background.h5]" regions.split=val
```

About 58% of positions are A or T, against a few percent for the cytosine assays, so
Fiber-seq regions hold many more calls.

### Add an assay

If your assay calls the same sites as one of the above, use that `assay_type`, and set
`data.cfg.rc_augmentation` if its strand handling differs (RC augmentation is only valid
when the data merges both strands evenly).

Otherwise, add it in [`src/smf_net/data.py`](src/smf_net/data.py):

1. Add its name to `AssayType` and an `AssaySpec` to `ASSAYS`, giving its site rule
   (`target_rule`) and RC augmentation default. If an existing rule (`cg_gc`, `gch` or
   `at`) describes its sites, that is all.
2. For a new site rule, add it to `AssaySpec.target_rule` and implement it in
   `get_valid_mask` (which positions are call sites) and `reverse_complement_dna_and_smf`
   (where a call goes on the other strand).
3. For `metrics.py`, add the rule's site types to `SITE_TYPES` and `site_types` in
   [`src/smf_net/evaluation/flank_baselines.py`](src/smf_net/evaluation/flank_baselines.py).
4. Add tests next to the existing ones in `tests/test_data.py`.

Don't rename or change the existing assays: the released models depend on them.

## Repository layout

```
src/smf_net/             the package: model, network, data loading, diffusion schedule
src/smf_net/evaluation/  sampling, metrics, marginalization, ISM and PRISM, and their plots
scripts/                 command-line apps (train, sample, metrics, marginalize, ism)
examples/                notebooks, small example inputs (examples/data/) and the demo's expected outputs
data_preparation/        building training data from raw reads
tests/                   pytest tests
```

## Development

```bash
pixi run test     # run the tests
pixi run fmt      # sort imports and format with ruff
pixi run check    # fail on unformatted code or lint findings
```

## Citation

SMFNet was developed by Till Schwämmle and Nathaniel Diamant in the laboratories of Anshul Kundaje (Stanford University) and Arnaud Krebs (EMBL Heidelberg).

If you use SMFNet in your work, please cite our [bioRxiv preprint (10.64898/2026.10.30.762954v1)](https://www.biorxiv.org/content/10.64898/2026.10.30.762954v1).

## License

SMFNet is released under the MIT License (see [LICENSE](LICENSE)).
