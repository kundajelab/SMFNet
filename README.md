# SMFNet

SMFNet is a discrete diffusion model that predicts the distribution of single-molecule
footprinting (SMF) reads given DNA sequence as input. From 2048 bp of DNA, it generates
individual reads over the central 1024 bp, each a string of per-site calls: 1 where the
molecule was occupied (protected from the methylase) and 0 where it was accessible.
Sampling many reads gives the range of molecular states a sequence encodes, not only its
average accessibility.

Models can be trained on three assays (`assay_type`):

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

SMFNet's environment is managed with [pixi](https://pixi.sh), which installs Python,
PyTorch, the CUDA libraries and every other dependency from the lockfile `pixi.lock`. No
system CUDA toolkit or compilers are needed.

```bash
git clone https://github.com/kundajelab/SMFNet.git
cd SMFNet
pixi install            # GPU environment (Linux with an NVIDIA driver for CUDA 12)
pixi install -e cpu     # CPU-only environment (Linux or Apple Silicon macOS)
```

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
and Fiber-seq data and documents the format in full.

## Repository layout

```
src/smf_net/             the package: model, network, data loading, diffusion schedule
src/smf_net/evaluation/  sampling, metrics, marginalization, ISM and PRISM, and their plots
scripts/                 command-line apps (train, sample, metrics, marginalize, ism)
examples/                notebooks, with small example inputs in examples/data/
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
