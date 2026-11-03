# nf-smfont

```
  ╭═══════════════════════════════════════╮
  ║  ┌─┐┌─┐      ┌─┐┌┬┐┌─┐┌─┐┌┐┌┌┬┐       ║
  ║  │││├┤  ───  └─┐││││├┤ │ ││││ │       ║
  ║  ┘└┘└   ───  └─┘┴ ┴└  └─┘┘└┘ ┴        ║
  ║                                       ║
  ║    Single Molecule Footprinting       ║
  ║    with Oxford Nanopore reads         ║
  ╰═══════════════════════════════════════╯
```

[![Nextflow](https://img.shields.io/badge/nextflow%20DSL2-%E2%89%A526.04.0-23aa62.svg)](https://www.nextflow.io/)
[![run with conda](http://img.shields.io/badge/run%20with-conda-3EB049?labelColor=000000&logo=anaconda)](https://docs.conda.io/en/latest/)
[![run with docker](https://img.shields.io/badge/run%20with-docker-0db7ed?labelColor=000000&logo=docker)](https://www.docker.com/)
[![run with singularity](https://img.shields.io/badge/run%20with-singularity-1d355c.svg?labelColor=000000)](https://sylabs.io/docs/)

**nf-smfont** is a comprehensive Nextflow pipeline for **Single Molecule Footprinting (SMF)** analysis using **Oxford Nanopore Technologies (ONT)** data. The pipeline processes raw ONT sequencing data (pod5/fast5 files) or pre-processed BAM/CRAM files to identify transcription factor binding sites, chromatin accessibility, and co-occupancy patterns at single-molecule resolution.

## Introduction

Single Molecule Footprinting (SMF) is a powerful technique that when combined with long-read sequencing enables simultaneous detection of DNA methylation and protein-DNA interactions at single-molecule resolution over kilobases. This pipeline enables researchers to:

- **Perform bulk methylation analysis** for quality control and comparisons between experiments
- **Analyze chromatin accessibility** at single-molecule resolution
- **Identify transcription factor (TF) binding sites** through accessibility patterns
- **Detect co-occupancy** of multiple transcription factors
- **Study nucleosome positioning** and chromatin structure

The pipeline supports the complete workflow from raw ONT data to publication-ready results, including basecalling, alignment, methylation calling, single-molecule sorting, co-occupancy (DHS accessibility, TF and/or RNApolII co-occupancy) analysis, and phi score analysis {beta}.

## Pipeline Overview

The nf-smfont pipeline consists of several modular components that can be run independently step by step or as a complete workflow:

### Core Modules

1. **Basecalling** (`--basecall`)
   - Convert pod5/fast5 files to FASTQ using Dorado or Bonito
   - Support for methylation-aware basecalling models

2. **Mapping** (`--mapping`)
   - Align reads to reference genome using minimap2
   - Generate sorted, indexed BAM/CRAM files

3. **Methylation Calling** (`--methyl`)
   - Extract methylation information from modified basecalls
   - Support for CpG, GpC, and adenine modifications (any mod calling models trained with remora can be used)

4. **Methylation BED Generation** (`--metBED`)
   - Convert methylation calls to mBED format [using this data structure](https://github.com/timplab/nanopore-methylation-utilities)
   - Support for CpG, GpC, and adenine modifications only
   - Filter reads based on their methylation rate
   - Generate per-read statistics

5. **Frequency Analysis** (`--metFreq`)
   - Calculate bulk methylation frequencies
   - Generate browser tracks and QC plots

6. **Single Molecule Sorting** (`--SMsorting`)
   - Sort molecules based on footprinting patterns
   - Support for
      - TF motifs (3 bins: upstream_bin[-35,-25],  TFBS[-7,7], downstream_bin[25,35])
      - TSS (4 bins: upstream_bin[-58,-43], TATAbox[-36,-22], RNApolII[28,47], post_polII[54,69])
      - DHS features (default 100bp centered bin)

7. **Co-occupancy Analysis** (`--coop`)
   - Analyze co-binding of multiple transcription factors
   - Statistical analysis of TF cooperation

8. **Nucleosome Analysis** (`--nucl_reg`) - BETA
   - Calculate phi scores for nucleosome positioning
   - Analyze periodic methylation patterns

## Quick Start

### Prerequisites

- **Nextflow** (≥25.04.0)
- **Container technology**: Docker or Singularity/Apptainer, **and** Conda (or mamba)

### Input Data Formats

The pipeline accepts multiple input formats:

#### Required Inputs

- **Raw ONT data**: pod5 or fast5 files

or

- **Processed data**: BAM/CRAM files with `MM` and `ML` methylation tags (for SMF analysis)

#### Optional Inputs

- **Reference genome** in FASTA or mmi format
- **Feature files**: BED/TS/RDS(GRanges) files with TF binding sites, TSS coordinates, or DHS regions

## Usage

### Installation

Clone the pipeline:

```bash
git clone https://git.embl.org/grp-krebs/nf-smfont.git
cd nf-smfont
```

Alternatively, you can run the pipeline directly from the repository without cloning:

```bash
nextflow run https://git.embl.org/grp-krebs/nf-smfont
```

### Basic Command

Methylation calling and mapping of ONT pod5 files:

```bash
nextflow run nf-smfont/main.nf \
   -profile singularity,multi_gpus \
   --input_folder /path/to/input/files/ \
   --input_ext pod5 \
   --sample_name SMF_Sample \
   --ref /path/to/genome.fa \
   --outdir results \
   --basecall --mapping --methyl \
   --basecaller_cfg dna_r10.4.1_e8.2_400bps_sup@v4.2.0 \
   --remora_cfg res_dna_r10.4.1_e8.2_400bps_sup@v4.2.0_5mC@v2 \
   --output_bam true
```
> [!WARNING]
> Please provide pipeline parameters via the CLI or Nextflow `-params-file` option. Custom config files including those provided by the `-c` Nextflow option can be used to provide any configuration _**except for parameters**_

### Complete SMF-ONT Analysis

For a full SMF-ONT analysis including single-molecule sorting:

```bash
nextflow run nf-smfont/main.nf \
   -profile singularity,multi_gpus \
   --input_folder /path/to/pod5/files/ \
   --input_ext pod5 \
   --sample_name SMF_Sample \
   --ref /path/to/dm6.fa \
   --outdir results \
   --basecall --mapping --methyl --metBED --metFreq --SMsorting --coop \
   --basecaller dorado \
   --basecaller_cfg dna_r10.4.1_e8.2_400bps_sup@v4.2.0 \
   --remora_cfg custom \
   --remora_model_path /g/krebs/boulanger/DB/nanopore/dorado_models/res_dna_r10.4.1_e8.2_400bps_sup@v4.0.1_5mC@v2 \
   --llr 2.0 \
   --BSgenome BSgenome.Dmelanogaster.UCSC.dm6 \
   --motifs cpg,gpc \
   --features /path/to/DHS_features.bed \
   --featureNature DHS \
   --coop_pairmode Freq5 \
   --coop_rightwindow 40 \
   --coop_min_cov 10 \
   --coop_sig_threshold 0.05 \
   --coop_min_cov_adjust 0 \
   --coop_dhs_min_acc_freq 10 \
   --coop_dhs_max_acc_freq 90
```

### Key Parameters

#### Input/Output
- `--input_folder`: Directory containing input files (multiple separated by commas - required)
- `--input_ext`: File extension (`pod5`, `fast5`, `bam`, `cram`) - (multiple separated by commas - required)
- `--sample_name`: Sample identifier for output naming (required)
- `--outdir`: Output directory (required)
- `--publish_dir_mode`: Publishing mode to output directories (default: `copy`)

#### Workflow Control
- `--basecall`: Enable basecalling
- `--mapping`: Enable read mapping
- `--methyl`: Enable methylation calling
- `--metBED`: Generate methylation BED files
- `--metFreq`: Calculate methylation frequencies
- `--SMsorting`: Enable single-molecule sorting
- `--coop`: Enable co-occupancy analysis
- `--nucl_reg`: Enable nucleosome analysis

#### Basecalling Options
- `--basecaller`: Basecaller to use (`dorado`, `bonito`)
- `--basecaller_cfg`: Basecalling model configuration
- `--basecaller_model_path`: Path to custom basecaller model (if `basecaller_cfg` is set `custom`)
- `--basecaller_args`: Additional arguments for the basecaller
- `--cuda_device`: CUDA device to use for basecalling (default: `cuda:all`)
- `--qscore_filter`: Minimum Q-score for filtering reads (default: `10`)
- `--output_fastq`: Output FASTQ files (default: `false`)
- `--output_bam`: Output BAM files (default: `true`)

#### Mapping Options
- `--ref`: Reference genome FASTA file (required if doing full SMF analysis)
- `--ref_mmi`: Reference genome minimap2 index file (optional)

#### Methylation calling
- `--remora_cfg`: Methylation calling configuration (required)
- `--remora_model_path`: Path to custom remora model (if `remora_cfg` is set `custom`)
- `remora_args`: Additional arguments for remora (default: `modified-bases-threshold 0.05`)

#### Methylation Analysis
- `--motifs`: Methylation motifs to analyze (cpg, gpc, a, c {beta})
- `--filter_read_met_rate`: Filter reads by methylation rate (default: `true`)
- `--min_read_met_rate`: Minimum read methylation rate (default: `0.05`)
- `--max_read_met_rate`: Maximum read methylation rate (default: `0.50`)
- `--reads_stats`: Generate per-read methylation statistics (default: `true`)

#### SMF-Specific Options
- `--BSgenome`: BSgenome package for the reference genome (e.g. `BSgenome.Dmelanogaster.UCSC.dm6`)
- `--BSgenome_path`: Path to the BSgenome package (need to be build with `R4.2.X` series)
- `--features`: Feature file(s) for single-molecule sorting (multiple separated by commas)
- `--featureNature`: Type of features (`TF`, `TSS`, `DHS`) - (multiple separated by commas)
- `--min_cytosine_coverage`: Minimum coverage for cytosine sites (default: `20`)
- `--collapse_strand_frequencies`: Collapse strand information for palyndromic contexts (e.g. GpC, CpG) (default: `true`)

#### Co-occupancy Analysis Options
- `--coop_pairmode`: Pair selection mode (`all` or `Freq<1-100>`, default: `Freq5`)
- `--coop_rightwindow`: Maximum distance in bp from the anchor feature to a linked feature (default: `40`)
- `--coop_min_cov`: Minimum number of single molecules covering both features to test a pair (default: `10`)
- `--coop_sig_threshold`: Adjusted p-value threshold for calling a pair significant (default: `0.05`)
- `--coop_min_cov_adjust`: Minimum `cov.adjust` value to retain a pair in the master table (default: `0`, no filter)
- `--coop_dhs_min_acc_freq`: Minimum DHS accessibility frequency in percent (default: `10`)
- `--coop_dhs_max_acc_freq`: Maximum DHS accessibility frequency in percent (default: `90`)

## Pipeline Outputs

### Directory Structure

```
results/
├── BAM/                      # Aligned BAM/CRAM files (fail, pass, pass_methFiltered)
├── mBED/                     # BED files with methylation information (freqs, mBED, read_stats)
├── multiqc/                  # General quality control report
├── QC/                       # Pipeline specific quality control reports and plots
├── SMF/                      # Single-molecule analysis results
│   ├── BigWig/                       # Bulk methylation in bigwig format
│   ├── MethCalls/                    # Bulk methylation in QS format (GRanges object)
│   ├── SM_sorting/                   # Single molecule sorting results
│   ├──  ??? /                        # Co-occupancy analysis
│   └──  Phiscore /                   # phi score analysis
└── pipeline_info/            # Pipeline execution information
```

### Key Output Files

#### Methylation Data
- `*_methcalls_*.bed.gz`: per-read methylation information in BED format
- `*_freqs_*.bed.gz`: Cytosine methylation frequencies
- `*_read_stats_*.bed.gz`: Per-read methylation statistics

#### SMF Analysis
- `SMvect_*.qs`: Single-molecule sorting results - list of readIDs
- `pattern_*.qs`: Single-molecule sorting results - per region information (binary patterns)
- `states_*.qs`: Single-molecule sorting results - per region information (biological states)

## Configuration

### Execution Profiles

The pipeline includes several pre-configured profiles:

- `docker`: Use Docker containers
- `singularity`: Use Singularity containers
- `apptainer`: Use Apptainer containers
- `conda`: Use Conda environments
- `multi_gpus`: Enable multi-GPU support for Dorado and Bonito basecalling (Default: 1 GPU)
<!-- - `test`: Run with test data -->
### Resource Configuration

Adjust computational resources in `nextflow.config`:

```nextflow
process {
    cpus = 8
    memory = '32 GB'
    time = '4 h'
}
```

### Container Images

The pipeline uses custom containers hosted on Docker Hub:

- `krebslabrep/smfont_dorado`: Dorado basecaller
- `krebslabrep/smfont_bonito`: Bonito basecaller
- `krebslabrep/smfont_utils`: SMF analysis tools
<!-- - `ontresearch/modkit`: Methylation extraction -->
## Advanced Usage

### Co-occupancy Analysis

To analyze transcription factor co-occupancy:

```bash
nextflow run nf-smfont \
  --input_folder /path/to/data/ \
  --features /path/to/TF1_sites.bed,/path/to/TF2_sites.bed \
  --featureNature TF,TF \
  --SMsorting --coop \
  --TF_hybrid_sorting true
```

### Nucleosome Positioning

For nucleosome positioning analysis:

```bash
nextflow run nf-smfont \
  --nucl_reg \
  --sliding_win_size 20000 \
  --pseudocount false
```

### Custom Feature Analysis

Provide custom feature files in BED, TSV, or RDS format:

```bash
nextflow run nf-smfont \
  --features /path/to/custom_features.bed \
  --featureNature DHS \
  --bins_size 500
```

## Container and Environment Management

### Docker Images

The pipeline uses optimized container images:

```yaml
containers:
  dorado: krebslabrep/smfont_dorado:0.8.0
  bonito: krebslabrep/smfont_bonito:0.7.3
  utils: krebslabrep/smfont_utils:v1.0 # under development
```

### Conda Environments

For now, the pipeline requires Conda environments for tool dependencies.
Therefore, Conda is activated although container engine is requested.
Containarisation is planned for future releases.
<!-- Alternative conda environments are provided for each tool: -->
- `fast5converter`: pod5/fast5 file conversion
- `bonito`: Bonito basecaller
- `BAMutils`: BAM file processing
- `methylTools`: Methylation analysis
- `SMFenv`: R-based SMF analysis
- `pycoqc`: pycoqc report

### Module System

For HPC clusters using easybuild module systems:

```bash
nextflow run nf-smfont --use_module true
```
You can modify the `nextflow.config` file to specify required modules:

Required modules:
- `dorado`
- `bonito`
- `minimap2`
- `SAMtools`
- `R` (bundle with bioconductor packages)

## Citation and Publication

### Primary Citation

If you use nf-smfont for your research, please cite:

> Molecular co-accessibility identifies coordinated regulation between distant cis-regulatory elements
>
> Mathias Boulanger, Kasit Chatsirisupachai, Rene Snajder, Marc Jan Bonder, Karine Lapouge, Kim Remans and Arnaud R. Krebs
>
> bioRxiv, 2025. doi: [10.1101/2025.08.05.668660](https://doi.org/10.1101/2025.08.05.668660)

### Software Dependencies

This pipeline uses code and infrastructure developed and maintained by the [nf-core](https://nf-co.re) community, reused here under the [MIT license](https://github.com/nf-core/tools/blob/master/LICENSE).

> The nf-core framework for community-curated bioinformatics pipelines.
>
> Philip Ewels, Alexander Peltzer, Sven Fillinger, Harshil Patel, Johannes Alneberg, Andreas Wilm, Maxime Ulysse Garcia, Paolo Di Tommaso & Sven Nahnsen.
>
> Nat Biotechnol. 2020 Feb 13. doi: 10.1038/s41587-020-0439-x.
> In addition, references of tools and data used in this pipeline are as follows:

- **Nextflow**: Di Tommaso et al., Nature Biotechnology 2017
- **Dorado**: Oxford Nanopore Technologies
- **Bonito**: Oxford Nanopore Technologies
- **minimap2**: Li, Bioinformatics 2018
- ...

## Development and Contributing

### Repository Structure

```
nf-smfont/
├── main.nf                # Main workflow entry point
├── nextflow.config        # Configuration file
├── workflows/             # Workflow definitions
├── modules/               # Process modules
│   ├── local/              # Custom modules
│   └── nf-core/            # Community modules
├── subworkflows/          # Subworkflow components
│   ├── local/              # Custom Subworkflows
│   └── nf-core/            # Community Subworkflows
├── bin/                   # Custom scripts
├── assets/                # Static assets and configs
└── conf/                  # Additional configurations
```

### Custom Scripts

The `bin/` directory contains specialized scripts that need to be able to run independently of the pipeline. These scripts are used for specific tasks such as:

- `BAM_to_mBED.py`: Convert BAM to methylation BED
- `smf_sort.R`: Single molecule sorting
- `smf_cooc.R`: Co-occupancy analysis
- `smf_phiscore.R`: Nucleosome positioning
- `freq_from_mBED.py`: Frequency calculation
- ...

### Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Develop your new functionality (contact us if you need help)
4. Submit a pull request

### Issue Reporting

Report bugs and feature requests through:
- GitLab issues: https://git.embl.de/grp-krebs/boulanger_2025_smf-ont/issues
- Contact: [Contact the Krebs Lab](https://www.embl.org/groups/krebs/)

## License and Disclaimer

This pipeline is distributed under the MIT License. See `LICENSE` file for details.

**Disclaimer**: This software is provided "as is" without warranty. Users are responsible for validating results and ensuring appropriate use for their research applications.

## Acknowledgments

### Development Team

- **Mathias Boulanger**: Pipeline development and SMF-ONT methodology
- **Michela Palamin**: Debugging, reviewing and testing

### Funding

This work was supported by:
- European Molecular Biology Laboratory (EMBL)
- Deutsche Forschungsgemeinschaft (DFG)

### Infrastructure

Special thanks to:
- EMBL IT Services for computational infrastructure
- EMBL GeneCore sequencing facility for sequencing platform support
- EMBL Data Science Centre / MODIS team for their outstanding support in the development of the nanopore data management in their [Lab Integrated Data platform](https://labid-demo.embl.de/home)
- nf-core community for workflow framework

---
