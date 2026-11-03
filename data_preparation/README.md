# Data preparation

Scripts that turn released data into the H5 files SMFNet trains on. They run outside the
`smf_net` package, take their inputs as arguments, and document their steps in each
subfolder.

| Folder | Assay |
| --- | --- |
| [`SMF-ONT/`](SMF-ONT/README.md) | SMF-ONT CG+GC (Oxford Nanopore), mouse |
| [`Short-read SMF/`](Short-read SMF/README.md) | Short-read SMF CG+GC or GC-only (Illumina), mouse |
| [`fiberseq/`](fiberseq/README.md) | Fiber-seq m6A (PacBio HiFi), human |
