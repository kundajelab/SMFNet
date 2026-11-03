#!/usr/bin/env Rscript
# Extract GpC SMF signal and a separate endogenous CpG methylation channel.
# Keep the shared script in the same directory; run with --help for options.

script_argument <- grep("^--file=", commandArgs(FALSE), value = TRUE)[1]
script_path <- sub("^--file=", "", script_argument)
script_directory <- dirname(normalizePath(script_path))

source(file.path(script_directory, "extract_illumina_gc_cg.R"))
run_extraction(gc_only = TRUE)
