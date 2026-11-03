#!/usr/bin/env Rscript
# Prepare alignments and a reusable QuasR project from a sample table or FASTQs.
# Run with --help to inspect input formats and command-line options.

main <- function(args = commandArgs(trailingOnly = TRUE)) {
  if (!length(args) || "--help" %in% args) {
    cat("Prepare QuasR alignments and a saved project for the Illumina extractors.\n",
      "Required: --sample-file samples.tsv OR --fastq-dir DIR; --out-dir DIR\n",
      "Options: --genome BSgenome.Mmusculus.UCSC.mm10 --workers 8\n",
      "         --paired fr --bisulfite undir --overwrite false\n",
      "         --alignment-parameter '-e 70 -X 1000 -k 2 --best -strata'\n",
      "         --lib-paths DIR1:DIR2\n",
      "Sample tables: FileName1, FileName2, SampleName for FASTQ pairs;\n",
      "or FileName, SampleName for existing BAMs. Paths are relative to\n",
      "the sample table. --fastq-dir pairs *_1.fastq.gz with *_2.fastq.gz.\n",
      "Outputs: project.rds, samples.tsv, alignments.tsv, alignments/,\n",
      "cache/, package_versions.tsv and sessionInfo.txt.\n",
      sep = ""
    )
    return(invisible(NULL))
  }
  # Parse and validate options before creating output directories.
  options <- list(
    genome = "BSgenome.Mmusculus.UCSC.mm10", workers = "8", paired = "fr",
    bisulfite = "undir", overwrite = "false",
    `alignment-parameter` = "-e 70 -X 1000 -k 2 --best -strata"
  )
  allowed <- c(names(options), "sample-file", "fastq-dir", "out-dir", "lib-paths")
  if (length(args) %% 2L) stop("Each option requires a value; use --help")
  for (i in seq(1L, length(args), by = 2L)) {
    key <- sub("^--", "", args[i])
    if (!startsWith(args[i], "--") || !key %in% allowed) stop("Unknown option: ", args[i])
    options[[key]] <- args[i + 1L]
  }
  if (is.null(options[["sample-file"]]) == is.null(options[["fastq-dir"]])) {
    stop("Supply exactly one of --sample-file and --fastq-dir")
  }
  if (is.null(options[["out-dir"]])) stop("Missing --out-dir")
  workers <- suppressWarnings(as.numeric(options$workers))
  if (!is.finite(workers) || workers < 1 || workers != floor(workers)) stop("Invalid --workers")
  if (!options$overwrite %in% c("true", "false")) stop("Invalid --overwrite")
  if (!is.null(options[["lib-paths"]])) {
    .libPaths(c(strsplit(options[["lib-paths"]], .Platform$path.sep, fixed = TRUE)[[1]], .libPaths()))
  }

  # Normalize file paths so the saved project is independent of the working directory.
  if (!is.null(options[["sample-file"]])) {
    sample_table_path <- normalizePath(options[["sample-file"]], mustWork = TRUE)
    samples <- read.delim(sample_table_path, check.names = FALSE, stringsAsFactors = FALSE)
    columns <- if (all(c("FileName1", "FileName2") %in% names(samples))) {
      c("FileName1", "FileName2")
    } else {
      "FileName"
    }
    if (!all(c(columns, "SampleName") %in% names(samples))) stop("Invalid sample table columns")
    samples <- samples[, c(columns, "SampleName"), drop = FALSE]
    for (column in columns) {
      paths <- path.expand(samples[[column]])
      relative <- !grepl("^/|^[A-Za-z]:[/\\\\]", paths)
      paths[relative] <- file.path(dirname(sample_table_path), paths[relative])
      samples[[column]] <- normalizePath(paths, mustWork = TRUE)
    }
  } else {
    folder <- normalizePath(options[["fastq-dir"]], mustWork = TRUE)
    # Pair read files by their shared basename and check that every mate exists.
    read1 <- sort(list.files(folder, pattern = "_1\\.fastq\\.gz$", full.names = TRUE))
    if (!length(read1)) stop("No *_1.fastq.gz files found")
    read2 <- sub("_1\\.fastq\\.gz$", "_2.fastq.gz", read1)
    if (!all(file.exists(read2))) stop("Missing read-2 FASTQ: ", paste(read2[!file.exists(read2)], collapse = ", "))
    samples <- data.frame(
      FileName1 = read1, FileName2 = read2,
      SampleName = sub("_1\\.fastq\\.gz$", "", basename(read1))
    )
  }
  if (!nrow(samples) || anyNA(samples) || any(samples$SampleName == "")) stop("Empty or incomplete sample table")
  if (anyDuplicated(samples[, setdiff(names(samples), "SampleName"), drop = FALSE])) stop("Repeated input files")
  output_directory <- options[["out-dir"]]
  dir.create(output_directory, recursive = TRUE, showWarnings = FALSE)
  output_directory <- normalizePath(output_directory, mustWork = TRUE)
  project_file <- file.path(output_directory, "project.rds")
  if (file.exists(project_file) && options$overwrite != "true") stop("Project exists; use --overwrite true")
  for (name in c("alignments", "cache")) dir.create(file.path(output_directory, name), showWarnings = FALSE)
  sample_file <- file.path(output_directory, "samples.tsv")
  write.table(samples, sample_file, sep = "\t", quote = FALSE, row.names = FALSE)
  for (package in c("QuasR", "BSgenome", options$genome)) {
    if (!requireNamespace(package, quietly = TRUE)) stop("Missing R package: ", package)
  }
  # Use one worker pool for alignment and stop it when this function exits.
  cluster <- if (workers > 1) parallel::makeCluster(as.integer(workers)) else 1L
  if (workers > 1) {
    on.exit(parallel::stopCluster(cluster), add = TRUE)
    parallel::clusterCall(cluster, function(paths) {
      .libPaths(paths)
      loadNamespace("QuasR")
      NULL
    }, .libPaths())
  }

  # Reuse existing compatible alignments or align the supplied reads.
  project <- QuasR::qAlign(
    sampleFile = sample_file, aligner = "Rbowtie", genome = options$genome,
    alignmentsDir = file.path(output_directory, "alignments"), projectName = "prj",
    paired = options$paired, bisulfite = options$bisulfite,
    alignmentParameter = options[["alignment-parameter"]],
    cacheDir = file.path(output_directory, "cache"), clObj = cluster
  )
  # Save atomically, then record alignment paths and dependency versions.
  temporary <- tempfile(pattern = "project_", tmpdir = output_directory, fileext = ".rds")
  on.exit(unlink(temporary), add = TRUE)
  saveRDS(project, temporary)
  if (!file.rename(temporary, project_file)) stop("Could not publish project.rds")
  write.table(QuasR::alignments(project)[[1]], file.path(output_directory, "alignments.tsv"),
    sep = "\t", quote = FALSE, row.names = FALSE
  )
  packages <- c("QuasR", "Rbowtie", "BSgenome", "Biostrings", options$genome)
  versions <- data.frame(Package = c("R", packages), Version = c(
    as.character(getRversion()),
    vapply(packages, function(p) as.character(packageVersion(p)), character(1))
  ))
  write.table(versions, file.path(output_directory, "package_versions.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
  writeLines(capture.output(sessionInfo()), file.path(output_directory, "sessionInfo.txt"))
  cat("Saved QuasR project:", project_file, "\n")
}

if (sys.nframe() == 0L) main()
