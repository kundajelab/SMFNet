#!/usr/bin/env Rscript
# Shared engine and GC+CG entry point. The GC-only entry point sources this file.
# Dependencies: SingleMoleculeFootprinting, QuasR, Matrix, GenomicRanges,
# Biostrings, rhdf5, and the selected BSgenome package.
# Run Rscript extract_illumina_gc_cg.R --help for usage.

# Read paired command-line options and validate them before opening inputs.
parse_options <- function(gc_only = FALSE) {
  args <- commandArgs(trailingOnly = TRUE)
  if ("--help" %in% args || !length(args)) {
    cat("Extract SMFNet H5 from aligned Illumina bisulfite reads and a BED.\n",
      "Required: --bed regions.bed --out reads.h5 --sample SAMPLE\n",
      "          --sample-file QuasR.tsv OR --project-rds project.rds\n",
      "Options (each takes a value):\n",
      "  --genome BSgenome.Mmusculus.UCSC.mm10\n",
      "  --alignments-dir DIR --cache-dir DIR --workers 4 --batch-size 32\n",
      "  --bed-coordinates bed|one-based (default bed)\n",
      "  --coverage N (default GC+CG: 1; GC-only: 50) --min-sites 15\n",
      "  --gc-context CONTEXT (default GC+CG: DGCHN; GC-only: GC)\n",
      "  --cg-context NWCGW --bulk-mode reads|qmeth (default reads)\n",
      "  --encoding standard|legacy-gc-cg|legacy-gc (default standard)\n",
      "  --compression 1 --overwrite false --paired fr --bisulfite undir\n",
      "  --alignment-parameter '-e 70 -X 1000 -k 2 --best -strata'\n",
      "  --lib-paths DIR1:DIR2 (optional extra R library directories)\n\n",
      "BED is 0-based half-open by default. Region keys are 1-based closed;\n",
      "smf_pos/methyl_pos are 0-based offsets from the first region base.\n",
      "Standard encoding: QuasR 0 -> -1 missing; 1 -> 1 unmethylated;\n",
      "2 -> 0 methylated. Legacy options select alternate call encodings,\n",
      "including alternate missing-call behavior.\n",
      "Combined mode: GC and CG calls share smf_mat/smf_pos/read_id.\n",
      "GC-only mode: GC calls use smf_*; endogenous CG calls use methyl_*.\n",
      "--min-sites applies to the complete accessibility channel.\n",
      "Default reads mode derives bulk coverage from the same read matrix.\n",
      "Use --bulk-mode qmeth for a separate bulk methylation scan.\n",
      "One sample per output; independent samples can run as an array job.\n",
      sep = ""
    )
    return(NULL)
  }
  options <- list(
    genome = "BSgenome.Mmusculus.UCSC.mm10", workers = "4",
    `batch-size` = "32", `bed-coordinates` = "bed",
    coverage = if (gc_only) "50" else "1", `min-sites` = "15",
    `gc-context` = if (gc_only) "GC" else "DGCHN", `cg-context` = "NWCGW",
    `bulk-mode` = "reads", encoding = "standard", compression = "1",
    overwrite = "false", paired = "fr", bisulfite = "undir",
    `alignment-parameter` = "-e 70 -X 1000 -k 2 --best -strata"
  )
  allowed <- c(
    names(options), "bed", "out", "sample", "sample-file", "project-rds",
    "alignments-dir", "cache-dir", "lib-paths"
  )
  if (length(args) %% 2) stop("Every option needs a value; use --help")
  for (i in seq(1L, length(args), by = 2L)) {
    key <- sub("^--", "", args[i])
    if (!startsWith(args[i], "--") || !key %in% allowed) stop("Unknown option: ", args[i])
    options[[key]] <- args[i + 1L]
  }
  for (key in c("bed", "out", "sample")) if (is.null(options[[key]])) stop("Missing --", key)
  if (is.null(options[["sample-file"]]) == is.null(options[["project-rds"]])) {
    stop("Supply exactly one of --sample-file and --project-rds")
  }
  choices <- list(
    `bed-coordinates` = c("bed", "one-based"),
    `bulk-mode` = c("qmeth", "reads"),
    encoding = c("standard", "legacy-gc-cg", "legacy-gc"),
    overwrite = c("true", "false")
  )
  for (key in names(choices)) if (!options[[key]] %in% choices[[key]]) stop("Invalid --", key)
  for (key in c("workers", "batch-size", "coverage", "min-sites", "compression")) {
    value <- suppressWarnings(as.numeric(options[[key]]))
    if (!is.finite(value) || value != floor(value) || value < 0 ||
      (key %in% c("workers", "batch-size") && value == 0)) {
      stop("Invalid --", key)
    }
    options[[key]] <- as.integer(value)
  }
  if (options$compression > 9) stop("Compression must be 0..9")
  for (key in c("gc-context", "cg-context")) {
    if (!grepl("^[ACGTRYSWKMBDHVN]+$", options[[key]])) stop("Invalid IUPAC context: ", key)
  }
  if (file.exists(options$out) && options$overwrite != "true") stop("Output exists; use --overwrite true")
  options
}

filter_context <- function(methylation_ranges, genome, context) {
  # Select sites matching an IUPAC context on the appropriate strand.
  if (!length(methylation_ranges)) {
    methylation_ranges$GenomicContext <- character()
    return(methylation_ranges)
  }
  context_ranges <- GenomicRanges::GRanges(seqnames(methylation_ranges), ranges(methylation_ranges), strand(methylation_ranges))
  seqinfo(context_ranges) <- seqinfo(methylation_ranges)
  context_width <- if (nchar(context) %% 2L) nchar(context) else nchar(context) + 1L
  context_sequences <- Biostrings::getSeq(genome, suppressWarnings(trim(resize(context_ranges, width = context_width, fix = "center"))))
  for (site_index in which(width(context_sequences) != context_width)) {
    padding <- paste(rep("N", context_width - width(context_sequences[site_index])), collapse = "")
    context_sequences[site_index] <- if (as.character(strand(context_ranges[site_index])) == "+") {
      paste0(padding, as.character(context_sequences[site_index]))
    } else {
      paste0(as.character(context_sequences[site_index]), padding)
    }
  }
  keep <- Biostrings::vcountPattern(context, context_sequences, fixed = !grepl("[^ACGT]", context)) > 0
  matching_ranges <- methylation_ranges[keep]
  matching_ranges$GenomicContext <- rep(context, length(matching_ranges))
  matching_ranges
}

# Convert per-site calls to the requested signed integer representation.
format_calls <- function(calls, encoding) {
  # Materialize only the current region; output is stored as signed 8-bit values.
  calls <- as.matrix(calls)
  if (encoding == "standard") {
    if (any(!is.na(calls) & !calls %in% 0:2)) stop("Unexpected QuasR call values outside 0/1/2")
    # Missing calls remain -1; methylated calls are encoded as 0.
    encoded_calls <- matrix(-1L, nrow(calls), ncol(calls))
    encoded_calls[which(calls == 1)] <- 1L
    encoded_calls[which(calls == 2)] <- 0L
  } else if (encoding == "legacy-gc-cg") {
    encoded_calls <- matrix(-1L, nrow(calls), ncol(calls))
    encoded_calls[which(calls == 2)] <- 0L
  } else {
    encoded_calls <- matrix(0L, nrow(calls), ncol(calls))
    encoded_calls[which(calls == 1)] <- 1L
    encoded_calls[which(calls == 2)] <- -1L
  }
  # Transpose for the HDF5 dimension convention so Python sees reads x positions.
  t(encoded_calls)
}

# Join contexts by molecule ID and genomic position, never by row order.
merge_accessibility_calls <- function(gc_calls, cg_calls) {
  channels <- Filter(function(x) !is.null(x) && nrow(x) > 0L && ncol(x) > 0L,
    list(gc_calls, cg_calls))
  if (!length(channels)) return(NULL)
  for (calls in channels) {
    if (is.null(rownames(calls)) || is.null(colnames(calls)) ||
        anyDuplicated(rownames(calls)) || anyDuplicated(colnames(calls))) {
      stop("Each channel must have unique molecule IDs and positions")
    }
  }
  read_ids <- unique(unlist(lapply(channels, rownames), use.names = FALSE))
  positions <- unique(unlist(lapply(channels, colnames), use.names = FALSE))
  positions <- positions[order(as.numeric(positions))]
  # QuasR uses zero for missing calls; preserve it until output encoding.
  merged <- matrix(0L, length(read_ids), length(positions),
    dimnames = list(read_ids, positions))
  for (calls in channels) {
    calls <- as.matrix(calls)
    observed <- which(!is.na(calls) & calls != 0, arr.ind = TRUE)
    if (!nrow(observed)) next
    destination <- cbind(
      match(rownames(calls), read_ids)[observed[, 1]],
      match(colnames(calls), positions)[observed[, 2]])
    incoming <- calls[observed]
    existing <- merged[destination]
    # Overlapping contexts may identify the same site; do not duplicate it.
    if (any(existing != 0 & existing != incoming)) {
      stop("Conflicting GC/CG calls for the same molecule and collapsed position")
    }
    merged[destination] <- incoming
  }
  merged
}

output_channels <- function(calls, gc_only) {
  if (gc_only) {
    # Only GC reports accessibility; CG retains endogenous methylation.
    list(smf = calls$GC, methyl = calls$CG)
  } else {
    # Both enzymes report accessibility on the same molecules.
    list(smf = merge_accessibility_calls(calls$GC, calls$CG))
  }
}

run_extraction <- function(gc_only = FALSE) {
  options <- parse_options(gc_only)
  if (is.null(options)) {
    return(invisible(NULL))
  }
  if (!is.null(options[["lib-paths"]])) .libPaths(c(strsplit(options[["lib-paths"]], .Platform$path.sep)[[1]], .libPaths()))
  packages <- c(
    "SingleMoleculeFootprinting", "QuasR", "Matrix", "GenomicRanges",
    "Biostrings", "rhdf5", options$genome
  )
  for (p in packages) suppressPackageStartupMessages(library(p, character.only = TRUE))
  genome <- BSgenome::getBSgenome(options$genome)
  # Convert input regions to the coordinate convention used by GenomicRanges.
  bed <- read.table(options$bed,
    sep = "\t", header = FALSE, comment.char = "#", quote = "",
    stringsAsFactors = FALSE, colClasses = "character"
  )
  if (ncol(bed) < 3 || !nrow(bed)) stop("BED needs at least three columns and one region")
  starts <- as.numeric(bed[[2]]) + if (options[["bed-coordinates"]] == "bed") 1 else 0
  ends <- as.numeric(bed[[3]])
  if (anyNA(starts) || anyNA(ends) || any(starts < 1 | ends < starts | starts != floor(starts) | ends != floor(ends))) {
    stop("Invalid BED coordinates")
  }
  regions <- GenomicRanges::GRanges(bed[[1]], IRanges::IRanges(starts, ends))
  names(regions) <- paste0(seqnames(regions), ":", start(regions), "-", end(regions))
  regions <- regions[!duplicated(names(regions))]
  bounds <- GenomeInfoDb::seqlengths(genome)[as.character(seqnames(regions))]
  if (anyNA(bounds) || any(end(regions) > bounds)) stop("BED contig/bounds do not match genome")
  output <- normalizePath(options$out, mustWork = FALSE)
  inputs <- c(options$bed, options[["sample-file"]], options[["project-rds"]])
  if (output %in% normalizePath(inputs, mustWork = TRUE)) stop("Output must differ from inputs")
  dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)

  # Create or restore one alignment project and reuse it for every region.
  if (!is.null(options[["project-rds"]])) {
    project <- readRDS(options[["project-rds"]])
  } else {
    alignment_options <- list(
      sampleFile = options[["sample-file"]], genome = options$genome,
      aligner = "Rbowtie", projectName = "prj", paired = options$paired,
      bisulfite = options$bisulfite, alignmentParameter = options[["alignment-parameter"]], clObj = 1
    )
    if (!is.null(options[["alignments-dir"]])) alignment_options$alignmentsDir <- options[["alignments-dir"]]
    if (!is.null(options[["cache-dir"]])) {
      dir.create(options[["cache-dir"]], recursive = TRUE, showWarnings = FALSE)
      alignment_options$cacheDir <- options[["cache-dir"]]
    }
    project <- suppressWarnings(suppressMessages(do.call(QuasR::qAlign, alignment_options)))
  }
  sample_indices <- which(QuasR::alignments(project)[[1]]$SampleName == options$sample)
  if (!length(sample_indices)) stop("Sample not found: ", options$sample)
  project <- project[sample_indices]
  project@aligner <- "Rbowtie"
  if (options$encoding != "standard") warning("Using legacy conversion: missing-call encoding differs from standard")

  # Extract each region independently; workers return matrices to the parent.
  extract <- function(region) {
    suppressMessages({
      single_molecule_calls <- SingleMoleculeFootprinting::GetSingleMolMethMat(project, region, options$sample)[[options$sample]]
      if (is.null(single_molecule_calls) || !nrow(single_molecule_calls) || !ncol(single_molecule_calls)) {
        return(NULL)
      }
      if (options[["bulk-mode"]] == "reads") {
        methylation_ranges <- SingleMoleculeFootprinting:::MethSM.to.MethGR(setNames(list(single_molecule_calls), options$sample), unique(seqnames(region)))
      } else {
        methylation_ranges <- QuasR::qMeth(project, query = region, mode = "allC", collapseBySample = TRUE, keepZero = TRUE, clObj = 1)
      }
      # Collapse and filter both contexts before routing them by assay mode.
      contexts <- c("GC", "CG")
      result <- setNames(vector("list", length(contexts)), contexts)
      for (context in contexts) {
        broad <- if (context == "GC") "NGCNN" else "NNCGN"
        filtered <- filter_context(methylation_ranges, genome, broad)
        call_matrix <- single_molecule_calls[, colnames(single_molecule_calls) %in% as.character(start(filtered)), drop = FALSE]
        if (!length(filtered) || !ncol(call_matrix)) {
          result[[context]] <- call_matrix
          next
        }
        # HCG and CG use the same collapse direction; HCG is the accepted name
        # in the package's bulk-collapse API (avoids its spurious CG warning).
        filtered <- SingleMoleculeFootprinting:::CollapseStrands(filtered, if (context == "GC") "GC" else "HCG")
        call_matrix <- SingleMoleculeFootprinting:::CollapseStrandsSM(call_matrix, context, genome, as.character(seqnames(region)))
        filtered <- SingleMoleculeFootprinting::CoverageFilter(filtered, thr = options$coverage)
        filtered <- filter_context(filtered, genome, options[[if (context == "GC") "gc-context" else "cg-context"]])
        call_matrix <- call_matrix[, colnames(call_matrix) %in% as.character(start(filtered)), drop = FALSE]
        # Explicitly order columns and keep the declared region bounds.
        absolute <- as.numeric(colnames(call_matrix))
        keep <- which(absolute >= start(region) & absolute <= end(region))
        keep <- keep[order(absolute[keep])]
        result[[context]] <- call_matrix[, keep, drop = FALSE]
      }
      result
    })
  }

  # Build the complete output in a temporary file before publishing it.
  temporary <- tempfile(pattern = paste0(basename(output), "."), tmpdir = dirname(output), fileext = ".partial")
  on.exit(unlink(temporary), add = TRUE)
  rhdf5::h5createFile(temporary)
  groups <- c("smf_mat", "smf_pos", "read_id")
  if (gc_only) groups <- c(groups, "methyl_mat", "methyl_pos", "methyl_read_id")
  for (g in groups) rhdf5::h5createGroup(temporary, g)
  handle <- rhdf5::H5Fopen(temporary)
  rhdf5::h5writeAttribute("reads_by_positions", handle, "matrix_order")
  rhdf5::h5writeAttribute("one_based_closed", handle, "region_coordinates")
  rhdf5::h5writeAttribute("first_base_zero", handle, "position_origin")
  rhdf5::h5writeAttribute(options$encoding, handle, "encoding")
  rhdf5::h5writeAttribute(options$sample, handle, "sample")
  rhdf5::h5writeAttribute(if (gc_only) "gc_only" else "gc_cg", handle, "assay_mode")
  rhdf5::h5writeAttribute(if (gc_only) "GC" else "GC+CG", handle, "accessibility_contexts")
  rhdf5::H5Fclose(handle)
  # No open HDF5 handles are inherited by fork workers. Only the parent writes.
  if (.Platform$OS.type != "unix" && options$workers > 1L) stop("Use --workers 1 on non-Unix systems")
  extract_tagged <- function(region) list(data = extract(region))
  # Skip empty channels and write positions, calls and molecule identifiers together.
  write_channel <- function(call_matrix, region, methyl = FALSE) {
    if (is.null(call_matrix) || !nrow(call_matrix) || ncol(call_matrix) < if (methyl) 1L else max(1L, options[["min-sites"]])) {
      return(FALSE)
    }
    stem <- if (methyl) "methyl" else "smf"
    name <- names(region)
    relative_positions <- as.integer(as.numeric(colnames(call_matrix)) - start(region))
    if (anyDuplicated(relative_positions)) stop("Duplicate collapsed positions: ", name)
    rhdf5::h5write(relative_positions, temporary, paste0(stem, "_pos/", name))
    values <- format_calls(call_matrix, options$encoding)
    path <- paste0(stem, "_mat/", name)
    rhdf5::h5createDataset(temporary, path,
      dims = dim(values),
      chunk = pmax(1L, pmin(dim(values), c(256L, 1024L))),
      storage.mode = "integer", H5type = "H5T_STD_I8LE", level = options$compression
    )
    rhdf5::h5write(values, temporary, path)
    rhdf5::h5write(rownames(call_matrix), temporary, paste0(if (methyl) "methyl_read_id/" else "read_id/", name))
    TRUE
  }
  # Bound memory use by extracting a limited number of regions at a time.
  n_written <- n_empty <- 0L
  start_time <- proc.time()[[3]]
  for (first in seq(1L, length(regions), by = options[["batch-size"]])) {
    region_indices <- first:min(length(regions), first + options[["batch-size"]] - 1L)
    batch <- lapply(region_indices, function(i) regions[i])
    results <- if (options$workers == 1L) {
      lapply(batch, extract_tagged)
    } else {
      parallel::mclapply(batch, extract_tagged,
        mc.cores = options$workers,
        mc.preschedule = TRUE, mc.set.seed = FALSE
      )
    }
    failed <- vapply(results, inherits, logical(1), "try-error")
    if (any(failed)) stop("Extraction worker failed: ", paste(results[failed], collapse = "\n"))
    if (any(vapply(results, is.null, logical(1)))) stop("An extraction worker exited without a result")
    results <- lapply(results, `[[`, "data")
    for (i in seq_along(batch)) {
      if (is.null(results[[i]])) {
        n_empty <- n_empty + 1L
        next
      }
      channels <- output_channels(results[[i]], gc_only)
      n_written <- n_written + write_channel(channels$smf, batch[[i]])
      if (gc_only) write_channel(channels$methyl, batch[[i]], TRUE)
    }
    cat(sprintf("%d/%d regions; %d accessibility matrices written; %d empty; %.1fs\n", max(region_indices), length(regions), n_written, n_empty, proc.time()[[3]] - start_time))
  }
  # All worker errors abort instead of silently publishing incomplete data.
  if (!file.rename(temporary, output)) stop("Could not publish output: ", output)
  cat("Wrote ", output, "\n", sep = "")
}

if (sys.nframe() == 0L) run_extraction(FALSE)
