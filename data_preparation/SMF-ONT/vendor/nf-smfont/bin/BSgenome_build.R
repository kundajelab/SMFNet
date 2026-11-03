#!/bin/rscript

# Loading arguments
	if(!suppressMessages(require("optparse", character.only = TRUE))) stop("Package not found: optparse")

	option_list = list(
		make_option(c("-g", "--genome"), type="character", default=NULL, help="Name of BSgenome to use [required]", metavar="character"),
		make_option(c("-z", "--taskMeta"), type="character", default='BSgenome_build', help="Name of the current task for versioning", metavar="character")
	)
	opt_parser = OptionParser(option_list=option_list, description="Script to call methylated cytosines out of SMF-Nanopore data.")
	args <- parse_args(opt_parser)

# Loading libraries
	dep.l = c('this.path', 'BSgenome', 'BiocManager', 'BiocVersion')
    for (i in dep.l) {if(!suppressMessages(require(i, character.only = TRUE))) stop(paste0("Package not found: ", i))}
    source(paste0(this.dir(), '/smf_utils.R'))

# MAIN: Create GRanges object containing all methylation calls of the experiment
    if (args$genome %in% available.genomes()){
        .libPaths(c('./', .libPaths()))
        BiocManager::install(args$genome, ask = FALSE, update=FALSE)
    } else {
        stop(paste0("The BSgenome: ", args$genome, " is not available!"))
    }

# Save Rsession info
	getVersion(args$taskMeta)
