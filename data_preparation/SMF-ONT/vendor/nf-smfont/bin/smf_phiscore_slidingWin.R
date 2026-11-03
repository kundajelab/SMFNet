#!/bin/rscript

# Add the working directory in the libPath to add the user's BSgenome installation
	.libPaths( c( './', .libPaths() ) )

# Loading arguments
	if(!suppressMessages(require("optparse", character.only = TRUE))) stop("Package not found: optparse")

	option_list = list(
	## required	
		make_option(c("-g", "--genome"), type="character", default=NULL, help="Name of BSgenome to use [required]", metavar="character"),
		make_option(c("-w", "--tileWindows"), type="integer", default=20000, help="size in basepair if the genomic sliding windows", metavar="Positive number"),
		make_option(c("-t", "--threads"), type="integer", default=12, help="Number of thread to use (>0)", metavar="Positive number"),
		make_option(c("-z", "--taskMeta"), type="character", default='smf_phiscore_slidingWin', help="Name of the current task for versioning", metavar="character")
	)
	opt_parser = OptionParser(option_list=option_list, description="Script to call methylated cytosines out of SMF-Nanopore data.")
	args <- parse_args(opt_parser)

# Loading libraries
	dep.l <- c('this.path')
	for (i in dep.l) {if(!suppressMessages(require(i, character.only = TRUE))) stop(paste0("Package not found: ", i))}
	source(paste0(this.dir(), '/smf_utils.R'))

# Check the ability to work
	## Genome loading
		if (is.null(args$genome)) {
			print_help(opt_parser)
			stop("An argument in --genome is required (-g BSgenome.Mmusculus.UCSC.mm10 ).\n", call.=FALSE)
		} else {
			if(!suppressMessages(require("BSgenome", character.only = TRUE))) stop("Package not found: BSgenome")
			if(!suppressMessages(require(args$genome, character.only = TRUE))) stop(paste0("Package not found: ", args$genome))
			GENOME=get(args$genome)

			# Extract std chromosomes from the loaded genome
			tiles <- keepStandardChromosomes(GRanges(seqinfo(GENOME)), pruning.mode ='coarse')
			norm_chr = as.character(seqnames(tiles))
		}

	## tile width
		if (!is.numeric(args$tileWindows)) {
			print_help(opt_parser)
			stop("The argument in --tileWindows is not numeric (-w 10).\n", call.=FALSE)
		} else if(args$tileWindows < 1) {
			print_help(opt_parser)
			stop("The argument in --tileWindows is lower than 1 (-w 10).\n", call.=FALSE)
		} else {
			phiscoreWin <- args$tileWindows
		}

	## thread
		if (!is.numeric(args$threads)) {
			print_help(opt_parser)
			stop("The argument in --threads is not numeric (-t 10).\n", call.=FALSE)
		} else if(args$threads < 1) {
			print_help(opt_parser)
			stop("The argument in --threads is lower than 1 (-t 10).\n", call.=FALSE)
		}

# MAIN
	## genome tiles
		regs <- tileGenome(seqlengths(GENOME)[as.character(seqnames(tiles))], tilewidth=phiscoreWin, cut.last.tile.in.chrom=TRUE)
		seqinfo(regs) <- seqinfo(GENOME)

	## chunk the tiles (max 1000 jobs)
		nJob = 1000
		chunkNum <- ifelse(length(regs) / 10 > nJob, ifelse(length(regs) / 20 > nJob,  round(length(regs)/nJob), 20), 10)
		chunks <- paste(seq(1,length(regs), chunkNum), c(seq(1,length(regs), chunkNum)[-1]-1, length(regs)), sep='-')

		write.table(chunks, file='./chunk.txt', row.names=F, col.names=F, quote=F)
		write.table(as.character(seqnames(tiles)), file='./chr.txt', row.names=F, col.names=F, quote=F)

# Save Rsession info
	getVersion(args$taskMeta)
