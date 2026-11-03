#!/bin/rscript

# Add the working directory in the libPath to add the user's BSgenome installation
	.libPaths( c( './', .libPaths() ) )

# Loading arguments
	if(!suppressMessages(require("optparse", character.only = TRUE))) stop("Package not found: optparse")

	option_list = list(
	## required	
		make_option(c("-g", "--genome"), type="character", default=NULL, help="Name of BSgenome to use [required]", metavar="character"),

	## optional
		make_option(c("-i", "--InputFolder"), type="character", default='./', help="Path to folder containing all bed.gz / tbi files", metavar="/path/to/MethycallFolder/"),
		make_option(c("-e", "--expName"), type="character", default=NULL, help="Name of the SMF-ONT experiment", metavar="character"),
		make_option(c("-c", "--chr"), type="character", default=NULL, help="Name of the chromosome to merge related chunks", metavar="character"),
		make_option(c("-w", "--tileWindows"), type="integer", default=20000, help="size in basepair if the genomic sliding windows", metavar="Positive number"),
		make_option(c("-t", "--threads"), type="integer", default=12, help="Number of thread to use (>0)", metavar="Positive number"),
		make_option(c("-z", "--taskMeta"), type="character", default='smf_phiscore_merge', help="Name of the current task for versioning", metavar="character")
	)
	opt_parser = OptionParser(option_list=option_list, description="Script to call methylated cytosines out of SMF-Nanopore data.")
	args <- parse_args(opt_parser)

# Loading libraries
	dep.l <- c('this.path', 'qs', 'parallel', 'gtools', 'magrittr')
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

	## Input folder
		if (!dir.exists(args$InputFolder)) {
			print_help(opt_parser)
			stop("The folder in --InputFolder does not exist (-i /path/to/MethycallFolder/).\n", call.=FALSE)
		}

	## Sample name
		EXPNAME = ifelse(
			is.null(args$expName),
			paste0(format(Sys.time(), "%Y-%m-%d_%H-%M"), '_SMF_bulk_analysis'),
			args$expName
		)

	## check subest chr
		if(is.null(args$chr)){
			all_chr=TRUE
		} else {
			all_chr=FALSE
			if(!args$chr %in% seqnames(tiles)){stop('the chromosome name provided is not present in the genome!')}
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

	## Merge chunk files per chr
		cat(paste0('Identifying chunk size and chunk ', ifelse(all_chr, 'per chr', paste0('for the chr ', args$chr)) ,'...\n'))
		fileList <- gsub('//','/',list.files(args$InputFolder, pattern=paste0('Phiscore_', EXPNAME, '_chunk_?(\\d+)-?(\\d+).qs'), full.names=T))

		chunkName <- unlist(strsplit(unlist(strsplit(basename(fileList[1]), '[.]'))[1], '_'))
		sizeChunk <- as.numeric(unlist(strsplit(chunkName[length(chunkName)], '-'))[2]) - as.numeric(unlist(strsplit(chunkName[length(chunkName)], '-'))[1]) +1

		startChunks <- seq(1,length(regs), sizeChunk)

		regs.L <- lapply(as.character(seqnames(tiles)), function(i){regs[seqnames(regs) == i]})
		names(regs.L) <- as.character(seqnames(tiles))
		chunkBorderABS <- unlist(lapply(regs.L, function(k){paste0('1-', length(k))}))
		cumChunk <- 0
		chunkBorder <- c()
		for (k in seq_along(regs.L)) {
			sub <- regs.L[[k]]
			chunkBorder <- c(chunkBorder, paste0(cumChunk+1, '-', cumChunk+length(sub)))
			cumChunk <- cumChunk + length(sub)
		}
		names(chunkBorder) <- names(regs.L)

		if(!all_chr){
			regs.L <- regs.L[args$chr]
		}

	## merging per chromosome
		cat('Merging chunk files per chromosome...\n')
		for (i in seq_along(regs.L)) {
			chrN <- names(regs.L[i])
			sub <- regs.L[[i]]
			outFile <- paste0(args$InputFolder, 'Phiscore_', EXPNAME, '_' , chrN, '.qs')
			if(!file.exists(outFile)){
				cat(paste0('Processing the ', chrN, '...\n'))
				chunks <- unlist(strsplit(chunkBorder[[chrN]], '-'))
				ABSchunks <- unlist(strsplit(chunkBorderABS[[chrN]], '-'))
				toKeepStart <- unlist(lapply(chunks, function(k){startChunks[findInterval(k, startChunks)]}))
				subFileList <- mixedsort(grep(paste(paste0('_', seq(toKeepStart[1],toKeepStart[2], sizeChunk), '-'), collapse='|'), fileList, value=T))

				output <- unlist(mclapply(subFileList, qread, mc.cores=args$threads), recursive=FALSE)
				if(toKeepStart[1] == as.numeric(chunks[1])){
					output <- output[seq(ABSchunks[1], ABSchunks[2], 1)] #subset only for chunk inside the given chr
				} else {
					output <- output[seq(as.numeric(chunks[1])-toKeepStart[1]+1, as.numeric(chunks[2])-toKeepStart[1]+1, 1)] #subset only for chunk inside the given chr
				}
			
				out <- lapply(c("x_vect", "y_vect", "z_vect"), function(e){unlist(mclapply(output, function(k){k[[e]]}, mc.cores=args$threads))})
				names(out) <- c("x_vect", "y_vect", "z_vect")
				rm(output)

				qsave(out, outFile)
				rm(out)
			} else {
				cat(paste0('The ', chrN, ' already processed. Skipping...\n'))
			}
		}

# Save Rsession info
	getVersion(args$taskMeta)
