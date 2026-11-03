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
		make_option(c("-c", "--chunk"), type="character", default=NULL, help="chromosome chunk to run (1-10)", metavar="character"),
		make_option(c("-w", "--tileWindows"), type="integer", default=20000, help="size in basepair if the genomic sliding windows", metavar="Positive number"),
		make_option(c("-p", "--pseudocount"), action="store_true", default=FALSE, help="Add a pseudocount to the contagency tables [default= %default]", metavar="logical"),
		make_option(c("-o", "--outliersFilter"), action="store_true", default=FALSE, help="Remove polarized Cs (thres = 0.1 - 0.9) [default= %default]", metavar="logical"),
		make_option(c("-b", "--bedFile"), action="store_true", default=FALSE, help="Create also a bgzip bed file for tabix easy access [default= %default]", metavar="logical"),
		make_option(c("-t", "--threads"), type="integer", default=12, help="Number of thread to use (>0)", metavar="Positive number"),
		make_option(c("-f", "--forceGenomWide"), action="store_true", default=FALSE, help="Force Genome-Wide analysis [default= %default]", metavar="logical"),
		make_option(c("-z", "--taskMeta"), type="character", default='smf_phiscore', help="Name of the current task for versioning", metavar="character")
	)
	opt_parser = OptionParser(option_list=option_list, description="Script to call methylated cytosines out of SMF-Nanopore data.")
	args <- parse_args(opt_parser)

# Loading libraries
	dep.l <- c('this.path', 'qs', 'parallel', 'psych', 'magrittr')
	for (i in dep.l) {if(!suppressMessages(require(i, character.only = TRUE))) stop(paste0("Package not found: ", i))}
	source(paste0(this.dir(), '/smf_utils.R'))

# Check the ability to work
	options(scipen=999) # avoid sci notation in the vectors
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

	## check for input files
		filePath <- gsub('//', '/', dir(args$InputFolder, pattern='.bed.gz', all.files=FALSE, full.names=TRUE, ignore.case=TRUE))
		SM_files <- data.frame(bed=filePath[grepl('.bed.gz$', filePath)], tabix=filePath[grepl('.bed.gz.tbi$', filePath)])
		if(nrow(SM_files) != 1){stop('Mulitple BED input files found...')}

	## Sample name
		EXPNAME = ifelse(
			is.null(args$expName),
			paste0(format(Sys.time(), "%Y-%m-%d_%H-%M"), '_SMF_bulk_analysis'),
			args$expName
		)

	## Chunk region
		if (is.null(args$chunk)) {
			warning('You are running the analysis on the entire Genome!\nThis may require a massive ressource usage (>14 TB memory for Drosophila genome) and days/weeks of computation...\nPlease abord if not wanted.\n\n')
			if(!args$forceGenomWide){
				print_help(opt_parser)
				stop("An argument in --chunk is required (-c 1-50).\n", call.=FALSE)
			}
		} else {
			chunkBorder <- as.numeric(unlist(strsplit(args$chunk, "-")))
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

	## Adjust output name according to input arguments
		pseudoName=NULL
		if(args$pseudocount){pseudoName='_withPseudocount'}
		filtername=NULL
		if(args$outliersFilter){filtername='_OutliersFiltered'}

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
		# subset to the chunk of interest if required
			if(exists('chunkBorder')){
				regs <- regs[seq(chunkBorder[1], chunkBorder[2])]
			}

	## output files
		if(exists('chunkBorder')){
			OutFile <- paste0('./Phiscore_', EXPNAME,'_chunk_', args$chunk, pseudoName, filtername,'.qs')
		} else {
			OutFile <- paste0('./Phiscore_', EXPNAME, '_GW_', pseudoName, filtername,'.rds')
		}

		if(args$bedFile){
			BedFile=gsub('.qs$', '.bed.gz', OutFile)
		}

	## Phiscore analysis
		VectorsToPlot <- lapply(seq_along(regs), function(i){
			range <- regs[i]
			# add sliding windows width (e.g 20kb -> 40kb)
			end(range) <- end(range) + phiscoreWin
			range <- trim(range)
			if(width(range) > phiscoreWin) {
				coordMaxToLook <- start(range) + phiscoreWin
			} else {
				coordMaxToLook <- end(range)
			}
			
			sm.mat = getCMethMatrix(SM_files=SM_files, range=range) %>% .[, colSums(!is.na(.)) > 0, drop = FALSE] # remove uninformative sites

			# Cytosine Phiscores
			if(length(colnames(sm.mat)) != 0){
				# Remove columns with too many O or 1. For Phiscore only (inducing bias at high synchronized region?) ++ ordering the column (important)
				# Filter out outliers is not in default anymore since correction of contingency table before phi-score calculation.
					if(args$outliersFilter){
						Matrix <- rm_outliers(sm.mat[, order(as.numeric(colnames(sm.mat)))], 0.1, 0.9)
					} else {
						Matrix <- sm.mat[, order(as.numeric(colnames(sm.mat)))]
					}

				# calculate the phi score
					# get only the combination for subset of Cytosines (20kb sliding win)
						cyt <- colnames(Matrix)
						cytToCal <- cyt[as.numeric(cyt) < coordMaxToLook] #viewpoints
					
					if(length(cytToCal) != 0){
						combinaisons_colonnes <- unlist(mclapply(cytToCal, function(col){
							onTarg <- which(as.numeric(cyt) < as.numeric(col) + phiscoreWin)
							compC <- cyt[seq(which(cyt == col), onTarg[length(onTarg)])]
							return(lapply(compC[-1], function(k){return(as.numeric(c(compC[1], k)))}))
						}, mc.cores=args$threads), recursive=F)

						x_vect <- unlist(mclapply(combinaisons_colonnes, function(k){k[1]}, mc.cores=args$threads)) # cumulative coordinate of the viewpoints 
						y_vect <- unlist(mclapply(combinaisons_colonnes, function(k){k[2] - k[1]}, mc.cores=args$threads)) # distance from the viewpoints 
						z_vect <- unlist(mclapply(combinaisons_colonnes, function(k){  #related phiscores/pearson from the C viewpoint and the distant one
							conT = table(Matrix[,as.character(k[1])], Matrix[,as.character(k[2])])
							
							#fixing polarized cytosine tables: (don't need to remove outliers anymore except to reduce computation time and resources)
								if(any(unique(dim(conT)) != 2)){
									#column
									if(length(colnames(conT)) == 1){
										conT = cbind(conT, x=rep(0, length(rownames(conT))))
										colnames(conT) <- c(colnames(conT)[1], ifelse(colnames(conT)[1] == '0', '1', '0'))
									}
									#row
									if(length(rownames(conT)) == 1){
										conT = rbind(conT, x=rep(0, length(colnames(conT))))
										rownames(conT) <- c(rownames(conT)[1], ifelse(rownames(conT)[1] == '0', '1', '0'))
									}
								}

							# apply a pseudocount... not in default and could be dangerous
							if(args$pseudocount){conT = conT + 1}

							return(phi(conT, digits = 3))
						}, mc.cores=args$threads))

						if(args$bedFile){
							gz1 <- gzfile(BedFile, "a")
							write.table(
								data.frame(
									seqnames=seqnames(range), 
									start=unique(x_vect), 
									end=unique(x_vect), 
									dist=tapply(y_vect, x_vect, paste, collapse=','),
									phiScores=tapply(z_vect, x_vect, paste, collapse=',')
									), 
								file=gz1, append=TRUE, sep="\t", quote=FALSE, row.names=FALSE, col.names=FALSE)
							close(gz1)
						}

						return(list(x_vect=x_vect, y_vect=y_vect, z_vect=z_vect))
					} else {
						return(list(x_vect=NULL, y_vect=NULL, z_vect=NULL))
					}
			} else {
				return(list(x_vect=NULL, y_vect=NULL, z_vect=NULL))
			}
		})
		qsave(VectorsToPlot, OutFile)

# Save Rsession info
	getVersion(args$taskMeta)

