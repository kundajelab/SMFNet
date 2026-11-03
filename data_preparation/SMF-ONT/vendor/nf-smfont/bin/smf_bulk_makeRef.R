#!/bin/rscript

# Add the working directory in the libPath to add the user's BSgenome installation
	.libPaths( c( './', .libPaths() ) )

# Loading arguments
	if(!suppressMessages(require("optparse", character.only = TRUE))) stop("Package not found: optparse")

	option_list = list(
	## required	
		make_option(c("-g", "--genome"), type="character", default=NULL, help="Name of BSgenome to use [required]", metavar="character"),

	## optional
		make_option(c("-m", "--mode"), type="character", default='cpg,gpc', help="vector (comma separated) of methylation context to analysis", metavar="character"),
		make_option(c("-t", "--threads"), type="integer", default=12, help="Number of thread to use (>0)", metavar="number"),
		make_option(c("-z", "--taskMeta"), type="character", default='smf_bulk_makeRef', help="Name of the current task for versioning", metavar="character")
	)
	opt_parser = OptionParser(option_list=option_list, description="Script to call methylated cytosines out of SMF-Nanopore data.")
	args <- parse_args(opt_parser)

# Loading libraries
	dep.l <- c('this.path', 'qs', 'BSgenome')
	for (i in dep.l) {if(!suppressMessages(require(i, character.only = TRUE))) stop(paste0("Package not found: ", i))}
	source(paste0(this.dir(), '/smf_utils.R'))

# Check the ability to work
	## Genome loading
		if (is.null(args$genome)) {
			print_help(opt_parser)
			stop("An argument in --genome is required (-g BSgenome.Mmusculus.UCSC.mm10 ).\n", call.=FALSE)
		} else {
			if(!suppressMessages(require(args$genome, character.only = TRUE))) stop(paste0("Package not found: ", args$genome))
			GENOME=get(args$genome)

			# Extract std chromosomes from the loaded genome
			tiles <- keepStandardChromosomes(GRanges(seqinfo(GENOME)), pruning.mode ='coarse')
			norm_chr = as.character(seqnames(tiles))

			#testGenome <- unlist(strsplit(args$genome, "[.]"))
		}

	## methylation motifs
		motifs <- unlist(strsplit(tolower(args$mode), ','))
		motifs <- unlist(lapply(motifs, function(mot){
			if(!tolower(mot) %in% c('cpg', 'gpc', 'a', 'cg', 'gc', 'c')){
				print_help(opt_parser)
				stop(paste0('The argument "', mot,'" in --mode is supported (-m cpg,gpc,a,c).\n'), call.=FALSE)
			}
			if(tolower(mot) == 'cg'){
				return('cpg')
			} else if(tolower(mot) == 'gc'){
				return('gpc')
			} else {
				return(mot)
			}
		}))
		
		if(length(motifs) == 1){
			contextView = motifs
		} else {
			if(length(motifs) == 3){
				contextView='cpggpca'
			} else {
				if(length(grep('a', motifs)) == 1){
					contextView=paste0(motifs[-grep('a', motifs)], 'a')
				} else {
					contextView='cpggpc'
				}
			}
		}

	## thread arg
		if (!is.numeric(args$threads)) {
			print_help(opt_parser)
			stop("The argument in --threads is not numeric (-t 10).\n", call.=FALSE)
		} else if(args$threads < 1) {
			print_help(opt_parser)
			stop("The argument in --threads is lower than 1 (-t 10).\n", call.=FALSE)
		}

# MAIN: create reference GRnages object for methylated sites
	ref_file = paste0('context_methylation_ref_', args$genome, '_', contextView,'.qs')
	AllC <- AllSitegr(tiles=tiles, BSgenome=GENOME, mode=motifs, threads=args$threads)
	qsave(AllC, ref_file)

# Save Rsession info
	getVersion(args$taskMeta)
