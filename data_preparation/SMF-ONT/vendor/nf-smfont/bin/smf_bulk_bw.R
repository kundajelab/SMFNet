#!/bin/rscript

# Add the working directory in the libPath to add the user's BSgenome installation
	.libPaths( c( './', .libPaths() ) )

# Loading arguments
	if(!suppressMessages(require("optparse", character.only = TRUE))) stop("Package not found: optparse")

	option_list = list(
	## required	
		make_option(c("-g", "--genome"), type="character", default=NULL, help="Name of BSgenome to use [required]", metavar="character"),
		make_option(c("-i", "--InputQs"), type="character", default=NULL, help="Path to .qs file containing GRanges object with the SMF signal", metavar="/path/to/BulkSMF.qs"),

	## optional
		make_option(c("-e", "--expName"), type="character", default=NULL, help="Name of the SMF-ONT experiment", metavar="character"),
		make_option(c("-m", "--mode"), type="character", default='cpg,gpc', help="Vector (comma separated) of methylation context to analysis", metavar="character"),
		make_option(c("-k", "--keepMeth"), type="logical", default=FALSE, help="DON'T invert methylation signal in the output files (default: FALSE)", metavar="boolean"),
		make_option(c("-t", "--threads"), type="integer", default=12, help="Number of thread to use (>0)", metavar="number"),
		make_option(c("-c", "--minCoverage"), type="integer", default=20, help="Minimal number of read over Cs to measure methylation rate", metavar="number"),
		make_option(c("-z", "--taskMeta"), type="character", default='smf_bulk_bw', help="Name of the current task for versioning", metavar="character")
	)
	opt_parser = OptionParser(option_list=option_list, description="Script to call methylated cytosines out of SMF-Nanopore data.")
	args <- parse_args(opt_parser)

# Loading libraries
	dep.l <- c('this.path', 'qs', 'magrittr', 'tools', 'parallel')
	for (i in dep.l) {if(!suppressMessages(require(i, character.only = TRUE))) stop(paste0("Package not found: ", i))}
	source(paste0(this.dir(), '/smf_utils.R'))

# Check the ability to work
	## Genome loading
		if (is.null(args$genome)) {
			print_help(opt_parser)
			stop("An argument in --genome is required (-g BSgenome.Mmusculus.UCSC.mm10 ).\n", call.=FALSE)
		}

	## SMF data
		if (is.null(args$InputQs)) {
			print_help(opt_parser)
			stop("An argument in --InputQs is required (-i BulkSMF_XXX_cpggpc.qs ).\n", call.=FALSE)
			if(!file.exists(args$InputQs)){
				print_help(opt_parser)
				stop("The file provided --InputQs doesn't exist (-i BulkSMF_XXX_cpggpc.qs ).\n", call.=FALSE)
			}
		}

	## Sample name
		EXPNAME = ifelse(
			is.null(args$expName),
			paste0(format(Sys.time(), "%Y-%m-%d_%H-%M"), '_SMF_bulk_analysis'),
			args$expName
		)

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

	## Minimum coverage arg
		if (!is.numeric(args$minCoverage)) {
			print_help(opt_parser)
			stop("The argument in --minCoverage is not numeric (-c 20).\n", call.=FALSE)
		} else if(args$minCoverage < 1) {
			print_help(opt_parser)
			stop("The argument in --minCoverage cannot be lower than 1 (-c 20).\n", call.=FALSE)
		}

# Load data
	if(args$minCoverage > 1) {
		AllC <- qread(args$InputQs) %>% CovFilter(., args$minCoverage)
	} else {
		AllC <- qread(args$InputQs)
	}

	subsetIndex=list(
		GCi = AllC$type == 'GCH',
		CGi = AllC$type == 'HCG',
		GCGi = AllC$type == 'GCG',
		Ai = AllC$type == 'A'
	)

# MAIN: Make WIG file for visualisation
	FILTER <- colnames(mcols(AllC)) %>%
		grep('.Met_rate',., value=T) %>% gsub('.Met_rate', '',.) %>%
		strsplit(., '_') %>% 
		lapply(., function(i){
			filterN = grep("meth\\w+iltered", i, fixed=F, value=T)
			filterHeader <- ifelse(filterN == 'methFiltered',
				paste(i[grep(filterN, i):(grep('max', i)+1)], collapse='_'),
				filterN
			)
		}) %>% unlist()

	Outdir <- paste0(EXPNAME, '_BW_minCov_', args$minCoverage,'/')
	if(!dir.exists(Outdir)) { dir.create(Outdir) }

	if(args$keepMeth) { suffix = '_meth_' } else { suffix = '_invertedMeth_' }

	trash <- mclapply(seq_along(FILTER), function(i){
		exp <- paste0(EXPNAME, '_', FILTER[i])
		data <- AllC[,grepl('.Met_rate', colnames(mcols(AllC)))]

		fileName <- paste0(Outdir, 'GW_SMF_', contextView, suffix, exp, '_minCov_', args$minCoverage, '.bw')
		exportBW(gR=data, feature=colnames(mcols(data[,grepl(FILTER[i], colnames(mcols(data)))])), 
			expName=exp, filename=fileName, vpPos=5727525, vpChr='chr2R', plotView=50000, BSgenome=args$genome, fixed=FALSE, invert=!args$keepMeth)

		if(length(motifs) > 1){
			trash <- lapply(motifs, function(mot){
				fileName <- paste0(Outdir, 'GW_SMF_', mot, suffix, exp, '_minCov_', args$minCoverage, '.bw')
				if(mot == 'cpg'){
					indexToKeep <- subsetIndex$CGi
				} else if(mot == 'gpc'){
					indexToKeep <- subsetIndex$GCi
				} else if(mot == 'a'){
					indexToKeep <- subsetIndex$Ai
				} else  if(mot == 'c'){
					indexToKeep <- Reduce(`|`, list(subsetIndex$CGi, subsetIndex$GCi, subsetIndex$GCGi))
				}
				subData = data[indexToKeep,]
				exportBW(gR=subData, feature=colnames(mcols(subData[,grepl(FILTER[i], colnames(mcols(subData)))])), 
					expName=exp, filename=fileName, vpPos=5727525, vpChr='chr2R', plotView=50000, BSgenome=args$genome, fixed=FALSE, invert=!args$keepMeth)
			})
		}
	}, mc.cores=args$threads)

# Save Rsession info
	getVersion(args$taskMeta)
