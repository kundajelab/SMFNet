#!/bin/rscript

# Add the working directory in the libPath to add the user's BSgenome installation
	.libPaths( c( './', .libPaths() ) )

# Loading arguments
	if(!suppressMessages(require("optparse", character.only = TRUE))) stop("Package not found: optparse")

	option_list = list(
	## required
		make_option(c("-p", "--pairsFile"), type="character", default=NULL, help="Path to the feature pair object (.qs), output of smf_cooc_makePairs.R [required].", metavar="/path/to/pairs.qs"),
		make_option(c("-a", "--smvecA"), type="character", default=NULL, help="Path to the single-molecule sorted vector (.qs) for the reference feature nature [required].", metavar="/path/to/SMvecA.qs"),
		make_option(c("-x", "--featureNature"), type="character", default="TF", help="Nature of the pair: TF, TSS, DHS, TF-TSS, TSS-TF, DHS-TSS or TSS-DHS [default= %default]", metavar="character"),

	## optional
		make_option(c("-b", "--smvecB"), type="character", default=NULL, help="Path to the single-molecule sorted vector (.qs) for the linked feature nature.\nDefaults to --smvecA for homotypic pairs (TF, TSS or DHS).", metavar="/path/to/SMvecB.qs"),
		make_option(c("-m", "--mode"), type="character", default="single", help="Calling mode(s) used at the features, comma-sep and ordered as in --featureNature (e.g. std,hybrid for TSS-TF).\nOnly the TSS mode (std|downstream) impacts the state declaration.", metavar="character"),
		make_option(c("-e", "--expName"), type="character", default=NULL, help="Name of the experiment", metavar="character"),
		make_option(c("-c", "--minCov"), type="integer", default=10, help="Minimal number of single molecules covering both features of a pair to test it [default= %default]", metavar="Positive number"),
		make_option(c("-w", "--overwrite"), action="store_true", default=FALSE, help="Overwrite existing contingency file [default= %default]"),
		make_option(c("-t", "--threads"), type="integer", default=12, help="Number of thread to use (>0)", metavar="Positive number"),
		make_option(c("-z", "--taskMeta"), type="character", default='smf_cooc', help="Name of the current task for versioning", metavar="character")
	)
	opt_parser = OptionParser(option_list=option_list, description="Script to compute co-occupancy/co-accessibility contingency tables (Fisher's exact test) for a set of feature pairs.")
	args <- parse_args(opt_parser)

# Loading libraries
	dep.l <- c('this.path', 'qs', 'parallel', 'GenomicRanges', 'magrittr')
	for (i in dep.l) {if(!suppressMessages(require(i, character.only = TRUE))) stop(paste0("Package not found: ", i))}
	source(paste0(this.dir(), '/smf_utils.R'))

# Check the ability to work
	## Pairs input
		if(is.null(args$pairsFile)){
			print_help(opt_parser)
			stop("An argument in --pairsFile is required (-p path/to/pairs.qs).\n", call.=FALSE)
		} else if(!file.exists(args$pairsFile)){
			print_help(opt_parser)
			stop("The file in --pairsFile does not exist (-p path/to/pairs.qs).\n", call.=FALSE)
		}

	## Single-molecule vector input(s)
		if(is.null(args$smvecA)){
			print_help(opt_parser)
			stop("An argument in --smvecA is required (-a path/to/SMvecA.qs).\n", call.=FALSE)
		} else if(!file.exists(args$smvecA)){
			print_help(opt_parser)
			stop("The file in --smvecA does not exist (-a path/to/SMvecA.qs).\n", call.=FALSE)
		}

		if(is.null(args$smvecB)){
			args$smvecB <- args$smvecA
		} else if(!file.exists(args$smvecB)){
			print_help(opt_parser)
			stop("The file in --smvecB does not exist (-b path/to/SMvecB.qs).\n", call.=FALSE)
		}

	## Feature nature / pair type
		supportedNat <- c('tf', 'tss', 'dhs', 'tf-tss', 'tss-tf', 'dhs-tss', 'tss-dhs')
		if(!tolower(args$featureNature) %in% supportedNat){
			print_help(opt_parser)
			stop("The --featureNature argument is not supported (-x TF, -x TSS, -x DHS, -x TSS-TF or -x TSS-DHS).\n", call.=FALSE)
		}

		if(tolower(args$featureNature) %in% c('tf', 'tss', 'dhs')){
			pairType <- rep(toupper(args$featureNature), 2)
		} else {
			pairType <- unlist(strsplit(toupper(args$featureNature), '-'))
		}
		pairTypeStr <- ifelse(pairType[1] == pairType[2], pairType[1], paste(pairType, collapse='-'))

	## Mode(s) (only used to pick the TSS state-declaration flavor)
		modeVec <- unlist(strsplit(args$mode, ','))
		stateMode <- list()
		if('TSS' %in% pairType){
			tssMode <- modeVec[tolower(modeVec) %in% c('std', 'downstream')]
			if(length(tssMode) == 0){
				print_help(opt_parser)
				stop("A TSS mode (std or downstream) is required in --mode when --featureNature involves TSS.\n", call.=FALSE)
			}
			stateMode[['TSS']] <- tssMode[1]
		}

	## Minimal coverage
		if (!is.numeric(args$minCov)) {
			print_help(opt_parser)
			stop("The argument in --minCov is not numeric (-c 10).\n", call.=FALSE)
		} else if(args$minCov < 1) {
			print_help(opt_parser)
			stop("The argument in --minCov is lower than 1 (-c 10).\n", call.=FALSE)
		}

	## thread
		if (!is.numeric(args$threads)) {
			print_help(opt_parser)
			stop("The argument in --threads is not numeric (-t 10).\n", call.=FALSE)
		} else if(args$threads < 1) {
			print_help(opt_parser)
			stop("The argument in --threads is lower than 1 (-t 10).\n", call.=FALSE)
		}

	## Sample name
		EXPNAME = ifelse(
			is.null(args$expName),
			paste0(format(Sys.time(), "%Y-%m-%d_%H-%M"), '_SMF_coop_analysis'),
			args$expName
		)

# MAIN:
	cat(paste0("\nComputing ", pairTypeStr, " contingency tables (min single-molecule coverage = ", args$minCov, ")...\n"))

	fileCont <- paste0(EXPNAME, '_', gsub('-', '_', tolower(args$featureNature)), '_contingency_minCov', args$minCov, '.qs')

	if(!file.exists(fileCont) | args$overwrite){
		# Load the feature pairs
			Pairs <- qread(args$pairsFile)
			TFBSpair <- Pairs$TFBSpair
			rm(Pairs)
			if(length(TFBSpair) == 0){stop("No feature pairs found in --pairsFile.\n")}

		# Load the single-molecule vectors
			smA <- qread(args$smvecA)
			smB <- if(args$smvecB == args$smvecA){smA}else{qread(args$smvecB)}

		# Declare the combined biological states for this pair nature
			TF_statesL <- MakePairStates(pairType=pairType, mode=stateMode)
			TF_TF_states <- TF_statesL[['TF_TF']]
			order <- getReadOrder(featureNature=tolower(args$featureNature))
			TF_TF_states <- TF_TF_states[order]

		# Compute the per-pair contingency tables / Fisher tests
			conT.L <- computePairContingency(
				TFBSpair=TFBSpair, smA=smA, smB=smB, TF_TF_states=TF_TF_states,
				pairType=pairTypeStr, minCov=args$minCov, threads=args$threads
			)

			if(length(conT.L) == 0){
				cat("\tNo pair passed the minimum coverage filter... Saving an empty contingency object.\n")
			} else {
				cat(paste0("\t", length(conT.L), " pairs tested.\n"))
			}

		# Adjust p-values (BH) per contingency comparison type
			conT.L <- adjustContingencyPvalues(conT.L, threads=args$threads)

		attr(conT.L, 'featureNature') <- tolower(args$featureNature)
		attr(conT.L, 'pairType') <- pairTypeStr
		attr(conT.L, 'minCov') <- args$minCov
		attr(conT.L, 'expName') <- EXPNAME

		qsave(conT.L, fileCont)
	} else {
		stop(paste0(fileCont, ' already exists... Use -w argument to overwrite.'))
	}

# Save Rsession info
	getVersion(args$taskMeta)
