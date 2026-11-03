#!/bin/rscript

# Add the working directory in the libPath to add the user's BSgenome installation
	.libPaths( c( './', .libPaths() ) )

# Loading arguments
	if(!suppressMessages(require("optparse", character.only = TRUE))) stop("Package not found: optparse")

	option_list = list(
	## required
		make_option(c("-p", "--pairsFile"), type="character", default=NULL, help="Path to the feature pair object (.qs), output of smf_cooc_makePairs.R [required].", metavar="/path/to/pairs.qs"),
		make_option(c("-i", "--inputCont"), type="character", default=NULL, help="Path to the contingency object (.qs) for the same comparison, output of smf_cooc.R [required].", metavar="/path/to/contingency.qs"),
		make_option(c("-x", "--featureNature"), type="character", default="TF", help="Nature of the pair: TF, TSS, DHS, TSS-TF or TSS-DHS [default= %default]", metavar="character"),

	## optional
		make_option(c("-e", "--expName"), type="character", default=NULL, help="Name of the experiment/comparison", metavar="character"),
		make_option(c("-s", "--sigThreshold"), type="double", default=0.05, help="Adjusted p-value threshold to call a pair significant [default= %default]", metavar="numeric"),
		make_option(c("-c", "--minCovAdjust"), type="integer", default=0, help="Minimal cov.adjust value to keep a pair in the master table [default= %default]", metavar="Positive number"),
		make_option(c("-l", "--minAccFreq"), type="double", default=10, help="Minimal DHS accessibility frequency (%%) to keep a DHS feature in a pair, only applied when --featureNature involves DHS [default= %default]", metavar="numeric"),
		make_option(c("-u", "--maxAccFreq"), type="double", default=90, help="Maximal DHS accessibility frequency (%%) to keep a DHS feature in a pair, only applied when --featureNature involves DHS [default= %default]", metavar="numeric"),
		make_option(c("-w", "--overwrite"), action="store_true", default=FALSE, help="Overwrite existing master table file [default= %default]"),
		make_option(c("-t", "--threads"), type="integer", default=12, help="Number of thread to use (>0)", metavar="Positive number"),
		make_option(c("-z", "--taskMeta"), type="character", default='smf_cooc_masterTable', help="Name of the current task for versioning", metavar="character")
	)
	opt_parser = OptionParser(option_list=option_list, description="Script to fold a co-occupancy/co-accessibility contingency object into a single master table (.qs), with feature-nature specific filtering (e.g. 10-90% accessibility for DHS).")
	args <- parse_args(opt_parser)

# Loading libraries
	dep.l <- c('this.path', 'qs', 'parallel', 'GenomicRanges', 'dplyr', 'tibble', 'magrittr')
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

	## Contingency input
		if(is.null(args$inputCont)){
			print_help(opt_parser)
			stop("An argument in --inputCont is required (-i path/to/contingency.qs).\n", call.=FALSE)
		} else if(!file.exists(args$inputCont)){
			print_help(opt_parser)
			stop("The file in --inputCont does not exist (-i path/to/contingency.qs).\n", call.=FALSE)
		}

	## Feature nature / pair type
		supportedNat <- c('tf', 'tss', 'dhs', 'tf-tss', 'tss-tf', 'dhs-tss', 'tss-dhs')
		if(!tolower(args$featureNature) %in% supportedNat){
			print_help(opt_parser)
			stop("The --featureNature argument is not supported (-x TF, -x TSS, -x DHS, -x TSS-TF or -x TSS-DHS).\n", call.=FALSE)
		}
		involvesDHS <- grepl('dhs', tolower(args$featureNature))

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
			paste0(format(Sys.time(), "%Y-%m-%d_%H-%M"), '_SMF_coop_masterTable'),
			args$expName
		)

# MAIN:
	fileMaster <- paste0(EXPNAME, '_', gsub('-', '_', tolower(args$featureNature)), '_masterTable.qs')

	if(!file.exists(fileMaster) | args$overwrite){
		cat(paste0("\nBuilding the ", toupper(args$featureNature), " master table...\n"))

		# Load the feature pairs (carries per-site freq used for the DHS accessibility filter, and strand for orientation)
			Pairs <- qread(args$pairsFile)
			TFBSpair <- Pairs$TFBSpair
			rm(Pairs)

			pairLookup <- tibble(
				PairID = TFBSpair$pairID,
				strandA = as.character(strand(TFBSpair)),
				strandB = TFBSpair$linkSite.strand
			)

		# Load the contingency object
			conT.L <- qread(args$inputCont)
			if(length(conT.L) == 0){
				cat("\tNo pair available in the contingency object... Saving an empty master table.\n")
				data <- tibble()
			} else {
				comNames <- c('coor', 'Pair_name', 'PairID')
				compTypes <- names(conT.L[[1]])
				compTypes <- compTypes[!compTypes %in% comNames]

				# Flatten every comparison type of the contingency object into a long tibble
					data <- bind_rows(mclapply(compTypes, function(type){
						dist <- unlist(lapply(conT.L, function(i){
							splitD <- as.numeric(unlist(strsplit(unlist(strsplit(i$coor, ':'))[2], '-')))
							abs(splitD[1] - splitD[2])
						}))
						tibble(
							analysis = type,
							PairID = unlist(lapply(conT.L, function(i){i$PairID})),
							Pair_name = unlist(lapply(conT.L, function(i){i$Pair_name})),
							coor = unlist(lapply(conT.L, function(i){i$coor})),
							dist = dist,
							odd = unlist(lapply(conT.L, function(i){as.numeric(i[[type]]$odd)})),
							score = unlist(lapply(conT.L, function(i){as.numeric(i[[type]]$score)})),
							pvalue = unlist(lapply(conT.L, function(i){as.numeric(i[[type]]$pvalue)})),
							p.adjust = unlist(lapply(conT.L, function(i){as.numeric(i[[type]]$p.adjust)})),
							conf.int = lapply(conT.L, function(i){i[[type]]$conf.int}),
							cont.l = lapply(conT.L, function(i){i[[type]]$cont}),
							anc1.freq = unlist(lapply(conT.L, function(i){100 * i[[type]]$cont[1, 1] / sum(i[[type]]$cont[, 1])})),
							anc2.freq = unlist(lapply(conT.L, function(i){100 * i[[type]]$cont[1, 1] / sum(i[[type]]$cont[1, ])})),
							cov = unlist(lapply(conT.L, function(i){as.numeric(i[[type]]$cov)})),
							cov.adjust = unlist(lapply(conT.L, function(i){as.numeric(i[[type]]$cov.adjust)})),
							Clmin = unlist(lapply(conT.L, function(i){as.numeric(i[[type]]$conf.int[1])})),
							Clmax = unlist(lapply(conT.L, function(i){as.numeric(i[[type]]$conf.int[2])}))
						)
					}, mc.cores=args$threads))

				# Pair naming / nature (homotypic vs heterotypic) and genomic orientation
					anc <- dplyr::bind_rows(mclapply(strsplit(data$Pair_name, ':'), function(i){
						tibble(Pair_name.anc1 = i[1], Pair_name.anc2 = i[2])
					}, mc.cores=args$threads))
					data <- dplyr::bind_cols(data, anc)
					data$pair.nature <- ifelse(data$Pair_name.anc1 == data$Pair_name.anc2, 'homotypic', 'heterotypic')
					data$pair.nature <- factor(data$pair.nature, levels=c('homotypic', 'heterotypic'))

					data <- dplyr::left_join(data, pairLookup, by='PairID')
					data$orientation <- mapply(function(a, b){
						if(is.na(a) || is.na(b) || a == '*' || b == '*'){return(NA_character_)}
						if(a == b){return(ifelse(a == '+', 'forward', 'reverse'))}
						return(ifelse(paste0(a, b) == '+-', 'convergent', 'divergent'))
					}, data$strandA, data$strandB)
					data$orientation <- factor(data$orientation, levels=c('forward', 'reverse', 'convergent', 'divergent'))
					data$strandA <- NULL
					data$strandB <- NULL

				# Significance call and expName
					data$signif <- ifelse(is.na(data$p.adjust), 'untested', ifelse(data$p.adjust < args$sigThreshold, 'significant', 'non significant'))
					data$signif <- factor(data$signif, levels=c('non significant', 'significant', 'untested'))
					data$expName <- EXPNAME

				# Coverage filter (generic across feature natures)
					if(args$minCovAdjust > 0){
						nBefore <- nrow(data)
						data <- dplyr::filter(data, cov.adjust >= args$minCovAdjust)
						cat(paste0("\tCoverage filter (cov.adjust >= ", args$minCovAdjust, "): ", nBefore, " -> ", nrow(data), " rows.\n"))
					}

				# DHS-specific accessibility filter (10-90% by default): drop pairs where either DHS feature
				# of the pair is outside the [minAccFreq, maxAccFreq) accessibility range.
					if(involvesDHS){
						nBefore <- nrow(data)
						isDHS1 <- grepl('^DHS', data$Pair_name.anc1, ignore.case = TRUE)
						isDHS2 <- grepl('^DHS', data$Pair_name.anc2, ignore.case = TRUE)
						validFreq1 <- !is.na(data$anc1.freq) & data$anc1.freq >= args$minAccFreq & data$anc1.freq < args$maxAccFreq
						validFreq2 <- !is.na(data$anc2.freq) & data$anc2.freq >= args$minAccFreq & data$anc2.freq < args$maxAccFreq
						keep <- (!isDHS1 | validFreq1) & (!isDHS2 | validFreq2)
						data <- data[keep, ]
						cat(paste0("\tDHS accessibility filter (", args$minAccFreq, "-", args$maxAccFreq, "%): ", nBefore, " -> ", nrow(data), " rows.\n"))
					}
			}

		attr(data, 'featureNature') <- tolower(args$featureNature)
		attr(data, 'expName') <- EXPNAME
		attr(data, 'sigThreshold') <- args$sigThreshold
		attr(data, 'minCovAdjust') <- args$minCovAdjust
		if(involvesDHS){
			attr(data, 'minAccFreq') <- args$minAccFreq
			attr(data, 'maxAccFreq') <- args$maxAccFreq
		}

		qsave(data, fileMaster)
		cat(paste0("\tMaster table saved: ", fileMaster, " (", nrow(data), " rows).\n"))
	} else {
		stop(paste0(fileMaster, ' already exists... Use -w argument to overwrite.'))
	}

# Save Rsession info
	getVersion(args$taskMeta)
