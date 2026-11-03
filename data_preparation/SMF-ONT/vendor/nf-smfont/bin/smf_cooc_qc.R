#!/bin/rscript

# Add the working directory in the libPath to add the user's BSgenome installation
	.libPaths( c( './', .libPaths() ) )

# Loading arguments
	if(!suppressMessages(require("optparse", character.only = TRUE))) stop("Package not found: optparse")

	option_list = list(
	## required
		make_option(c("-i", "--inputCont"), type="character", default=NULL, help="Comma-sep path(s) to contingency object(s) (.qs), output of smf_cooc.R [required].", metavar="/path/to/contingency1.qs,/path/to/contingency2.qs"),

	## optional
		make_option(c("-e", "--expName"), type="character", default=NULL, help="Name of the experiment", metavar="character"),
		make_option(c("-s", "--sigThreshold"), type="double", default=0.05, help="Adjusted p-value threshold to call a pair significant [default= %default]", metavar="numeric"),
		make_option(c("-t", "--threads"), type="integer", default=12, help="Number of thread to use (>0)", metavar="Positive number"),
		make_option(c("-z", "--taskMeta"), type="character", default='smf_cooc_qc', help="Name of the current task for versioning", metavar="character")
	)
	opt_parser = OptionParser(option_list=option_list, description="Script to generate generic QC plots (odds-ratio vs distance, significance and coverage distributions) for co-occupancy/co-accessibility contingency tables.")
	args <- parse_args(opt_parser)

# Loading libraries
	dep.l <- c('this.path', 'qs', 'parallel', 'dplyr', 'tidyr', 'tibble', 'magrittr', 'ggplot2', 'scales', 'patchwork')
	for (i in dep.l) {if(!suppressMessages(require(i, character.only = TRUE))) stop(paste0("Package not found: ", i))}
	source(paste0(this.dir(), '/smf_utils.R'))
	options(bitmapType='cairo')

# Check the ability to work
	## Contingency input(s)
		if(is.null(args$inputCont)){
			print_help(opt_parser)
			stop("An argument in --inputCont is required (-i path/to/contingency.qs).\n", call.=FALSE)
		}
		inputCont <- unlist(strsplit(args$inputCont, ','))
		for (i in inputCont) {
			if (!file.exists(i)) {
				print_help(opt_parser)
				stop(paste0("The file '", i, "' in --inputCont does not exist (-i path/to/contingency1.qs,path/to/contingency2.qs).\n"), call.=FALSE)
			}
		}

	## Sample name
		EXPNAME = ifelse(
			is.null(args$expName),
			paste0(format(Sys.time(), "%Y-%m-%d_%H-%M"), '_SMF_coop_QC'),
			args$expName
		)

	## thread
		if (!is.numeric(args$threads)) {
			print_help(opt_parser)
			stop("The argument in --threads is not numeric (-t 10).\n", call.=FALSE)
		} else if(args$threads < 1) {
			print_help(opt_parser)
			stop("The argument in --threads is lower than 1 (-t 10).\n", call.=FALSE)
		}

# MAIN:
	cat("\nGenerating co-occupancy QC plots...\n")

	# Flatten every contingency object into a long tibble of per-pair, per-comparison metrics
		comNames <- c('coor', 'Pair_name', 'PairID')
		data <- bind_rows(mclapply(inputCont, function(f){
			conT.L <- qread(f)
			combo <- attr(conT.L, 'featureNature')
			if(is.null(combo)){combo <- tools::file_path_sans_ext(basename(f))}

			if(length(conT.L) == 0){return(NULL)}

			compTypes <- names(conT.L[[1]])
			compTypes <- compTypes[!compTypes %in% comNames]

			bind_rows(lapply(compTypes, function(type){
				dist <- unlist(lapply(conT.L, function(i){
					splitD <- as.numeric(unlist(strsplit(unlist(strsplit(i$coor, ':'))[2], '-')))
					abs(splitD[1] - splitD[2])
				}))
				tibble(
					combo = combo,
					analysis = type,
					PairID = unlist(lapply(conT.L, function(i){i$PairID})),
					Pair_name = unlist(lapply(conT.L, function(i){i$Pair_name})),
					dist = dist,
					odd = unlist(lapply(conT.L, function(i){as.numeric(i[[type]]$odd)})),
					score = unlist(lapply(conT.L, function(i){as.numeric(i[[type]]$score)})),
					pvalue = unlist(lapply(conT.L, function(i){as.numeric(i[[type]]$pvalue)})),
					p.adjust = unlist(lapply(conT.L, function(i){as.numeric(i[[type]]$p.adjust)})),
					cov = unlist(lapply(conT.L, function(i){as.numeric(i[[type]]$cov)})),
					cov.adjust = unlist(lapply(conT.L, function(i){as.numeric(i[[type]]$cov.adjust)}))
				)
			}))
		}, mc.cores=args$threads))

	if(nrow(data) == 0){
		cat("\tNo pair available in any of the contingency object(s)... Skipping plot generation.\n")
	} else {
		data$signif <- ifelse(is.na(data$p.adjust), 'untested', ifelse(data$p.adjust < args$sigThreshold, 'significant', 'non significant'))
		data$signif <- factor(data$signif, levels=c('non significant', 'significant', 'untested'))
		data$log2odd <- log2(data$odd)

		write.table(data, paste0(EXPNAME, '_coop_summary_metrics.tsv'), sep='\t', quote=FALSE, row.names=FALSE)

		# 1) Odds-ratio vs genomic distance
			p1 <- ggplot(data, aes(x=dist, y=log2odd, colour=signif)) +
				geom_point(alpha=0.5, size=0.8) +
				geom_hline(yintercept=0, linetype='dashed', colour='grey40') +
				scale_x_log10(labels=comma) +
				scale_colour_manual(values=c('non significant'='grey60', 'significant'='firebrick', 'untested'='grey85')) +
				facet_wrap(~combo+analysis, scales='free') +
				labs(x='Genomic distance between paired features (bp)', y='log2(Odds ratio)', colour=paste0('p.adjust < ', args$sigThreshold)) +
				theme_bw()

		# 2) p-value / adjusted p-value distribution
			p2 <- ggplot(tidyr::pivot_longer(data, cols=c('pvalue', 'p.adjust'), names_to='type', values_to='value'), aes(x=value, fill=type)) +
				geom_histogram(position='identity', alpha=0.5, bins=50) +
				facet_wrap(~combo+analysis, scales='free_y') +
				labs(x='p-value', y='Number of pairs', fill=NULL) +
				theme_bw()

		# 3) Coverage distribution
			p3 <- ggplot(tidyr::pivot_longer(data, cols=c('cov', 'cov.adjust'), names_to='type', values_to='value'), aes(x=value, fill=type)) +
				geom_histogram(position='identity', alpha=0.5, bins=50) +
				scale_x_log10(labels=comma) +
				facet_wrap(~combo+analysis, scales='free_y') +
				labs(x='Number of single molecules covering the pair', y='Number of pairs', fill=NULL) +
				theme_bw()

		# 4) Number of (significant) pairs per comparison / combo
			p4 <- data %>%
				dplyr::count(combo, analysis, signif) %>%
				ggplot(aes(x=analysis, y=n, fill=signif)) +
				geom_col(position='stack') +
				scale_fill_manual(values=c('non significant'='grey60', 'significant'='firebrick', 'untested'='grey85')) +
				facet_wrap(~combo, scales='free_x') +
				coord_flip() +
				labs(x=NULL, y='Number of pairs', fill=NULL) +
				theme_bw()

		pdf(paste0(EXPNAME, '_coop_qc_plots.pdf'), width=10, height=8)
			print(p1)
			print(p2)
			print(p3)
			print(p4)
		dev.off()
	}

# Save Rsession info
	getVersion(args$taskMeta)
