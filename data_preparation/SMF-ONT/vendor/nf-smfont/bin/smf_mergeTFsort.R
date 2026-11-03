#!/bin/rscript

# Add the working directory in the libPath to add the user's BSgenome installation
	.libPaths( c( './', .libPaths() ) )

# Loading arguments
	if(!suppressMessages(require("optparse", character.only = TRUE))) stop("Package not found: optparse")

	option_list = list(
	## required
		make_option(c("-i", "--InputFolder"), type="character", default='./', help="path to folder containing SM single and cluster vectors stored in rds or qs files .\n", metavar="/path/SMvectorFolders/"),
		make_option(c("-f", "--featureFile"), type="character", default=NULL, help="Path to rds/txt/csv/xls file with feature coordinate [required].", metavar="/path/to/feature_file.rds"),

	## optional
		make_option(c("-e", "--expName"), type="character", default=NULL, help="Name of the experiment", metavar="character"),
		make_option(c("-o", "--outFolder"), type="character", default='NULL', help="path to the output folder [optional]\nThe inputFolder is used as default.", metavar="/path/to/output_folder/"),
		make_option(c("-c", "--minCoverage"), type="integer", default=20, help="Minimal number of read over Cs to measure methylation rate", metavar="number"),
		make_option(c("-t", "--threads"), type="integer", default=12, help="Number of thread to use (>0)", metavar="Positive number"),
		make_option(c("-z", "--taskMeta"), type="character", default='smf_mergeTFsort', help="Name of the current task for versioning", metavar="character")
	)
	opt_parser = OptionParser(option_list=option_list, description="Script to sort single nanopore molecule for specific genomic loci.")
	args <- parse_args(opt_parser)

# Loading libraries
	dep.l <- c('this.path', 'qs','parallel', 'GenomicFeatures', 'ggplot2', 'patchwork', 'stringr', 'dplyr', 'RColorBrewer', 'forcats', 'magrittr')
	for (i in dep.l) {if(!suppressMessages(require(i, character.only = TRUE))) stop(paste0("Package not found: ", i))}
	source(paste0(this.dir(), '/smf_utils.R'))

# Check the ability to work
	## Sample name
		EXPNAME = ifelse(
			is.null(args$expName),
			paste0(format(Sys.time(), "%Y-%m-%d_%H-%M"), '_SMF_singleMolecule_analysis'),
			unlist(strsplit(args$expName, ','))
		)

	## check input folder
		if (is.null(args$InputFolder)){
			print_help(opt_parser)
			stop("An argument in --InputFolder is mandatory (-i /path/SMvectorFolders/).\n", call.=FALSE)
		} else {
			if (!dir.exists(args$InputFolder)) {
				print_help(opt_parser)
				stop("The folder in --InputFolder does not exist (-i /path/SMvectorFolders/).\n", call.=FALSE)
			} else {
				INPUTPATH=gsub('//', '/', paste0(args$InputFolder, '/'))
			}
		}

		Flist <- list.files(INPUTPATH, pattern=paste0(EXPNAME, '_sorted_reads_TFmats_'), full.names= TRUE)
		if(length(grep('single|cluster', Flist)) == 2){
			names(Flist)[grep('single', Flist)] <- 'single'
			names(Flist)[grep('cluster', Flist)] <- 'cluster'
		} else {
			stop(paste0("The sample ", EXPNAME, " doesn't present a single and cluster SM vector!\n"))
		}

	## setting up output folder
		if (!is.null(args$outFolder)){
			OUTPUTPATH = args$InputFolder
			if(!dir.exists(OUTPUTPATH)){dir.create(OUTPUTPATH, recursive=TRUE)}
		} else {
			OUTPUTPATH = INPUTPATH
		}

	## Feature TF file
		if(!is.null(args$featureFile)){
			if (!file.exists(args$featureFile)) {
				print_help(opt_parser)
				stop("The file in --featureFile does not exist (-f path/to/feature_file.rds).\n", call.=FALSE)
			} else {
				feature <- readFeatureFile(args$featureFile)$feature %>% removePalin(., args$threads)
			}
		} else {
			print_help(opt_parser)
			stop("An argument in --featureFile is required (-f path/to/feature_file.rds).\n", call.=FALSE)
		}

	## Minimum coverage arg
		if (!is.numeric(args$minCoverage)) {
			print_help(opt_parser)
			stop("The argument in --minCoverage is not numeric (-c 20).\n", call.=FALSE)
		} else if(args$minCoverage < 1) {
			print_help(opt_parser)
			stop("The argument in --minCoverage cannot be lower than 1 (-c 20).\n", call.=FALSE)
		}

	## thread
		if (!is.numeric(args$threads)) {
			print_help(opt_parser)
			stop("The argument in --threads is not numeric (-t 10).\n", call.=FALSE)
		} else if(args$threads < 1) {
			print_help(opt_parser)
			stop("The argument in --threads is lower than 1 (-t 10).\n", call.=FALSE)
		}

#MAIN:
	# create the Cluster object
		cat("\nLoading reference TF object...\n")
		Cluster.l <- Make_TFBS_cluster(feature)
		TF_cluster <- Cluster.l[['ClusterGr']]
		TF_list <- Cluster.l[['Clusterlist']]

		TF_cluster$rank <- unlist(mclapply(names(TF_cluster), function(i){
			return(median(TF_list[[i]]$rank, na.rm=TRUE))
			}, mc.cores=args$threads))

		TF_cluster$name <- unlist(mclapply(names(TF_cluster), function(i){
			return(paste0(unique(TF_list[[i]]$name), collapse='_'))
			}, mc.cores=args$threads))

		TF_list <- mclapply(TF_list, function(i){
			i$TF_place <- seq(1, length(i))
			i$clusterSize <- length(i)
			return(i)
			}, mc.cores=args$threads)

	# declare bins and groups
		cat("\nDeclaration of TF binding states...\n")
		allPos = expand.grid(rep(list(c(0,1)), 3))
		patternStrings=names(table(apply(allPos,1,function(x){(paste(as.character((x)),collapse=''))})))
		states = makeBinToClass(patternStrings, 'hybrid')
		statesF = as.factor(unlist(lapply(seq_along(states),function(i){rep(names(states[i]),length(states[[i]]))}))[order(unlist(states))])
		statesF = factor(statesF,levels=names(states))

		TFc = colorRampPalette(brewer.pal(9,"Set1"))(9)[c(2,3,4)]
		names(TFc) <- c('nucleosome', 'unbound', 'bound')

	# merge sorting info
		cat("\nMerging single molecule vectors...\n")
		outHybridFile <- gsub('_sorted_reads_TFmats_single_Bins', '_sorted_reads_TFmats_hybrid_Bins', Flist['single'])

		TF_inC <- unlist(GRangesList(TF_list), use.name=FALSE)
		TF_toModif <- names(TF_inC)

		singleV <- qread(Flist['single'])
		clusterV <- qread(Flist['cluster'])

		TF_inC_cov <- unlist(GRangesList(TF_list[names(clusterV)]), use.name=FALSE)
		TFcov_inC <- TF_inC[TF_toModif %in% names(singleV)]

		outPlot <- paste0(OUTPUTPATH, 'Plot_hybrid_TF_sorting_SM_merging')
		if(!dir.exists(outPlot)){dir.create(outPlot)}

		# Merge Single molecule vectors
			if(!file.exists(outHybridFile)){
				HybridV <- singleV

				# plot distribution TFBS in categories
					dataToPlot <- data.frame(col='SM sorted TFBS',
						TFBS=as.numeric(c(summary(names(singleV) %in% TF_toModif)['TRUE'],summary(names(singleV) %in% TF_toModif)['FALSE'], summary(as.factor(TFcov_inC$clusterSize)))),
						inClus=as.factor(c('cluster', 'single', paste0(names(summary(as.factor(TFcov_inC$clusterSize))), 'TFs'))))
					dataToPlot$inClus <- fct_rev(dataToPlot$inClus) %>%  fct_relevel(., "10TFs", after = 2) %>%  fct_relevel(., "cluster", after = 0)


					p1 <- ggplot(data=dataToPlot[dataToPlot$inClus %in% c('cluster', 'single'),], aes(x=col, y=TFBS, fill=inClus)) +
						geom_col() +
						labs(title="TFBS composition", y = "TFBS", x='') +
						scale_fill_brewer(palette="Dark2") +
						theme_classic() +
						theme(legend.title=element_blank(),
							legend.box.margin = margin(250, 2, 2, 2))

					getPalette <- colorRampPalette(rev(c(brewer.pal(9, "Greens")[c(3,9)])))
					p2 <- ggplot(data=dataToPlot[!dataToPlot$inClus %in% c('cluster', 'single'),], aes(x=col, y=TFBS, fill=inClus)) +
						geom_col() +
						labs(title="", y = "", x='') +
						scale_fill_manual(values = getPalette(nrow(dataToPlot[!dataToPlot$inClus %in% c('cluster', 'single'),]))) +
						theme_classic() +
						theme(legend.title=element_blank(), plot.background = element_rect(fill = "transparent", color = NA),
							panel.background = element_rect(fill = "transparent"),
							plot.margin = margin( l = 200))

					layout <- c(area(1,1,3,3), area(1,3,2,4))
					p <- p1 + p2 + patchwork::plot_layout(design = layout)
					ggsave(filename=paste0(outPlot, '/TFBS_comp.pdf'), plot=p, width=6, height=8)

				# iterate along all covered TF clusters
					TFc_cov <- TF_cluster[names(clusterV)]
					sRs_corrected <- unlist(mclapply(seq_along(TFc_cov), function(r){
						clus <- TFc_cov[r]
						clusComp <- TF_list[[names(clus)]]

						CorSM_vector <- lapply(seq_along(clusComp), function(i){
							TF <- clusComp[i]
							SM_vector <- clusterV[[names(clus)]]
							SM_sing <- singleV[names(TF)]

							# reducing bin info into 3 bins
								bins <- string.split.last(names(SM_vector), '_') %>%
									strsplit(., '') %>%
									lapply(., function(st){return(paste(st[c(1, TF$TF_place+1, length(st))], collapse=''))}) %>%
									unlist()
								names(SM_vector) <- paste0(names(TF), '_', bins)

							# Reduce identical bin states
								stN <- unique(names(SM_vector))
								SM_vector <- lapply(stN, function(u){unlist(SM_vector[which(names(SM_vector) == u)], use.name=F)})
								names(SM_vector) = stN

							return(SM_vector)
						})
						names(CorSM_vector) <- names(clusComp)
						return(CorSM_vector)
					}, mc.cores=args$threads), recursive=FALSE)

				# replace original single TF sorting by cluster extrapolated one
					HybridV[names(sRs_corrected)] <- sRs_corrected
					qsave(HybridV, outHybridFile)
			} else {
				HybridV <- qread(outHybridFile)
				sRs_corrected <- HybridV[names(HybridV) %in% names(TF_inC_cov)]
			}

		# look at divergence for each bins / per TF
			commonTF <- names(sRs_corrected)[names(sRs_corrected) %in% names(singleV)]
			dataToPlot <- as.numeric(summary(names(sRs_corrected) %in% names(singleV))[c('FALSE', 'TRUE')]) %>% tibble() %>%
				rbind(.,tibble(.=as.numeric(summary(names(TFcov_inC) %in% names(sRs_corrected))['FALSE']))) %>% cbind(., tibble(TF=factor(c('New', 'Corrected', 'NoCorreted'))))

			dataToPlot <- dataToPlot %>%
				arrange(desc(TF)) %>%
	  			mutate(prop = . / sum(dataToPlot$.[dataToPlot$TF %in% c('Corrected', 'NoCorreted')]) *100) %>%
	  			mutate(labels = ifelse(TF == 'New', paste0(TF, '\n', ., '\n (+', round(prop, 1), '%)'), paste0(TF, '\n', ., '\n (', round(prop, 1), '%)')))

			p <- ggplot(dataToPlot, aes(x="", y=., fill=TF)) +
	  			geom_bar(stat="identity", width=1, color="white") +
	  			coord_polar("y", start=0) +
	  			labs(title="Correction of SM sorting for TFBS in TF clusters (TF dist < 75)", y = '', x='') +
	  			theme_void() +
	  			theme(legend.position="none", plot.title = element_text(hjust = 0.5, size=8)) +
	  			geom_text(aes(label = labels, fontface=2), position=position_stack(vjust = 0.5), color = "white", size=2) +
	  			scale_fill_brewer(palette="Dark2")
	  		ggsave(filename=paste0(outPlot, '/TFBS_correction.pdf'), plot=p, width=4, height=4)

		# Compute Binding frequencies for single and hybrid sorting for corrected TFs
			toCount <- list(new=HybridV[commonTF], ori=singleV[commonTF])
			#count per bin
				counts <- lapply(toCount, ComputeCount, patternStrings=patternStrings,  threads=args$threads)

			# sum up per group
				groupCounts <- lapply(counts, function(i){t(apply(i, 1, function(x){tapply(x, statesF, sum)}))})

			# get freq for fair comparison (num of read can change between ori and new for single TFs)
				groupFreq <- lapply(groupCounts, getFrequencyMat, cutoff=args$minCoverage)

		# plot QC between sorting
			folddataScatter <- function(dataList, threads){
				data <- mclapply(1:length(names(dataList)), function(i){
					mat <- as.data.frame(dataList[[i]])
					colnames(mat) <- paste(names(dataList)[i], colnames(mat), sep='.')
					return(mat)
				}, mc.cores=threads)
				return(do.call(cbind, data))
			}

			dataScatter <- folddataScatter(dataList=groupFreq, threads=args$threads)
			order <- lapply(names(states), function(n){grep(paste0('.', n), colnames(dataScatter), fixed=T)})
			names(order) <- names(states)

			for (i in names(states)) {
				CoI <- as.numeric(unlist(order[i]))
				png(paste0(outPlot,'/scatter_distribution_', i, '_TF_SM_sorting_correction.png'), width = 4960, height = 4960, res=600)
				pairs(as.matrix(dataScatter[, CoI]), upper.panel = panel.cor, diag.panel = panel.hist, lower.panel = panel.jet)
				dev.off()
				dataScatter[,paste0('Delta.', i)] <- (dataScatter[,CoI[1]]+1) / (dataScatter[,CoI[2]]+1)
			}

			ViolinData <- dataScatter[, grep('Delta', colnames(dataScatter))]
			ViolinData <- lapply(seq_along(ViolinData), function(i){
				nam <- colnames(ViolinData[,i, drop=FALSE])
				colN <- unlist(strsplit(nam, '[.]'))
				return(ViolinData[,i, drop=FALSE] %>% mutate(group=colN[2]) %>% rename(!!colN[1] := !!nam))
			})
			ViolinData <- do.call(rbind, ViolinData)

			p <- ggplot(ViolinData, aes(x=group, y=log10(Delta), fill=group)) +
				geom_violin(trim=FALSE, bw=0.05, show.legend=F) +
				geom_boxplot(width=0.1, fill="white", outlier.shape = 1, outlier.size=1) +
				labs(title='Comparison binding freq after sorting correction', x="", y = "log(Delta)") +
				scale_fill_manual(values=TFc) +
				theme_bw()
			ggsave(filename=paste0(outPlot, '/TFBS_correction_deltaFreq.pdf'), plot=p, width=8, height=6)

	cat("\nSuccess!\n")

# Save Rsession info
	getVersion(args$taskMeta)

