#!/bin/rscript

# Add the working directory in the libPath to add the user's BSgenome installation
	.libPaths( c( './', .libPaths() ) )

# Loading arguments
	if(!suppressMessages(require("optparse", character.only = TRUE))) stop("Package not found: optparse")

	option_list = list(
	## required
		make_option(c("-i", "--InputQs"), type="character", default=NULL, help="Path to .qs file containing GRanges object with the SMF signal", metavar="/path/to/BulkSMF.qs"),
		make_option(c("-s", "--stat"), type="character", default='qc.tsv', help="Path qc.tsv", metavar="/path/to/qc.tsv"),

	## optional
		make_option(c("-e", "--expName"), type="character", default=NULL, help="Name of the SMF-ONT experiment", metavar="character"),
		make_option(c("-m", "--mode"), type="character", default='cpg,gpc', help="Vector (comma separated) of methylation context to analysis", metavar="character"),
		make_option(c("-r", "--refData"), type="character", default=NULL, help="Reference data to compare (comma separated if multiple)", metavar="character"),
		make_option(c("-t", "--threads"), type="integer", default=12, help="Number of thread to use (>0)", metavar="number"),
		make_option(c("-c", "--minCoverage"), type="integer", default=20, help="Minimal number of read over Cs to measure methylation rate", metavar="number"),
		make_option(c("-y", "--refMinCoverage"), type="integer", default=20, help="Minimal number of read over Cs to measure methylation rate for Nanopore reference data", metavar="number"),
		make_option(c("-z", "--taskMeta"), type="character", default='smf_bulk_qc', help="Name of the current task for versioning", metavar="character")
	)
	opt_parser = OptionParser(option_list=option_list, description="Script to call methylated cytosines out of SMF-Nanopore data.")
	args <- parse_args(opt_parser)

# Loading libraries
	dep.l <- c('this.path', 'qs', 'magrittr', 'tools', 'parallel', 'readr', 'dplyr', 'GenomicRanges', 'ggplot2', 'RColorBrewer')
	for (i in dep.l) {if(!suppressMessages(require(i, character.only = TRUE))) stop(paste0("Package not found: ", i))}
	source(paste0(this.dir(), '/smf_utils.R'))

# Check the ability to work
	## SMF data
		if (is.null(args$InputQs)) {
			print_help(opt_parser)
			stop("An argument in --InputQs is required (-i BulkSMF_XXX_cpggpc.qs ).\n", call.=FALSE)
			if(!file.exists(args$InputQs)){
				print_help(opt_parser)
				stop("The file provided --InputQs doesn't exist (-i BulkSMF_XXX_cpggpc.qs ).\n", call.=FALSE)
			}
		}

	## stat data
		if(!file.exists(args$stat)){
			print_help(opt_parser)
			stop("The file provided --stat doesn't exist (-i qc.tsv ).\n", call.=FALSE)
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

	## Ref Minimum coverage arg
	if (!is.numeric(args$refMinCoverage)) {
		print_help(opt_parser)
		stop("The argument in --refMinCoverage is not numeric (-c 20).\n", call.=FALSE)
	} else if(args$refMinCoverage < 1) {
		print_help(opt_parser)
		stop("The argument in --refMinCoverage cannot be lower than 1 (-c 20).\n", call.=FALSE)
	}

# Load data
	if(args$minCoverage > 1) {
		AllC <- qread(args$InputQs) %>% CovFilter(., args$minCoverage)
	} else {
		AllC <- qread(args$InputQs)
	}
	stat <- read_tsv(args$stat, show_col_types=FALSE)

	subsetIndex=list(
		GCi = AllC$type == 'GCH',
		CGi = AllC$type == 'HCG',
		GCGi = AllC$type == 'GCG',
		Ai = AllC$type == 'A'
	)

	## Check ref data
		PlotRefData=FALSE
		if (!is.null(args$refData)) {
			refDataFiles <- unlist(strsplit(args$refData, ','))
			refData.l <- mclapply(refDataFiles, function(f){
				if(!file.exists(f)){
					message(paste('The dataset file', f, "doesn't exist. Skipping...\n"))
					return(GRanges())
				} else {
					if(file_ext(f) == 'qs'){
						ref = qread(f)
					} else if(file_ext(f) == 'rds'){
						ref = readRDS(f)
					} else {
						message(paste('The dataset file extension', file_ext(f), "isn't supported. Skipping...\n"))
						return(GRanges())
					}

					# Collapse GCG context id required
					ref <- ref[!duplicated(ref), which(colnames(mcols(ref)) != 'type')]
					ref <- ref[,sapply(mcols(ref), is.numeric)]

					if(length(mcols(ref)) > 0){
						# apply Ref mincoverage filter for ref Nanopore data
						if(length(grep('.read_count', colnames(mcols(ref)))) > 0){
							ref <- CovFilter(ref, args$refMinCoverage)
							ref <- ref[,grep('.Met_rate', colnames(mcols(ref)))]
							colnames(mcols(ref)) <- gsub('.Met_rate', '', colnames(mcols(ref)))
						}

						# check if methyl data are compatible
						if(nrow(mcols(ref)) != nrow(mcols(AllC))){
							message(paste0('Reference Data: ', f,' not compatible.\nReformatting... (may reduce the comparison points with the reference)\n'))
							ov <- findOverlaps(ref, AllC)
							tmp <- AllC[,0]
							elementMetadata(tmp) <- matrix(as.numeric(NA), nrow=nrow(mcols(AllC)), ncol=ncol(mcols(ref)))
							colnames(mcols(tmp)) <- colnames(mcols(ref))
							elementMetadata(tmp)[subjectHits(ov),] <- mcols(ref)[queryHits(ov),]
							ref <- tmp
							rm(tmp)
						}
						
						return(ref)
					} else {
						return(GRanges())
					}
				}
			}, mc.cores=args$threads)
			refData <- refData.l[[1]]
			if(length(refData.l) > 1){
				for (k in refData.l[2:length(refData.l)]) {
					if(nrow(mcols(k)) == nrow(mcols(refData))){
						mcols(refData) <- cbind(mcols(refData) , mcols(k))
					}
				}
			}
			rm(refData.l, refDataFiles)
			if(length(refData) > 0){
				PlotRefData=TRUE
			}
		}

# MAIN: QC SMF data
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

	plotPath <- paste0(EXPNAME, '_plots_minCov_', args$minCoverage,'/', FILTER)
	for (i in plotPath){ if(!dir.exists(i)) { dir.create(i, recursive=T) } }

	# How much site are covered before coverage filtering
		MakePie <- function(data, FILTER, file){
			pie <- data %>% select(c(expNames,Cs.AllCObject, No.cov.Cs)) %>% 
				bind_rows(.,.) %>%
				mutate(group=factor(lapply(FILTER, function(x){
					idx=grep(x, expNames)
					out=rep(x, length(idx))
					names(out) <- idx
					return(out)
					}) %>% unlist() %>% .[order(names(.))]
				, levels=FILTER)) %>%
				mutate(count=c(Cs.AllCObject[1:(nrow(.)/2)], No.cov.Cs[(nrow(.)/2+1):nrow(.)]))

			pie <- pie %>%
				mutate(prop = count / sum(Cs.AllCObject[1], No.cov.Cs[1]) *100) %>%
				mutate(group2=factor(rep(c('covered', 'No covered'), each=nrow(pie)/2), levels=c('covered', 'No covered'))) %>%
				mutate(labels = paste0(group2, '\n', count, '\n (', round(prop, 1), '%)')) %>%
				ggplot(., aes(x = "", y = count, fill = group2)) +
					geom_bar(stat="identity", width=1, color="white") +
					coord_polar("y", start=0) +
					labs(title=paste0("Expected total number of methylation sites: ",
						formatC(unclass(sum(pie$Cs.AllCObject[1], pie$No.cov.Cs[1])),
							decimal.mark=',', big.mark=" ", digits = 0, format = "f")
						), y = '', x='') +
					theme_void() +
					theme(legend.position="none", plot.title = element_text(hjust = 0.5, size=8)) +
					geom_text(aes(label = labels), position=position_stack(vjust = 0.5), color = "white", size=3) +
					facet_wrap(~group, strip.position = 'bottom') +
					scale_fill_brewer(palette="Dark2")
			ggsave(file=file, plot=pie)
			return(paste0(file, ' plotted!'))
		}
		pie <- MakePie(stat, FILTER, paste0(EXPNAME, '_plots_minCov_', args$minCoverage,'/covered_sites_rawdata.pdf'))
		rm(stat)

	# How much site are covered after coverage filtering
		tmp = mcols(AllC)[,grepl('.Met_rate', colnames(mcols(AllC))), drop= FALSE]
		pie <- tibble(expNames=colnames(tmp),
			Cs.AllCObject=sapply(seq_len(ncol(tmp)), function(i) {as.numeric(summary(is.na(tmp[, i]))['FALSE'])}),
			No.cov.Cs=sapply(seq_len(ncol(tmp)), function(i) {as.numeric(summary(is.na(tmp[, i]))['TRUE'])})
		) %>% MakePie(., FILTER, paste0(EXPNAME, '_plots_minCov_', args$minCoverage,'/covered_sites_minCov_', args$minCoverage ,'.pdf'))
		rm(tmp, pie)

	# Coverage distribution
		pDistV <- function(data, file, title, angle=TRUE){
			degres=0
			if(angle){degres=45}
			p <- ggplot(data, aes(x=expName, y=read_count, fill=expName)) + 
				geom_violin(trim=FALSE, bw=1, show.legend=FALSE) +
				geom_boxplot(width=0.1, fill="white", outlier.shape = NA) +
				labs(title=title, x="", y = "Read count") +
				scale_fill_brewer(palette="RdBu") + 
				theme_bw() +
				theme(axis.text.x = element_text(angle = degres, hjust = 1))
			ggsave(filename=file, plot=p)
		}

		data <- mclapply(seq_along(FILTER), function(i){
			path <- plotPath[i]
			Filt = FILTER[i]

			reads <- mcols(AllC) %>%
				.[,grep(Filt, colnames(.))] %>% 
				.[,grep('.read_count', colnames(.)), drop=FALSE]
			exp = colnames(reads)

			reads <- as.numeric(reads[,1])
			ExtremOutlier <- quantile(reads, probs=.995, na.rm=T)

			reads_gg <- data.frame(
				read_count = reads[!is.na(reads) & reads <= ExtremOutlier],
				expName = rep(Filt, length(reads[!is.na(reads) & reads <= ExtremOutlier]))
			)
			
			pDistV(reads_gg, paste0(path,'/distCov_', EXPNAME, '_', Filt, '.pdf'),
				title=paste(EXPNAME, Filt, "(Top 0.5% outliers removed)"), angle=FALSE)
			return(reads_gg)
		}, mc.cores=args$threads)
		data <- do.call(rbind, data)
		pDistV(data, paste0(EXPNAME, '_plots_minCov_', args$minCoverage,'/distCov_', EXPNAME, '.pdf'), title=paste(EXPNAME, "(Top 0.5% outliers removed)"), angle=TRUE)
		rm(data)

	# Plot comparison
		if(PlotRefData){
			AllC <- AllC[,grep('.Met_rate', colnames(mcols(AllC)))]
			correctN <- colnames(mcols(AllC)) %>%
				gsub('.Met_rate', '', .) %>%
				lapply(., function(i){
					unlist(lapply(FILTER, function(k){
						if(length(grep(k,i))>0){return(k)}
					}))
				}) %>% unlist()

			colnames(mcols(AllC)) <- correctN
			refName <- colnames(mcols(refData))
			spRefName <- grep(paste(FILTER, collapse='|'), refName)

			AllC <- cbind(mcols(AllC), mcols(refData))
			rm(refData)

			# Full resolution
			fileName <- paste0(dirname(plotPath[1]), '/GW_SMF_', contextView, '_meth_', EXPNAME, '_minCov_', args$minCoverage, '.pdf')
			makePairsPlot(data = AllC, file = fileName, threads = args$threads)
			
			if(length(motifs) > 1 ){
				trash <- mclapply(motifs, function(mot){
					fileName <- paste0(dirname(plotPath[1]), '/GW_SMF_', mot,'_meth_', EXPNAME, '_minCov_', args$minCoverage, '.pdf')
					if(mot == 'cpg'){
						indexToKeep <- subsetIndex$CGi
					} else if(mot == 'gpc'){
						indexToKeep <- subsetIndex$GCi
					} else if(mot == 'a'){
						indexToKeep <- subsetIndex$Ai
					} else  if(mot == 'c'){
						indexToKeep <- Reduce(`|`, list(subsetIndex$CGi, subsetIndex$GCi, subsetIndex$GCGi))
					}
					makePairsPlot(data = AllC[indexToKeep,], file = fileName, threads=args$threads)
				}, mc.cores=args$threads)
			}

			trash <- mclapply(seq_along(FILTER), function(i){
				path <- plotPath[i]
				dataSub <- AllC[,c(FILTER[i], grep(FILTER[i], refName[spRefName], value=TRUE), refName[-spRefName])]
				colnames(dataSub) <- c(FILTER[i], gsub(paste0('_', FILTER[i]), '', grep(FILTER[i], refName[spRefName], value=TRUE)), refName[-spRefName])

				fileName <- paste0(path, '/GW_SMF_', contextView, '_meth_', EXPNAME, '_', FILTER[i], '_minCov_', args$minCoverage, '.pdf')
				makePairsPlot(data = dataSub, file = fileName, threads = args$threads)
				
				if(length(motifs) > 1 ){
					trash <- lapply(motifs, function(mot){
						fileName <- paste0(path, '/GW_SMF_', mot,'_meth_', EXPNAME, '_', FILTER[i], '_minCov_', args$minCoverage, '.pdf')
						if(mot == 'cpg'){
							indexToKeep <- subsetIndex$CGi
						} else if(mot == 'gpc'){
							indexToKeep <- subsetIndex$GCi
						} else if(mot == 'a'){
							indexToKeep <- subsetIndex$Ai
						} else  if(mot == 'c'){
							indexToKeep <- Reduce(`|`, list(subsetIndex$CGi, subsetIndex$GCi, subsetIndex$GCGi))
						}
						makePairsPlot(data = dataSub[indexToKeep,], file = fileName, threads=args$threads)
					})
				}
			}, mc.cores=args$threads)
		}

# Save Rsession info
	getVersion(args$taskMeta)
