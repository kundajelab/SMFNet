#!/bin/rscript

# Add the working directory in the libPath to add the user's BSgenome installation
	.libPaths( c( './', .libPaths() ) )

# Loading arguments
	if(!suppressMessages(require("optparse", character.only = TRUE))) stop("Package not found: optparse")

	option_list = list(
	## required
		make_option(c("-f", "--featureFile"), type="character", default=NULL, help="Path to rds/txt/csv/xls file with feature coordinate [required].", metavar="/path/to/feature_file.rds"),
		make_option(c("-s", "--context"), type="character", default='cpg,gpc', help="Vector (comma sep) of methylation context to", metavar="character"),
		make_option(c("-i", "--inputSMvec"), type="character", default=NULL, help="Path to the SM sorted vector (.rds and .qs supported)", metavar="/path/to/SMvector.qs"),

	## optional
		make_option(c("-e", "--expName"), type="character", default=NULL, help="Name of the experiment", metavar="character"),
		make_option(c("-x", "--featureNature"), type="character", default="TF", help="Nature of the regions of interest: TF, TSS or DHS [default= %default]", metavar="character"),
		make_option(c("-m", "--mode"), type="character", default="single",
			help="Calling mode at the features:\n - single or cluster for TFs [default= %default]\n - std or downstream for TSSs\n - <base pair> window from the center of DHS", metavar="character"),
		make_option(c("-c", "--minCoverage"), type="integer", default=20, help="Minimal number of read over Cs to measure methylation rate", metavar="number"),
		make_option(c("-t", "--threads"), type="integer", default=12, help="Number of thread to use (>0)", metavar="Positive number"),
		make_option(c("-z", "--taskMeta"), type="character", default='smf_sort_extract', help="Name of the current task for versioning", metavar="character")
	)
	opt_parser = OptionParser(option_list=option_list, description="Script to sort single nanopore molecule for specific genomic loci.")
	args <- parse_args(opt_parser)

# Loading libraries
	dep.l <- c('this.path', 'qs', 'parallel', 'rtracklayer', 'magrittr')
	for (i in dep.l) {if(!suppressMessages(require(i, character.only = TRUE))) stop(paste0("Package not found: ", i))}
	source(paste0(this.dir(), '/smf_utils.R'))

# Check the ability to work
	## Input folder
		if(!is.null(args$inputSMvec)){
			if (!file.exists(args$inputSMvec)) {
				print_help(opt_parser)
				stop("The input file in --inputSMvec does not exist (-i /path/to/SMvector.qs).\n", call.=FALSE)
			} else {
				ext = tolower(unlist(strsplit(basename(args$inputSMvec), '[.]'))) %>% .[length(.)]
				if(ext == 'rds'){
					sRs.u = readRDS(args$inputSMvec)
				} else {
					sRs.u = qread(args$inputSMvec)
				}
			}
		} else {
			print_help(opt_parser)
			stop("An argument in --inputSMvec is required (-i /path/to/SMvector.qs).\n", call.=FALSE)
		}

	## Sample name
		EXPNAME = ifelse(
			is.null(args$expName),
			paste0(format(Sys.time(), "%Y-%m-%d_%H-%M"), '_SMF_singleMolecule_analysis'),
			unlist(strsplit(args$expName, ','))
		)

	## methylation motifs
		if(!is.null(args$context)){
			motifs <- unlist(lapply(unlist(strsplit(tolower(args$context), ',')), function(mot){
				if(!tolower(mot) %in% c('cpg', 'gpc', 'a', 'cg', 'gc', 'c')){
					print_help(opt_parser)
					stop(paste0('The argument "', mot,'" in --mode is supported (-s cpg,gpc,a,c).\n'), call.=FALSE)
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
		} else {
			print_help(opt_parser)
			stop("An argument in --context is required (-c cpg,gpc).\n", call.=FALSE)
		}

	## Feature inputs
		# File
			if(!is.null(args$featureFile)){
				if (!file.exists(args$featureFile)) {
					print_help(opt_parser)
					stop("The file in --featureFile does not exist (-f path/to/feature_file.rds).\n", call.=FALSE)
				} else {
					feature <- readFeatureFile(args$featureFile)$feature %>% removePalin(., args$threads)
					if (is.null(names(feature))) {
						names(feature) <- feature$name
					}
				}
			} else {
				print_help(opt_parser)
				stop("An argument in --featureFile is required (-f path/to/feature_file.rds).\n", call.=FALSE)
			}

		# Nature
			cluster=FALSE
			if(tolower(args$featureNature) == "tf") {
				if (!tolower(args$mode) %in% c("single", "cluster", "hybrid")) {
					print_help(opt_parser)
					stop("The '--mode' argument for TF sorting is not supported (-m single or -m cluster).\n", call.=FALSE)
				}
				if(tolower(args$mode) == "cluster"){
					cluster=TRUE
					maxDist=75
					minDist=15
					maxClusterSize=10

					cat(paste0("\nCreating cluster objects...\n"))
					Cluster.l <- Make_TFBS_cluster(TFBS=feature, maxDist=maxDist, minDist=minDist, maxClusterSize=maxClusterSize)
					TF_cluster <- Cluster.l[['ClusterGr']]
					TF_list <- Cluster.l[['Clusterlist']]
					rm(Cluster.l)

					# inportant vector for the counting part
						nb_TF=lengths(TF_list)
						TFcat=sort(unique(nb_TF))

					# replace the feature file by the clusters
						feature <- TF_cluster
				}
			} else if(tolower(args$featureNature) == "tss") {
				if (!tolower(args$mode) %in% c("std", "downstream")) {
					print_help(opt_parser)
					stop("The '--mode' argument for TSS sorting is not supported (-m std or -m downstream).\n", call.=FALSE)
				}
			} else if(tolower(args$featureNature) == "dhs") {
				winbp = as.numeric(args$mode)
				args$mode = paste0(winbp, 'bp')
			} else {
				print_help(opt_parser)
				stop("The --featureNature argument is not supported (-x TF or -x TSS).\n", call.=FALSE)
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

# MAIN:
	cat(paste0("\nComputation of the binding frequencies...\n"))
	# Create a vector describing all possible states for the feature type (TF= 2^3, TSS=2^4, DHS=2)
		binNum <- ifelse(tolower(args$featureNature) == "tf", 3, ifelse(tolower(args$featureNature) == "tss", 4, 1))
		allPos = expand.grid(rep(list(c(0,1)), binNum))
		patternStrings=names(table(apply(allPos,1,function(x){(paste(as.character((x)),collapse=''))})))

		# make specific list of vector describing bins for each cluster type (2TFs=4, 3TFs=5, 4TFs=6...)
		if(cluster){
			patternStrings.L <- lapply(TFcat, function(nbc){
				allPos = expand.grid(lapply(seq(nbc+2),function(i){c(0,1)}))
				return(names(table(apply(allPos,1,function(x){(paste(as.character((x)),collapse=''))}))))
			})
			names(patternStrings.L) = TFcat
		}

	# Create a individual count matrices per sample
		countMats = lapply(EXPNAME, function(sp){
			if(length(sRs.u) > 0){
				if(cluster){
					m.L <- lapply(TFcat, function(nbc){
						patternStrings <- patternStrings.L[[as.character(nbc)]]
						sRs.u.c = sRs.u[names(sRs.u) %in% names(nb_TF)[nb_TF == nbc]]
						if(length(sRs.u.c) > 0){
							Counts <- ComputeCount(sRs.u.c, patternStrings, threads=args$threads)

							# Check for duplicates
								if(nrow(Counts) != length(unique(rownames(Counts)))){
									Counts <- Counts[unique(rownames(Counts)),]
									warning(paste0('\nThe single molecule vector contains multiple occurances for the same viewpoint for the cluster with ', nbc,' TFBS!\nExtracting unique information...\n'),
										call. = TRUE, immediate. = TRUE, noBreaks. = TRUE)
								}

							# expend the matrix to all features (not necessary ? output object heavier...)
								m = matrix(NA, ncol=length(patternStrings), nrow=length(feature), dimnames=list(names(feature),patternStrings))
								m[rownames(Counts),] = Counts

							return(m)
						} else {
							return(matrix(NA, ncol=length(patternStrings), nrow=length(feature), dimnames=list(names(feature),patternStrings)))
						}
					})
					names(m.L) = TFcat
					return(m.L)
				} else {
					Counts <- ComputeCount(sRs.u, patternStrings, threads=args$threads)

					# Check for duplicates
						if(nrow(Counts) != length(unique(rownames(Counts)))){
							Counts <- Counts[unique(rownames(Counts)),]
							warning('\nThe single molecule vector contains multiple occurances for the same viewpoint!\nExtracting unique information...\n',
								call. = TRUE, immediate. = TRUE, noBreaks. = TRUE)
						}

					# expend the matrix to all features
						m = matrix(NA, ncol=length(patternStrings), nrow=length(feature), dimnames=list(names(feature),patternStrings))
						m[rownames(Counts),] = Counts

					return(m)
				}
			} else {
				if(cluster){
					m.L <- lapply(TFcat, function(nbc){
						patternStrings <- patternStrings.L[[as.character(nbc)]]
						return(matrix(NA, ncol=length(patternStrings), nrow=length(feature), dimnames=list(names(feature),patternStrings)))
					})
					names(m.L) = TFcat
					return(m.L)
				} else {
					return(matrix(NA, ncol=length(patternStrings), nrow=length(feature), dimnames=list(names(feature),patternStrings)))
				}
			}
		})
		qsave(countMats, paste0('pattern_count_', EXPNAME, '_', args$featureNature,'mats_', tolower(args$mode),'_Bins_', contextView, '_context.qs'))

	# Generate the frequencies matrix for all pattern combination and expend it to all the TFs in the input object (to enable comparisons between samples)
		freqMat.e = mclapply(seq_along(EXPNAME), function(k) {
			if(cluster) {
				freqMat.L <- lapply(TFcat, function(nbc){
					getFrequencyMat(countMats[[k]][[as.character(nbc)]], args$minCoverage)
					})
				names(freqMat.L) = TFcat
				return(freqMat.L)
			} else {
				return(getFrequencyMat(countMats[[k]], args$minCoverage))
			}
		}, mc.cores=args$threads)
		names(freqMat.e) <- EXPNAME
		qsave(freqMat.e, paste0('pattern_frequency_', EXPNAME, '_', args$featureNature,'mats_', tolower(args$mode),'_Bins_', contextView, '_context_',args$minCoverage,'_minCov.qs'))

	# States declaration (group methylation pattern into biological states)
		if(!cluster){
			states = makeBinToClass(patternStrings, tolower(args$mode))
			statesF = as.factor(unlist(lapply(seq_along(states),function(i){rep(names(states[i]),length(states[[i]]))}))[order(unlist(states))])
			statesF = factor(statesF,levels=names(states))
		}
		# } else {
		# 	allPos=expand.grid(c(0,1),c(0,1))
		# 	TF1=names(table(apply(allPos,1,function(x){(paste(as.character((x)),collapse=''))})))
		# 	names(TF1)=c('nucleosome','unassigned','bound','accessible')

		# 	TF2=names(table(apply(allPos,1,function(x){(paste(as.character((x)),collapse=''))})))
		# 	names(TF2)=c('nucleosome','bound','unassigned','accessible')

		# 	states=unlist(lapply(seq_along(TF1),function(i){lapply(seq_along(TF2),function(j){paste(TF1[i],TF2[j],sep='')})}))
		# 	names(states)=unlist(lapply(seq_along(TF1),function(i){lapply(seq_along(TF2),function(j){paste(names(TF1)[i],names(TF2)[j],sep='_')})}))
		# }


	#create the grouped counts object and the grouped frequency matrix
		if(!cluster){
				groupedCounts.l = mclapply(seq_along(EXPNAME), function(i){
					t(apply(countMats[[i]], 1, function(x){tapply(x, statesF, sum)}))
				}, mc.cores = args$threads)
				names(groupedCounts.l) <- EXPNAME
				qsave(groupedCounts.l, paste0('states_count_', EXPNAME, '_', args$featureNature,'mats_', tolower(args$mode),'_Bins_', contextView, '_context.qs'))

				groupedFreq.l = mclapply(seq_along(EXPNAME), function(k) {
					getFrequencyMat(groupedCounts.l[[k]], args$minCoverage)
				}, mc.cores=args$threads)
				names(groupedFreq.l)=EXPNAME
				qsave(groupedFreq.l, paste0('states_frequency_', EXPNAME, '_', args$featureNature,'mats_', tolower(args$mode),'_Bins_', contextView, '_context_',args$minCoverage,'_minCov.qs'))
		}
		# } else {
		# 	stateM=cbind(string.split(as.character(combined_statesF),'_',1),string.split(as.character(combined_statesF),'_',2))
		# 	#group states (important for the nucleosome occupied state)
		# 		grouped_states=	list(combined_states[stateM[,1]=='bound'& stateM[,2]=='bound'],
		# 			combined_states[stateM[,1]=='bound'& !stateM[,2]=='bound' & !stateM[,2]=='nucleosome'],
		# 			combined_states[!stateM[,1]=='bound'& !stateM[,1]=='nucleosome' &stateM[,2]=='bound'],
		# 			combined_states[(!stateM[,1]=='bound'& !stateM[,2]=='bound')|(stateM[,1]=='bound'& stateM[,2]=='nucleosome')|(stateM[,2]=='bound'& stateM[,1]=='nucleosome')])
		# 		grouped_states=rev(grouped_states)

		# }

# Save Rsession info
	getVersion(args$taskMeta)
