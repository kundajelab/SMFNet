#!/bin/rscript

# Add the working directory in the libPath to add the user's BSgenome installation
	.libPaths( c( './', .libPaths() ) )

# Loading arguments
	if(!suppressMessages(require("optparse", character.only = TRUE))) stop("Package not found: optparse")

	option_list = list(
	## required
		make_option(c("-g", "--genome"), type="character", default=NULL, help="Name of BSgenome to use [required]", metavar="character"),
		make_option(c("-f", "--featureFile"), type="character", default=NULL, help="Path to rds/txt/csv/xls file with feature coordinate [required].", metavar="/path/to/feature_file.rds"),
		make_option(c("-s", "--context"), type="character", default='cpg,gpc', help="Vector (comma sep) of methylation context to", metavar="character"),

	## optional
		make_option(c("-i", "--InputFolder"), type="character", default='./', help="Path to folder containing all bed.gz / tbi files", metavar="/path/to/MethycallFolder/"),
		make_option(c("-e", "--expName"), type="character", default=NULL, help="Name of the experiment", metavar="character"),
		make_option(c("-x", "--featureNature"), type="character", default="TF", help="Nature of the regions of interest: TF, TSS or DHS [default= %default]", metavar="character"),
		make_option(c("-m", "--mode"), type="character", default="single",
			help="Calling mode at the features:\n - single or cluster for TFs [default= %default]\n - std or downstream for TSSs\n - <base pair> window from the center of DHS", metavar="character"),
		make_option(c("-w", "--overwrite"), action="store_true", default=FALSE, help="Overwrite existing experiment [default= %default]"),
		make_option(c("-t", "--threads"), type="integer", default=12, help="Number of thread to use (>0)", metavar="Positive number"),
		make_option(c("-z", "--taskMeta"), type="character", default='smf_sort', help="Name of the current task for versioning", metavar="character")
	)
	opt_parser = OptionParser(option_list=option_list, description="Script to sort single nanopore molecule for specific genomic loci.")
	args <- parse_args(opt_parser)

# Loading libraries
	dep.l <- c('this.path', 'qs', 'parallel', 'rtracklayer', 'readr', 'readxl', 'dplyr', 'magrittr')
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

	## check for input files
		filePath <- gsub('//', '/', dir(args$InputFolder, pattern='.bed.gz', all.files=FALSE, full.names=TRUE, ignore.case=TRUE))
		SM_files <- data.frame(bed=filePath[grepl('.bed.gz$', filePath)], tabix=filePath[grepl('.bed.gz.tbi$', filePath)])
		if(nrow(SM_files) != 1){stop('Mulitple BED input files found...')}

	## Sample name
		EXPNAME = ifelse(
			is.null(args$expName),
			paste0(format(Sys.time(), "%Y-%m-%d_%H-%M"), '_SMF_singleMolecule_analysis'),
			args$expName
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
				if (!tolower(args$mode) %in% c("single", "cluster")) {
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

	## thread
		if (!is.numeric(args$threads)) {
			print_help(opt_parser)
			stop("The argument in --threads is not numeric (-t 10).\n", call.=FALSE)
		} else if(args$threads < 1) {
			print_help(opt_parser)
			stop("The argument in --threads is lower than 1 (-t 10).\n", call.=FALSE)
		}

# MAIN:
	cat(paste0("\nSorting around ", args$mode, " ", args$featureNature, "...\n"))
	# Split the genome in chunks to avoid memory over usage
		#create genomic chunks (based on feature positions)
		#sort TFobject
			wms = feature[order(start(feature))]

		#split by chr
			wms.c = split(wms, seqnames(wms))
			regs = unlist(GRangesList(lapply(seq_along(wms.c), function(i) {
				x = wms.c[[i]]
				l = length(x)
				if(l > 0) {
					#cut by TF position
					if(l < 25) {
						seq.i = seq(1, l, 1)
					} else {
						seq.i = c(seq(1, l, round(l/24)), l)
					}
					st.seq = start(x[seq.i])

					lowB = st.seq[1:(length(st.seq)-1)]-1000
					lowB[lowB < 0] <- 1

					if(length(st.seq) == 1){
						highB = st.seq[1] + 1000
					} else {
						highB = st.seq[2:(length(st.seq))] + 1000
					}
					maxLengthChr <- seqlengths(GENOME)[as.character(seqnames(x)[1])]
					highB[highB > maxLengthChr] <- maxLengthChr
					return(GRanges(seqnames(x[1]),IRanges(lowB,highB)))
				} else {
					return(GRanges())
				}
			})))

	# Define Bin object
		if(!cluster){
			if(length(colnames(mcols(feature))) > 3){
				feature <- resize(feature[,1:3], 1, fix='center')
			} else {
				feature <- resize(feature, 1, fix='center')
			}
			if(tolower(args$featureNature) == "tss"){
				if(tolower(args$mode) == "std"){
					bins <- list(upsMs = c(-58,-43), tMs = c(-36,-22), inrMs=c(-6,14), pMs = c(28,47)) # sorting promoter DMelanogaster (up - tata - inr - pol)
					binFileName='bin_used_TSSstd.qs'
				} else {
					bins <- list(upsMs = c(-58,-43), tMs = c(-36,-22), pMs = c(28,47), dosMs=c(54,69)) # sorting promoter Mmusculus (up - tata - pol - down)
					binFileName='bin_used_TSSdownstream.qs'
				}
			} else if (tolower(args$featureNature) == "tf"){
				bins <- list(upMs=c(-35,-25), inMs=c(-7,7), doMs=c(25,35))
				binFileName='bin_used_TF.qs'
			} else {
				bins <- list(inDHSs=c(-ceiling(winbp/2), ceiling(winbp/2)))
				binFileName=paste0('bin_used_DHS_', args$mode,'.qs')
			}

			binsL <- unlist(GRangesList(mclapply(1:length(bins), function(i){ # do not sort the bins or that can affect wrongly the sorting
				bin <- bins[[i]]
				return(MakeBins(RoI=feature, viewPoint=as.numeric(start(feature)), binAbsPos=bin, binName=names(bins[i]), rowNames=names(feature)))
				}, mc.cores=args$threads)))
			binsL = split(binsL, names(binsL))
			binsL <- binsL[names(feature)]
			qsave(binsL, binFileName)
		}

	# Create single molecule vectors for each sample
		sRsFile <- paste0('SMvec_', EXPNAME, '_sorted_reads_', args$featureNature,'mats_', tolower(args$mode),'_Bins_', contextView, '_context.qs')
		regsD <- cbind(as.data.frame(regs)[,1:3], sample=EXPNAME)

		if(!file.exists(sRsFile)  | args$overwrite){
			regsL <- split(regsD, 1:nrow(regsD))
			sRs <- mclapply(regsL, function(k) {
				if(!cluster){
					# sR <- sorTFbyReg_allGenomes_Nanopore(regDF=k, SM_files=SM_files, target_range=feature, inMs=c(-7,7), upMs=c(-35,-25), doMs=c(25,35), BSgenome=args$genome, cluster=cluster, strandCollapsing=TRUE)
					sR <- sorAnyBins_Nanopore(regDF=k, SM_files=SM_files, target_bins=binsL, BSgenome=args$genome, strandCollapsing=TRUE, binSpe=NULL)
				} else {
					target_range=TF_list
					sR <- sorTFbyReg_allGenomes_Nanopore(regDF=k, SM_files=SM_files, target_range=TF_list, inMs=c(-7,7), upMs=c(-35,-25), doMs=c(25,35), BSgenome=args$genome, cluster=cluster, strandCollapsing=TRUE)
				}
				return(sR)
			}, mc.cores=args$threads, mc.preschedule = FALSE)

			sRs.u <- do.call(c, sRs)
			names(sRs.u) = unlist(lapply(sRs, names))

			# genomic chunk overlaps could induce the computation of the same site multiple times
			# Remove duplicated names
			sRs.u <- sRs.u[!duplicated(names(sRs.u))]

			qsave(sRs.u, sRsFile)
			rm(sRs.u)
		} else {
			stop(paste0(sRsFile, ' already exists... Use -w argument to overwrite.'))
		}

# Save Rsession info
	getVersion(args$taskMeta)
