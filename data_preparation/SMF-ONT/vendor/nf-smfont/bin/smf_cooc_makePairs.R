#!/bin/rscript

# Add the working directory in the libPath to add the user's BSgenome installation
	.libPaths( c( './', .libPaths() ) )

# Loading arguments
	if(!suppressMessages(require("optparse", character.only = TRUE))) stop("Package not found: optparse")

	option_list = list(
	## required
		make_option(c("-g", "--genome"), type="character", default=NULL, help="Name of BSgenome to use [required]", metavar="character"),
		make_option(c("-f", "--featureFile"), type="character", default=NULL, help="Path to rds/txt/csv/xls file with feature coordinate [required].", metavar="/path/to/feature_file.rds"),

	## optional
        make_option(c("-q", "--freqFile"), type="character", default=NULL, help="Path to rds/txt/csv/xls file with frequency matrix for the features.", metavar="/path/to/frequency_file.rds"),
	    make_option(c("-x", "--featureNature"), type="character", default="TF", help="Nature of the regions of interest: TF, TSS or DHS [default= %default]", metavar="character"),
		make_option(c("-m", "--mode"), type="character", default="single",
			help="Calling mode at the features:\n - single or cluster for TFs [default= %default]\n - std or downstream for TSSs\n - <base pair> window from the center of DHS", metavar="character"),
		make_option(c("-p", "--pairmode"), type="character", default="Freq5", help="Mode to filter feature to pair together [default= %default]\nCan be 'all' or Freq and numeric value from 1 to 100 (Freq5)", metavar="character"),
        make_option(c("-w", "--overwrite"), action="store_true", default=FALSE, help="Overwrite existing pair file [default= %default]"),
        make_option(c("-r", "--rightwindow"), type="integer", default=40, help="Number of kilobases (kb) to look downstream from each feature to find pairing candidates", metavar="Positive number"),
		make_option(c("-t", "--threads"), type="integer", default=12, help="Number of thread to use (>0)", metavar="Positive number"),
		make_option(c("-z", "--taskMeta"), type="character", default='smf_coop_makePairs', help="Name of the current task for versioning", metavar="character")
	)
	opt_parser = OptionParser(option_list=option_list, description="Script to sort single nanopore molecule for specific genomic loci.")
	args <- parse_args(opt_parser)

# Loading libraries
	dep.l <- c('this.path', 'qs', 'parallel', 'magrittr')
	for (i in dep.l) {if(!suppressMessages(require(i, character.only = TRUE))) stop(paste0("Package not found: ", i))}
	source(paste0(this.dir(), '/smf_utils.R'))

# Check the ability to work
	## Genome loading
		# if (is.null(args$genome)) {
		# 	print_help(opt_parser)
		# 	stop("An argument in --genome is required (-g BSgenome.Mmusculus.UCSC.mm10 ).\n", call.=FALSE)
		# } else {
		# 	if(!suppressMessages(require("BSgenome", character.only = TRUE))) stop("Package not found: BSgenome")
		# 	if(!suppressMessages(require(args$genome, character.only = TRUE))) stop(paste0("Package not found: ", args$genome))
		# 	GENOME=get(args$genome)

		# 	# Extract std chromosomes from the loaded genome
		# 	tiles <- keepStandardChromosomes(GRanges(seqinfo(GENOME)), pruning.mode ='coarse')
		# 	norm_chr = as.character(seqnames(tiles))
		# }

	## Feature inputs
		# File
			if(!is.null(args$featureFile)){
                featureFile=unlist(strsplit(args$featureFile, ','))
                for(i in featureFile){
                    if (!file.exists(i)) {
                        print_help(opt_parser)
                        stop(paste0("The file '", i,  "' in --featureFile does not exist (-f path/to/feature_file1.rds,path/to/feature_file2.rds).\n"), call.=FALSE)
                    }
                }

                tmp <- mclapply(featureFile, readFeatureFile, bsgenome = args$genome, mc.cores=args$threads)
		        names(tmp) <- unlist(strsplit(toupper(args$featureNature), '-'))
                featureName = unlist(lapply(tmp, function(i){i$featureName}))
		        feature = lapply(tmp, function(i){i$feature  %>% removePalin(., args$threads)})
                rm(tmp)

                if(tolower(args$featureNature) %in% c("tf-tss", 'tss-tf')){
                    featureName = paste(c(featureName['TSS'], featureName['TF']), collapse='_vs_')
                    feature = feature[c('TSS', 'TF')]
                } else if(tolower(args$featureNature) %in% c("dhs-tss", 'tss-dhs')){
                    featureName = paste(c(featureName['TSS'], featureName['DHS']), collapse='_vs_')
                    feature = feature[c('TSS', 'DHS')]
                }
			} else {
				print_help(opt_parser)
				stop("An argument in --featureFile is required (-f path/to/feature_file.rds).\n", call.=FALSE)
			}

		# Nature
            winbp=0
			if(tolower(args$featureNature) == "tf") {
				if (!tolower(args$mode) %in% c("single", "cluster", "hybrid")) {
					print_help(opt_parser)
					stop("The '--mode' argument for TF sorting is not supported (-m single, -m cluster or -m hybrid).\n", call.=FALSE)
				}
                names(args$mode) <- 'TF'
			} else if(tolower(args$featureNature) == "tss") {
				if (!tolower(args$mode) %in% c("std", "downstream")) {
					print_help(opt_parser)
					stop("The '--mode' argument for TSS sorting is not supported (-m std or -m downstream).\n", call.=FALSE)
				}
                names(args$mode) <- 'TSS'
			} else if(tolower(args$featureNature) == "dhs") {
				winbp = as.numeric(args$mode)
				args$mode = paste0(winbp, 'bp')
                names(args$mode) <- 'DHS'
            } else if (tolower(args$featureNature) %in% c("tf-tss", 'tss-tf')) {
                args$mode <- unlist(strsplit(args$mode, ','))
                names(args$mode)[grep('std|downstream', tolower(args$mode))] <- 'TSS'
                names(args$mode)[grep('single|cluster|hybrid', tolower(args$mode))] <- 'TF'
                if (length(grep('TSS|TF', names(args$mode))) != 2) {
                    print_help(opt_parser)
                    stop("The argument in --mode for TSS-TF sorting is not good (-m std,single or -m downstream,hybrid).\n", call.=FALSE)
                } else {
                    args$mode <- c(args$mode['TSS'], args$mode['TF'])
                }
            } else if (tolower(args$featureNature) %in% c("dhs-tss", 'tss-dhs')) {
                args$mode <- unlist(strsplit(args$mode, ','))
                names(args$mode)[grep('std|downstream', tolower(args$mode))] <- 'TSS'
                names(args$mode)[!is.na(suppressWarnings(as.numeric(args$mode)))] <- 'DHS'
                winbp = as.numeric(args$mode['DHS'])
                args$mode['DHS'] <- paste0(winbp, 'bp')
                if (length(grep('TSS|DHS', names(args$mode))) != 2) {
                    print_help(opt_parser)
                    stop("The argument in --mode for TSS-DHS sorting is not good (-m std,100 or -m downstream,200).\n", call.=FALSE)
                } else {
                    args$mode <- c(args$mode['TSS'], args$mode['DHS'])
                }
			} else {
				print_help(opt_parser)
				stop("The --featureNature argument is not supported (-x TF, -x TSS, -x DHS, -x TSS-TF | -x DHS | -x TSS-DHS).\n", call.=FALSE)
			}

    ## Frequency file
        if (!is.null(args$freqFile)) {
            freqFile <- unlist(strsplit(args$freqFile, ','))
            for (i in freqFile) {
                if (!file.exists(i)) {
                    print_help(opt_parser)
                    stop(paste0("The file '", i,  "' in --freqFile does not exist (-q path/to/frequency_file.rds).\n"), call.=FALSE)
                }
            }
            Freq.l <- mclapply(freqFile, readFrequencyMatrixFile, mc.cores=args$threads)

            # test each frequency table against all feature to test if all features are present + names the freq object as the feature name
            FreqName <- sapply(Freq.l, function(i){
                test <- lapply(feature, function(n){
                    return(all(rownames(i) %in% names(n)))
                })
                detectedName <- names(test[test == TRUE])
                if(is.null(detectedName) || length(detectedName) == 0){
                    print_help(opt_parser)
                    stop("The frequency file does not contain any feature in --featureFile.\n", call.=FALSE)
                } else if(length(detectedName) > 1){
                    print_help(opt_parser)
                    stop("The frequency file contains more than one feature in --featureFile.\n", call.=FALSE)
                }
                return(detectedName)
            })
            names(Freq.l) <- FreqName
        } else {
            Freq.l <- NULL
        }

    ## feature pair mode
        if(!tolower(args$pairmode) %in% c(paste0(c('freq'), rep(seq(1, 100), each=2)), 'all')){
            print_help(opt_pars)
            stop("The argument in --pairmode is not supported (-p Freq5 | -p Freq12 | -p all).\n", call.=FALSE)
        } else if(!tolower(args$pairmode) == 'all'){
            if(is.null(Freq.l)) {
                print_help(opt_parser)
                stop(paste0("The --pairmode argument ", args$pairmode," is not supported without a frequency file (-q path/to/frequency_file.rds).\n", call.=FALSE))
            }
        }

    ## right windows
        if (!is.numeric(args$rightwindow)) {
            print_help(opt_parser)
            stop("The argument in --rightwindow is not numeric (-r 40).\n", call.=FALSE)
        } else if(args$rightwindow <= 0) {
            print_help(opt_parser)
            stop("The argument in --rightwindow is lower than or equal 0 (-r 40).\n", call.=FALSE)
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
	cat(paste0("\nGenerating ", paste(names(args$mode), collapse='-') ," feature pairs...\n"))
    # Make pairs
        RtP <- resize(GRangesList(feature), 1, fix='center') # Convert features to single-point GRanges centered on each feature
        RefFreq <- tolower(args$mode) # Get lowercase mode names for reference frequency files
        focus=list(TSS=c('polII', 'PIC.polII'), TF='bound', DHS='accessible') # Define focus states for each feature type (what biological state to look for)
        exclude=c(TSS=0.05, TF=0.015, DHS=winbp*1e-3) # Set exclusion distances (in kb) around features to avoid self-pairing and overlapping bins
        sub=list(TSS=c(), TF=c(), DHS=c()) # Define subsets for each feature type (empty for now, can be used to filter specific features)

        # Create file naming convention based on focus states and feature types
        focusFileNames=paste(sapply(focus[names(RefFreq)], function(k){k[1]}), names(RefFreq), sep='_', collapse='-')
        filePairs <- paste0(featureName, '_', focusFileNames,'pairs_win', args$rightwindow,'kb_excl', paste(exclude[names(RefFreq)], collapse='_and_'),'kb_', tolower(args$pairmode),'.rds')
        if(tolower(args$pairmode) %in% paste0(c('freq'), rep(seq(1, 100), each=2))){
            filePairs <- gsub('.rds', paste0('_RefFreq_', paste(RefFreq, collapse='_'),'.rds'), filePairs)
        }

		if(!file.exists(filePairs) | args$overwrite){
            Pairs <- makeFeaturePairs(TFBS=RtP[names(RefFreq)], win=args$rightwindow, exclude=exclude[names(RefFreq)], Pairmode=tolower(args$pairmode),
                siteSub=sub[names(RefFreq)], Freq=Freq.l[names(RefFreq)], threads=args$threads, sortedFeature = focus[names(RefFreq)])
            #Pairs <- extraMetaLRPairs(Pairs=Pairs, tiles=tiles, threads=args$threads)
			qsave(Pairs, filePairs)
			rm(filePairs)
		} else {
			stop(paste0(filePairs, ' already exists... Use -w argument to overwrite.'))
		}

# Save Rsession info
	getVersion(args$taskMeta)
