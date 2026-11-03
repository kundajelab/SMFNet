#!/bin/rscript

# Add the working directory in the libPath to add the user's BSgenome installation
	.libPaths( c( './', .libPaths() ) )

# Loading arguments
	if(!suppressMessages(require("optparse", character.only = TRUE))) stop("Package not found: optparse")

	option_list = list(
	## required	
		make_option(c("-g", "--genome"), type="character", default=NULL, help="Name of BSgenome to use [required]", metavar="character"),

	## optional
		make_option(c("-i", "--InputFolder"), type="character", default='./', help="Path to folder containing all bed.gz / tbi files", metavar="/path/to/MethycallFolder/"),
		make_option(c("-e", "--expName"), type="character", default=NULL, help="Name of the SMF-ONT experiment", metavar="character"),
		make_option(c("-m", "--mode"), type="character", default='cpg,gpc', help="Vector (comma separated) of methylation context to analysis", metavar="character"),
		make_option(c("-w", "--overwrite"), action="store_true", default=FALSE, help="Overwrite existing experiment [default= %default]"),
		make_option(c("-t", "--threads"), type="integer", default=12, help="Number of thread to use (>0)", metavar="Positive number"),
		make_option(c("-z", "--taskMeta"), type="character", default='smf_bulk', help="Name of the current task for versioning", metavar="character")
	)
	opt_parser = OptionParser(option_list=option_list, description="Script to call methylated cytosines out of SMF-Nanopore data.")
	args <- parse_args(opt_parser)

# Loading libraries
	dep.l <- c('this.path', 'qs', 'parallel', 'readr', 'dplyr', 'magrittr')
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

	## thread
		if (!is.numeric(args$threads)) {
			print_help(opt_parser)
			stop("The argument in --threads is not numeric (-t 10).\n", call.=FALSE)
		} else if(args$threads < 1) {
			print_help(opt_parser)
			stop("The argument in --threads is lower than 1 (-t 10).\n", call.=FALSE)
		}

# Create/Load GRanges all potential methylation sites (CpGs / CpGs and/or A)
	ref_file = paste0('context_methylation_ref_', args$genome, '_', contextView,'.qs')
	if(!file.exists(ref_file)){
		AllC <- AllSitegr(tiles=tiles, BSgenome=GENOME, mode=motifs, threads=args$threads)
	} else {
		AllC <- qread(ref_file)
	}
	subsetIndex=list(
		GCi = AllC$type == 'GCH',
		CGi = AllC$type == 'HCG',
		GCGi = AllC$type == 'GCG',
		Ai = AllC$type == 'A'
	)

# MAIN: Create GRanges object containing all methylation calls of the experiment
	NanoporeRDS <- paste0('BulkSMF_', EXPNAME, '_', contextView, '.qs')

	if(!file.exists(NanoporeRDS) | args$overwrite){
		files <- list.files(args$InputFolder, pattern='.bed.gz$', full.names=T)
		files <- files[grepl('freqs_|collapsed', files)] # keep only files with frequencies

		if(length(files) == 0){
			print_help(opt_parser)
			stop("No bed.gz file found in the input folder.\n", call.=FALSE)
		}
		
		NanoAllCs.l <- mclapply(files, function(f){
			filterN <- basename(f) %>%
						tools::file_path_sans_ext() %>%
						gsub('^freqs_std_collapsed_|^freqs_|\\.bed$', '', .)
			# reading frequency bed
				data=read_tsv(f, show_col_types = FALSE, progress = FALSE)
				# correct chr names if necessary to match with the BSgeonme names (when exotic reference fasta are used...)
					if(gsub(".*?([[:alpha:]]+).*", "\\1", data$chr[1]) != 'chr'){
						cat('Trying to fix chromosome name issue...\n')
						if(gsub(".*?([[:alpha:]]+).*", "\\1", data$chr[1]) == 'CHR'){
							data$chr <- gsub('CHR', 'chr', data$chr)
						} else if(is.na(suppressWarnings(tryCatch(as.numeric(substr(data$chr[1], 1, 1)))))){
							message('Not able to fix Chromosome names from input data.\nMight induce problem in the forward analysis!\nContinuing...')
						} else {
							data$chr <- paste0('chr', data$chr)
							data$chr[data$chr == 'chrmitochondrion_genome'] <- 'chrM'
						}
					}
				data$methylated_frequency <- round(data$methylated_frequency, 5)
				if(!'strand' %in% colnames(data)){
					data$strand <- '*'
				} else {
					if(unique(data$strand) == '.'){
						data$strand <- '*'
					}
				}
				NanoAllC_gr <- GRanges(seqnames = as.character(data$chr),
					ranges = IRanges(start = data$start , end = data$end),
					strand = data$strand,
					Met_rate=data$methylated_frequency,
					read_count=data$called_sites) %>%
				.[seqnames(.) %in% norm_chr] %>% sort(.,  ignore.strand=T)

				# collapse strand info for the same base (inherent to modkit output)
				if(length(unique(as.character(strand(NanoAllC_gr)))) > 1){
					tmp <- NanoAllC_gr
					names(tmp) <- paste0('site_', seq_along(tmp))
					strand(tmp) <- '*'
					dup <- tmp[duplicated(tmp),0]

					dup.l <- lapply(c('+', '-'), function(strd){
						strand(dup) <- strd
						return(subsetByOverlaps(NanoAllC_gr, dup))
					})
					dup$read_count = dup.l[[1]]$read_count + dup.l[[2]]$read_count
					dup$Met_rate = round(((dup.l[[1]]$Met_rate*dup.l[[1]]$read_count) + (dup.l[[2]]$Met_rate*dup.l[[2]]$read_count))/dup$read_count, 5)
					mcols(dup) <- mcols(dup)[,c('Met_rate', 'read_count')]

					tmp <- tmp[!duplicated(tmp, fromLast=T)]
					mcols(tmp)[names(dup),] <- mcols(dup)

					NanoAllC_gr <- tmp
					rm(dup, dup.l, tmp)
				} else {
					strand(NanoAllC_gr) <- '*'
				}

				Cs.out.Methcaller = nrow(elementMetadata(NanoAllC_gr))
				Cs.out.Methcaller.dm.rm = nrow(elementMetadata(NanoAllC_gr)[!duplicated(NanoAllC_gr),])

			# Assign Methylation rate and read count to all Cs
				data <- AllC
				ovCs=findOverlaps(NanoAllC_gr, data, ignore.strand=T)
				data$Met_rate <- as.numeric(NA)
				data$read_count <- as.numeric(NA)
				data$Met_rate[subjectHits(ovCs)] <- elementMetadata(NanoAllC_gr[queryHits(ovCs)])[,'Met_rate']
				data$read_count[subjectHits(ovCs)] <- elementMetadata(NanoAllC_gr[queryHits(ovCs)])[,'read_count']
				colnames(elementMetadata(data)) <- c('type', paste0(filterN, '.', c('Met_rate', 'read_count')))
			
			# QC coverage
				cov = summary(is.na(elementMetadata(data)[,2]))
				report <- data.frame(expNames = filterN,
					Cs.out.Methcaller=Cs.out.Methcaller,  
					Cs.out.Methcaller.dm.rm=Cs.out.Methcaller.dm.rm, 
					Cs.AllCObject=as.numeric(cov['FALSE']),
					No.cov.Cs=as.numeric(cov['TRUE'])
				)
			
			return(list(data=data, report=report))
		}, mc.cores=args$threads)
		
		# write report
			write_tsv(do.call(rbind, lapply(NanoAllCs.l, function(i){i$report})), 'qc.tsv', col_names=T)
		
		# collapse all data and save it
			NanoAllCs.l <- lapply(NanoAllCs.l, function(i){i$data})
			NanoAllCs <- NanoAllCs.l[[1]]
			if (length(NanoAllCs.l) > 1) {
				for (k in NanoAllCs.l[2:length(NanoAllCs.l)]) {elementMetadata(NanoAllCs) <- cbind(elementMetadata(NanoAllCs) , elementMetadata(k)[2:3])}
			}
			qsave(NanoAllCs, NanoporeRDS)
	} else {
		stop(paste0(NanoporeRDS, ' already exists... Use -w argument to overwrite.'))
	}

# Save Rsession info
	getVersion(args$taskMeta)
