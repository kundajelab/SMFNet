#!/bin/rscript

# Add the working directory in the libPath to add the user's BSgenome installation
	.libPaths( c( './', .libPaths() ) )

# Loading arguments
	if(!suppressMessages(require("optparse", character.only = TRUE))) stop("Package not found: optparse")

	option_list = list(
		make_option(c("-g", "--genome"), type="character", default=NULL, help="Name of BSgenome to use [required]", metavar="character"),
		make_option(c("-e", "--expName"), type="character", default=NULL, help="Name of the SMF-ONT experiment [required]", metavar="character"),
		make_option(c("-f", "--filterTag"), type="character", default=NULL, help="Filter tag to use [required]", metavar="character"),
		make_option(c("-m", "--mode"), type="character", default='cgp,gpc', help="String of the context methylation used", metavar="character"),
		make_option(c("-t", "--threads"), type="integer", default=12, help="Number of thread to use (>0)", metavar="number"),
		make_option(c("-z", "--taskMeta"), type="character", default='readStats_report', help="Name of the current task for versioning", metavar="character")
	)
	opt_parser = OptionParser(option_list=option_list, description="Script to call methylated cytosines out of SMF-Nanopore data.")
	args <- parse_args(opt_parser)

# Loading libraries
	dep.l <- c('this.path', 'rmarkdown')
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

	## Sample name
		EXPNAME = ifelse(
			is.null(args$expName),
			paste0(format(Sys.time(), "%Y-%m-%d_%H-%M"), '_SMF_bulk_analysis'),
			args$expName
		)

	## 
		if(is.null(args$filterTag)){
			print_help(opt_parser)
			stop("An argument in --filterTag is required (-f MethUnfiltered ).\n", call.=FALSE)
		}

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

# Make ReadStat report
	template <- paste0(this.dir(), '/../assets/readStat_template.Rmd')
	reportFile <-  paste0('ReadStat_', args$expName, '_', args$filterTag, '_', contextView, '.Rmd')
	
	toChange <- list(
		EEEEEE=args$expName,
		GGGGGG=args$genome,
		FFFFFF=args$filterTag,
		ZZZZZZ=args$threads,
		CCCCCC=contextView,
		PPPPPP=this.dir()
	)

	invisible(file.copy(template, reportFile))
	trash <- lapply(seq_along(toChange), function(num){
		lines_text <- readLines(reportFile)
		out_lines <- gsub(names(toChange[num]), toChange[[num]], lines_text)
		writeLines(out_lines, reportFile)	
	})
	
	rmarkdown::render(reportFile,
		output_format='html_document',
		output_dir='./',
		output_file = paste0('Report_ReadStat_', args$expName, '_', args$filterTag, '_', contextView, '.html')
	)

# Save Rsession info
	getVersion(args$taskMeta)
