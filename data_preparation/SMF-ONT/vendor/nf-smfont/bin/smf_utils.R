options(bitmapType='cairo')

######################### Common  ##########################

	# extract subset data from Bgzip bed file using tabix
	#' Function to extract regions of interest from a tabix-indexed bed.gz file
	#'
	#' @param file Path the bgzipped bed file to parse
	#' @param regions GRanges or string (chr2L:2000-5000) with the coordinates of the regions to extract
	#' @param strict restrict to bed lines within the regions of interest
	#' @param col_names string of the column names contained in the bed file (automatic naming of the 3 first column)
	#'
	#' @import GenomicRanges
	#' @import tibble
	#' @importFrom Rsamtools scanTabix
	#' @importFrom dplyr distinct bind_rows
	#' @importFrom readr type_convert
	#'
	#' @return Return a tibble or dataframe containing the bed entries mapped only for the regions of interest
	#'
	getFromTabix <- function(file, regions, strict=FALSE, col_names=NULL, tibble=FALSE){
		dep.l <- c('Biostrings', 'GenomicRanges', 'tibble', 'dplyr', 'Rsamtools', 'readr')
		for (i in dep.l) {if(!suppressMessages(require(i, character.only = TRUE))) stop(paste0("Package not found: ", i))}

		# extract region of interest using tabix
		errMsg <- "The regions argument syntax isn't correct!\nMust be like: 'chr2L:1-3000,chr3R:400-8000'\n"
		if (!class(regions) == "GRanges"){
			if(length(grep(":", regions)) == 0 | length(grep("-", regions)) == 0) {stop(errMsg)}
			splitregion <- unlist(strsplit(regions, ','))
			regions <- unlist(GRangesList(lapply(splitregion, function(r){
				tmp = unlist(strsplit(r, ":"))
				coord = as.numeric(unlist(strsplit(tmp[2], "-")))
				return(GenomicRanges::GRanges(tmp[1], IRanges(start=coord[1], end=coord[2]), strand='*'))
			})))
		}
		reads.list = Rsamtools::scanTabix(file, param=regions)
		ncolBED <- length(unlist(strsplit(reads.list[[1]][1], "\t")))
		reads.df <- do.call(dplyr::bind_rows, lapply(seq_along(reads.list), function(l){
			# conversion to tibble
				df <- tibble::as_tibble(do.call(rbind, strsplit(reads.list[[l]], "\t")), .name_repair = ~ paste0('X', seq(1:ncolBED)))
				df <- suppressMessages(readr::type_convert(df)) %>% dplyr::distinct()

			# filter for overlapping reads
				if(strict & dim(df)[1] > 0){
					df <- df %>% filter(V2 > start(regions)[l])  %>% filter(V3 < end(regions)[l])
				}
			return(df)
		}))
		# correct column names if necessary
		if(!is.null(col_names)){
			if(dim(reads.df)[1] == 0){
				reads.df <- as_tibble(matrix(nrow = 0, ncol = length(col_names)), .name_repair = ~ col_names)
			} else {
				colnames(reads.df)  <- col_names
			}
		} else {
			colnames(reads.df) <- c('chr', 'start', 'end', paste0('unknown_', seq(dim(reads.df)[2]-3)))
		}
		if(tibble){
			return(reads.df)
		} else {
			return(as.data.frame(reads.df))
		}
	}

	# plotting
		dep.l <- c('ggplot2', 'RColorBrewer', 'tools')
		for (i in dep.l) {if(!suppressMessages(require(i, character.only = TRUE))) stop(paste0("Package not found: ", i))}

		jet.colors <- grDevices::colorRampPalette(c("#00007F", "blue", "#007FFF", "cyan","#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"))

		panel.jet <- function(...){
			smoothScatter(..., nrpoints=0, add=TRUE, colramp=jet.colors)
		}

		panel.cor <- function(x, y, digits=3, prefix="", cex.cor, ...){
			usr <- par("usr"); on.exit(par(usr))
			par(usr = c(0, 1, 0, 1))
			r <- abs(cor(x, y, use="pairwise.complete.obs"))
			txt <- format(c(r, 0.123456789), digits=digits)[1]
			txt <- paste(prefix, txt, sep="")
			if(missing(cex.cor)) cex.cor <- 0.4/graphics::strwidth(txt)
			text(0.5, 0.5, txt, cex=cex.cor)
		}

		panel.hist <- function(x, ...){
			usr <- par("usr")
			on.exit(par(usr))
			par(usr = c(usr[1:2], 0, 1.5))
			h <- hist(x, plot = FALSE)
			breaks <- h$breaks; nB <- length(breaks)
			y <- h$counts
		  y <- y/max(y)
			rect(breaks[-nB], 0, breaks[-1], y, col="grey", ...)
		}

		makePairsPlot <- function(data, file, threads=1){
			# Pairs
				if(file_ext(file) == 'png'){
					png(file, width = 4960, height = 4960, res=600)
				} else {
					pdf(file, width = 12, height = 12)
				}
				suppressWarnings(pairs(data, upper.panel = panel.cor, diag.panel = panel.hist, lower.panel = panel.jet))
				dev.off()

			#distribution plot
				data <- mclapply(1:ncol(data), function(r){
					data.frame(met=data[,r], expName=rep(colnames(data)[r], length(data[,r])))
					}, mc.cores=args$threads)
				data <- do.call(rbind, data)
				data <- data[!is.na(data$met),]

				myplot <- ggplot(data, aes(x=met, color=expName)) +
				geom_density(adjust=2) +
				labs(title="Methylation distributions", x = "Methylation rate") +
				scale_fill_brewer(palette="Dark2") +
				theme_classic() +
				theme(legend.position = c(0.8,0.8), legend.title = element_blank())
				ggsave(filename=gsub('.png|.pdf', '_Distrib.pdf', file), plot=myplot, width = 8, height = 8)
		}

	# wrapper dealing with multi format feature input
	readFeatureFile <- function(file, bsgenome=NULL){
		dep.l <- c('qs', 'rtracklayer', 'readr', 'readxl', 'tools')
		for (i in dep.l) {if(!suppressMessages(require(i, character.only = TRUE))) stop(paste0("Package not found: ", i))}

        featureName = tools::file_path_sans_ext(basename(file))
	    extension <- tools::file_ext(file)
		if(extension == 'gz'){
			extension <- tools::file_ext(gsub('.gz', '', file))
        }

		if(extension %in% c('tsv', 'txt', 'xls', 'xlsx', 'csv')) {
			if (extension %in% c('xls', 'xlsx')) {
				feature <- read_excel(file)
			} else {
				feature <- read_delim(file, col_names = TRUE)
			}

			feature <- tryCatch({
				feature <- GRanges(feature)
				},
				error = function(e){
					print(e)
					stop(paste0('The feature file provided "', basename(file),
					'" doesn\'t contain correct headers to convert it in GRanges object.\n'), call.=FALSE)
				}
			)
		} else if(extension %in% c('rds', 'qs')) {
			if(extension== 'rds') {
				feature <- readRDS(file)
			} else {
				feature <- qread(file)
			}
			if(!class(feature) == 'GRanges') {
				stop(paste0('The R object provided in the file "', basename(file), '" is not a GRanges object.\n'), call.=FALSE)
			}
		} else if(extension == 'bed') {
			feature <- import.bed(file)
		} else if(extension %in% c('bedgraph', 'bg')) {
			feature <- import.bedGraph(file)
		} else {
			stop("The file extension provided in --featureFile isn't supported (-f path/to/feature_file.qs).\n", call.=TRUE)
		}
        if(!is.null(bsgenome)){
            if(!suppressMessages(require("BSgenome", character.only = TRUE))) stop("Package not found: BSgenome")
			if(!suppressMessages(require(bsgenome, character.only = TRUE))) stop(paste0("Package not found: ", bsgenome))
            seqinfo(feature) <- seqinfo(get(bsgenome))
        }
		return(list(feature=feature, featureName=featureName))
	}

    # wrapper to read frequency file
    readFrequencyMatrixFile <- function(file){
        dep.l <- c('qs', 'readr', 'tools')
        for (i in dep.l) {if(!suppressMessages(require(i, character.only = TRUE))) stop(paste0("Package not found: ", i))}
        extension <- tools::file_ext(file)
        if(extension == 'gz'){
			extension <- tools::file_ext(gsub('.gz', '', file))
        }
        if(extension %in% c('tsv', 'txt', 'csv')) {
            frequency <- read_delim(file, col_names = TRUE)
        } else if(extension %in% c('rds', 'qs')) {
            if(extension[length(extension)] == 'rds') {
                frequency <- readRDS(file)
            } else {
                frequency <- qread(file)
            }
        } else {
            stop("The file extension provided in --frequencyFile isn't supported (-f path/to/frequency_file.qs).\n", call.=TRUE)
        }
        # check if list and flatten if so
        if(is.list(frequency)) {
            frequency <- frequency[[1]]
        }

        # check if only one column is character and any others are numeric
        if(ncol(frequency) < 2) {
            stop(paste0('The frequency file "', basename(file), '" must contain at least two columns.\n'), call.=FALSE)
        }
        if(!any(sapply(frequency, is.character)) && !any(sapply(frequency, is.numeric))) {
            stop(paste0('The frequency file "', basename(file), '" must contain at least one character and one numeric column.\n'), call.=FALSE)
        }
        return(frequency)
    }

	# Remove Palindrome motif from feature file
	removePalin <- function(gR, threads, colname='name'){
		require('parallel')
		rmName <- FALSE
		if(is.null(names(gR))){
			rmName <- TRUE
			names(gR) <- as.character(seq_along(gR))
		}
		dups <- unlist(mclapply(unique(gR$name), function(k){
			data <- gR[mcols(gR)[,colname] == k]
			strand(data) <- '*'
			return(names(data[duplicated(data)]))
		}, mc.cores=threads))
		out <- gR[!names(gR) %in% dups]
		if(rmName){
			names(out) <- NULL
		}
		return(out)
	}

	string.split <- function(string,sep,pos){unlist(lapply(string,function(x){lapply(strsplit(x,sep),'[',pos)}))}

	string.split.last <- function(string, sep){
		if(!suppressMessages(require('stringr', character.only = TRUE))) stop("Package not found: stringr")
		loc <- str_count(string, sep)
		splitSp <- strsplit(string, sep)
		last <- unlist(lapply(seq_along(splitSp), function(i){splitSp[[i]][loc[i] + 1]}))
		return(last)
	}

	getVersion <- function(taskName){
		# Rsession Log
			sink(paste0(taskName, ".R_sessionInfo.log"))
			print(sessionInfo())
			sink()

		# Version file
			loaded.p <- sort(loadedNamespaces())
			r.version <- paste('    r-base:', strsplit(base::version[['version.string']], ' ')[[1]][3])
			dep.version <- unlist(lapply(loaded.p[loaded.p != 'base'], function(p){
				return(
					paste0('    r-', p, ': ', as.character(packageVersion(p)))
				)
			}))

			writeLines(c(paste0(taskName, ':'), r.version, dep.version), 'versions.yml')
	}

#################### Bulk SMF analysis  ####################

	#' Function to fix create a GRanges object containing all Methylation site expected for the SMF analysis
	#'
	#' @param tiles single-molecule cytosine methylation data frame
	#' @param BSgenome mode to keep ambiguous context methylation value {cpg|gpc|strict}
	#' @param mode single-molecule cytosine methylation data frame
	#' @param threads mode to keep ambiguous context methylation value {cpg|gpc|strict}
	#'
	#' @import parallel
	#' @import Biostrings
	#' @import GenomicRanges
	#' @import GenomeInfoDb
	#'
	#' @return GRanges object containing all reference site for methylation
	#'
	AllSitegr <- function(tiles, BSgenome, mode=c('cpg', 'gpc'), threads=2){
		dep.l <- c('parallel', 'Biostrings', 'GenomicRanges')
		for (i in dep.l) {if(!suppressMessages(require(i, character.only = TRUE))) stop(paste0("Package not found: ", i))}

		motSearchs <- lapply(tolower(mode), function(mot){
			if(mot == 'cpg'){
				mot='CG';pos='start'
			} else if(mot == 'gpc'){
				mot='GC';pos='end'
			} else if(mot == 'a'){
				mot=c('A', 'T');pos='start'
			}

			motSearch <- unlist(GenomicRanges::GRangesList(mclapply(mot, Biostrings::vmatchPattern, BSgenome, mc.cores=threads)))
			motSearch <- motSearch[GenomicRanges::strand(motSearch) == "+",]
			GenomicRanges::strand(motSearch) <- "*"
			motSearch <- GenomicRanges::resize(motSearch, 1, fix=pos)
			motSearch <- motSearch[motSearch %over% tiles]
			motSearch$type <- mot[1]

			GenomeInfoDb::seqlevels(motSearch) <- GenomeInfoDb::seqlevels(tiles)
			GenomicRanges::seqinfo(motSearch) <- GenomicRanges::seqinfo(tiles)
			return(sort(motSearch, ignore.strand=TRUE))
		})
		names(motSearchs) <- tolower(mode)

		#Assign context to Cs and collapse GCG info
			if(!is.null(motSearchs$cpg) && !is.null(motSearchs$gpc)){
				oGCG=GenomicRanges::findOverlaps(subject=motSearchs$cpg, query=motSearchs$gpc, type='equal')
				motSearchs$gpc$type='GCH'
				motSearchs$gpc <- motSearchs$gpc[-queryHits(oGCG)]
				motSearchs$cpg$type='HCG'
				motSearchs$cpg$type[subjectHits(oGCG)]='GCG'
			}

		#Collapse multiple motif info
			AllC=sort(unlist(GRangesList(motSearchs), use.names=FALSE), ignore.strand=TRUE)
			names(AllC) <- paste0('site_', seq_along(AllC))

		return(AllC)
	}

	#' Function to fix create a GRanges object containing all Methylation site expected for the SMF analysis
	#'
	#' @param AllC GRanges object containing expected methylation sites on the positive strand
	#'
	#' @import GenomicRanges
	#'
	#' @return GRanges object containing all reference site for methylation on the negative strand
	#'
	getRevAllC <- function(AllC){
		dep.l <- c('GenomicRanges')
		for (i in dep.l) {if(!suppressMessages(require(i, character.only = TRUE))) stop(paste0("Package not found: ", i))}
		revAllC <- AllC[AllC$type %in% c('HCG', 'GCH', 'GCG')] # subset C meth call
		if(length(revAllC) > 0){
			# locate the C in the second strand
				revAllC <- suppressWarnings(GenomicRanges::resize(revAllC, 3, fix='center'))
				start(revAllC) <- ifelse(revAllC$type == 'HCG', start(revAllC) + 1, start(revAllC))
				end(revAllC) <- ifelse(revAllC$type == 'GCH', end(revAllC) - 1, end(revAllC))
			# solve GCG / CGC context (assigning on CG context only)
				revAllC$secCinCGC <- unlist(lapply(unique(seqnames(revAllC)), function(chr){
					tmp <- revAllC[seqnames(revAllC) == chr]
					dist= c(-1, start(tmp)[-1] - start(tmp)[-length(tmp)])
					return(dist == 1)
				}))
				start(revAllC[revAllC$secCinCGC]) <- start(revAllC[revAllC$secCinCGC])+1
		}
		return(revAllC)
	}

	#' Function to fix create a GRanges object containing all Methylation site expected for the SMF analysis
	#'
	#' @param data GRanges object containing expected methylation sites including a 'read_count' column
	#' @param minCov GRanges object containing expected methylation sites on the positive strand
	#'
	#' @import GenomicRanges
	#'
	#' @return GRanges object containing all reference site for methylation on the negative strand
	#'
	CovFilter <- function(data, minCov){
		dep.l <- c('GenomicRanges')
		for (i in dep.l) {if(!suppressMessages(require(i, character.only = TRUE))) stop(paste0("Package not found: ", i))}
		colToGrep <- grep('read_count', colnames(GenomicRanges::elementMetadata (data)))
		read_counts <- as.data.frame(GenomicRanges::elementMetadata (data)[, colToGrep])
		ToNa <- lapply(1:ncol(read_counts), function(i){
			x <- read_counts[,i] < minCov
			x[is.na(x)] <- FALSE
			x
		})

		for (k in 1:length(ToNa)) {
			colMet <- colToGrep[k]-1
			GenomicRanges::elementMetadata (data)[ToNa[[k]], colMet] <- NA
		}
		return(data)
	}

	#' Function to create a BigWig file from a GRanges object containing all Methylation site expected for the SMF analysis
	#'
	#' @param gR GRanges object containing expected methylation sites including a 'read_count' column
	#' @param feature string of the column name to use
	#' @param expName Experiment name
	#' @param filename BigWig file to write
	#' @param vpPos
	#' @param vpChr
	#' @param plotView
	#' @param genome
	#' @param fixed
	#' @param bin
	#' @param invert If TRUE, invert the methylation signal (default = TRUE)
	#'
	#' @import rtracklayer
	#' @import GenomicRanges
	#' @import BSgenome
	#'
	#' @return GRanges object containing all reference site for methylation on the negative strand
	#'
	exportBW <- function(gR, feature, expName, filename, vpPos=5727525, vpChr='chr2R', plotView=50000, BSgenome, fixed=FALSE, bin=10, invert=TRUE){
		dep.l <- c('rtracklayer', 'GenomicRanges', 'GenomicAlignments', BSgenome)
		for (i in dep.l) {if(!suppressMessages(require(i, character.only = TRUE))) stop(paste0("Package not found: ", i))}
		GENOME <- get(BSgenome)

		# Input validation
		if(!feature %in% colnames(mcols(gR))) {
			stop(paste0("Feature '", feature, "' not found in GRanges object"))
		}

		gzname <- gsub('.bw', '.wig.gz', filename)
		gz <- gzfile(gzname, "w")

		browserPos <- paste0(vpChr,":", vpPos-plotView, "-", vpPos+plotView)
		positionLine <- paste0("browser position ", browserPos, "\n")
		cat(positionLine, file=gz)
		trackLine <- paste0("track type=wiggle_0 name=", expName, " visibility=full autoScale=off  viewLimits=0.0:100.0 maxHeightPixels=50:50:8 color=0,0,200  priority=10\n")
		cat(trackLine, file=gz, append=TRUE)
		if(fixed){
			# Create non-overlapping bins across the genome
			BinGR <- GenomicRanges::tileGenome(seqinfo(GENOME), tilewidth=bin, cut.last.tile.in.chrom=TRUE)
			gRf <- BinGR
			gRf$score <- as.numeric(NA)

			# Find which input features (gR) overlap with which bins (gRf)
			# Note: bins are non-overlapping, but multiple features can fall into the same bin
			ov <- as.matrix(GenomicAlignments::findOverlaps(gR, gRf, ignore.strand=T))

			# Handle multiple input features per bin by taking the mean
			if(nrow(ov) > 0) {
				# make methylation signal to 100 and invert if asked (default = TRUE)
				scores_to_assign <- if(invert){
					100 - round((mcols(gR[ov[,1]])[, feature] * 100), 3)
				} else {
					round((mcols(gR[ov[,1]])[, feature] * 100), 3)
				}

				# Handle multiple features per bin by taking mean
				score_df <- data.frame(bin_idx = ov[,2], score = scores_to_assign)
				mean_scores <- aggregate(score ~ bin_idx, score_df, mean, na.rm=TRUE)
				mcols(gRf[mean_scores$bin_idx])[,1] <- mean_scores$score
			}

			NAtozero=is.na(mcols(gRf)[,1])
			mcols(gRf[NAtozero])[,1] <- 0

			if(bin > 1) {
				#Remove the last tile of each chromosome if its length is not equal to the bin size
				gRf <- gRf[-which(end(gRf) - start(gRf) != bin-1)]
			}
			data <- GenomicRanges::GRanges(
				seqnames=seqnames(gRf),
				ranges=IRanges::IRanges(start=start(gRf), end=end(gRf)),
				strand = "*", # strand is not relevant for bigWig in binned data
				score=gRf$score)
		} else {
			# make methylation signal to 100 and invert if asked (default = TRUE)
			scores=round((mcols(gR)[,feature]*100), 3)
			if(invert){
				scores=100-scores
			}
			NAi=is.na(scores) #remove uncovered
			gRf=gR[!NAi]
			data <- GenomicRanges::GRanges(
				seqnames=seqnames(gRf),
				ranges=IRanges(start=start(gRf), end=end(gRf)),
				strand = strand(gRf),
				score=scores[!NAi])
		}

		rtracklayer::export.wig(data, gz, append = TRUE)
		close(gz)

		rtracklayer::wigToBigWig(gzname, seqinfo(GENOME))
		unlink(gzname, force=TRUE)
	}

################# Single Molecule Sorting  #################

	# make cluster object from input TFBS motifs
	Make_TFBS_cluster = function(TFBS, maxDist = 75, minDist = 15, maxClusterSize = 10) {
		if(!suppressMessages(require("GenomicRanges", character.only = TRUE))) stop("Package not found: GenomicRanges")
		TFBS_center <- resize(TFBS, 1, fix='center')
		ov = findOverlaps(TFBS_center, trim(resize(TFBS_center, maxDist*2, fix='center')), ignore.strand=TRUE)
		pair_dist = start(TFBS_center[subjectHits(ov)]) - start(TFBS_center[queryHits(ov)]) # distance between partners

		ov2 = ov[pair_dist > minDist] # make sure that distance >minDist top avoid overlapping TFBS & remove self & consider only positive distances (object is symmetrical)

		TF_cluster = reduce(GRanges(
				seqnames=seqnames(TFBS_center[queryHits(ov2)]),
				range=IRanges(start(TFBS_center[queryHits(ov2)]), start(TFBS_center[subjectHits(ov2)])),
				strand='*',
				seqinfo=seqinfo(TFBS)
			)) # start with the  first motif center and end with the last motif center

		TF_cluster$num_TF = countOverlaps(TF_cluster, TFBS_center) # get number of TFBS per cluster
		names(TF_cluster) = paste0('TFBS_cluster_', 1:length(TF_cluster)) # unique name of clusters

		TF_cluster = TF_cluster[TF_cluster$num_TF <= maxClusterSize] # remove to much dense cluster

		ov3 = findOverlaps(TF_cluster, TFBS_center, ignore.strand=T) # get all TFBS included in each cluster
		TF_list = split(TFBS[subjectHits(ov3)], queryHits(ov3)) # get list of TFBS included in each cluster
		names(TF_list) = names(TF_cluster)

		return(list(ClusterGr = TF_cluster, Clusterlist = TF_list))
	}

	# Make bins around TFBS/TSS features from absolute coordinates
	MakeBins <- function(RoI, viewPoint, binAbsPos, binName, rowNames=NULL, rmExtra=FALSE){
		bin <- GRanges(seqnames = seqnames(RoI),
			ranges = IRanges(
				start = ifelse(as.logical(strand(RoI) == '+'), viewPoint + binAbsPos[1], viewPoint - binAbsPos[2]),
				end = ifelse(as.logical(strand(RoI) == '+'), viewPoint + binAbsPos[2], viewPoint - binAbsPos[1])),
			seqinfo = seqinfo(RoI))
		if(!rmExtra){
			bin$name = binName
		}
		if(is.null(rowNames)){
			names(bin) <- names(RoI)
		} else {
			names(bin) <- rowNames
		}
		return(bin)
	}

	sorAnyBins_Nanopore <- function(regDF, SM_files, target_bins, BSgenome, strandCollapsing=FALSE, binSpe=NULL) {
		#reload libraries (important for parallelization)
		if(!suppressMessages(require("BSgenome", character.only = TRUE))) stop("Package not found: BSgenome")
		if(!suppressMessages(require(BSgenome, character.only = TRUE))) stop(paste0("Package not found: ", BSgenome))
		GENOME <- get(BSgenome)

		cat(paste0("	## Single Molecule sorting of the chunk ", rownames(regDF), "\n"))

		#define regions to extract methylation information
		reg <- GRanges(seqnames = Rle(regDF[1,1]), ranges = IRanges(regDF[1,2], end = regDF[1,3]), seqlengths=seqlengths(GENOME), seqinfo=seqinfo(GENOME))
		regExp <- reg # expand the region at the end to catch all the pairs within the window
		end(regExp) <- pmin(end(regExp) + 2000, seqlengths(regExp)[as.character(seqnames(regExp))])

		AllC=NULL
		if(strandCollapsing){
			AllC <- sort(unlist(GRangesList(lapply(c('GC', 'CG'), function(mot){
				if(mot == 'CG'){
					pos='start'; motName='HCG'
				} else {
					pos='end'; motName='GCH'
				}
				tmp <- matchPattern(mot, GENOME[[as.character(seqnames(reg))]])

				return(
					resize(
						GRanges(seqnames = Rle(as.character(seqnames(reg))),
							ranges = IRanges(start(tmp), end = end(tmp)),
							strand="+",
							seqlengths=seqlengths(GENOME),
							seqinfo=seqinfo(GENOME),
							type=motName),
						1, fix=pos)
				)
			}))))
			#identify the GCG
			AllC[start(AllC) %in% start(AllC[duplicated(AllC)])]$type <- 'GCG'
			AllC <- AllC[!duplicated(AllC)]
		}

		#call methylation vectors
		methAln <- .getQuasRqMeth(SM_files, regExp, strandCollapsing=strandCollapsing, AllC=AllC)

		if(length(methAln[[1]]) > 0) { #control that at leas one read is found in the region
			selCs= methAln$strand=="+"

			#Extract the bins included in the windows of observation
			InBin <- unlist(GRangesList(target_bins), use.names= F)
			middlewBin <- unique(InBin$name)[ceiling(length(unique(InBin$name))/2)]
			InBin <- resize(InBin[InBin$name == middlewBin], 1, fix='center')
			InBin <- split(InBin, names(InBin))
			InBin <- InBin[names(target_bins)]

			IncWd = queryHits(findOverlaps(InBin, reg, type='within'))
			binL = target_bins[IncWd]

			sRs = .sortAnyBin_vect_nanopore(methAln=methAln, selCs=selCs, binL=binL, binSpe=NULL)
			return(sRs)
		} else {
			return(list())
		}
	}

	.sortAnyBin_vect_nanopore <- function(methAln, selCs, binL, binSpe=NULL) {
		if(!suppressMessages(require("GenomicRanges", character.only = TRUE))) stop("Package not found: GenomicRanges")
		#create Crange
		Crange=IRanges(methAln[[2]][selCs], methAln[[2]][selCs])

			#create list of the fixed bins
		bin.ul = unlist(binL, use.names=F)
		bins = split(bin.ul, bin.ul$name)
		#bins <- lapply(bins, sort) !! do not re-sort bins

		if(!is.null(binSpe)){
			ToUseBin <- names(bins)[names(bins) %in% binSpe]
			bins=bins[ToUseBin]
		} else {
			bins <- bins[binL[[1]]$name]
		}

		#find overlaping Cs
		ovs <- lapply(bins, function(i){findOverlaps(query=ranges(i), subject=Crange)})

		#compute the methylation vectors
		Mvs <- lapply(seq_along(bins), function(i){methAln[[4]][selCs][subjectHits(ovs[[i]])]})

		#compute the readIDs
		IDvs <- lapply(seq_along(bins), function(i){paste(names(bins[[1]])[queryHits(ovs[[i]])],
			methAln[[1]][selCs][subjectHits(ovs[[i]])], sep='_')})

		#group Cs per region/per_read (need to rm NAs)
		g.Ms <- lapply(seq_along(bins), function(i){round(tapply(Mvs[[i]], IDvs[[i]], mean, na.rm=TRUE))})

		#identify NAs
		NAs.u.IDs.Ms <- lapply(g.Ms, function(i){unique(names(i[is.na(i)]))})

		#find unique IDs
		u.IDs <- lapply(seq_along(bins), function(i){
			uID <- sort(unique(IDvs[[i]]))
			return(uID[!uID %in% NAs.u.IDs.Ms[[i]]])
		})

		#intersect IDS
		intMat <- do.call(cbind, lapply(seq_along(bins), function(i){u.IDs[[1]] %in% u.IDs[[i]]}))
		ids <- seq_along(u.IDs[[1]])[rowSums(intMat) == ncol(intMat)] #makes sur the all bins are covered

		if(length(ids) > 0) {
			#binary met vectors
			sID <- u.IDs[[1]][ids]
			s.M <- lapply(seq_along(bins), function(i){g.Ms[[i]][sID]})

			patternMat <- do.call(cbind, s.M)
			colnames(patternMat) <- names(bins)
			pattern <- apply(patternMat, 1, function(x){paste0(as.character(x), collapse='')})

			# Name features
			read.id <- string.split.last(sID, '_')
			st.id <- unlist(lapply(seq_along(sID), function(i){gsub(paste0('_', read.id[i], '$'), '', sID[i])}))

			# SORTED READ LISTS
			sR = split(read.id, paste(st.id, pattern, sep='_'))
			sRl = split(sR, gsub(paste(unique(paste0('_', string.split.last(names(sR), '_'), '$')), collapse='|'), '', names(sR)))
		} else {
			sRl = list()
		}
		return(sRl)
	}

	sorTFbyReg_allGenomes_Nanopore <- function(regDF, SM_files, target_range, inMs=c(-15,15), upMs=c(-35,-25), doMs=c(25,35), BSgenome="BSgenome.Mmusculus.UCSC.mm10", cluster = FALSE, strandCollapsing=FALSE) {
		#reload libraries (important for parallelization)
		if(!suppressMessages(require("BSgenome", character.only = TRUE))) stop("Package not found: BSgenome")
		if(!suppressMessages(require(BSgenome, character.only = TRUE))) stop(paste0("Package not found: ", BSgenome))
		GENOME <- get(BSgenome)

		cat(paste0("	## Single Molecule sorting of the chunk ", rownames(regDF), "\n"))

		#define regions to extract methylation information
		reg <- GRanges(seqnames = Rle(regDF[1,1]), ranges = IRanges(regDF[1,2], end = regDF[1,3]), seqlengths=seqlengths(GENOME), seqinfo=seqinfo(GENOME))
		regExp <- reg # expand the region at the end to catch all the pairs within the window
		end(regExp) <- pmin(end(regExp)+2000, seqlengths(regExp)[as.character(seqnames(regExp))])

		AllC=NULL
		if(strandCollapsing){
			AllC <- sort(unlist(GRangesList(lapply(c('GC', 'CG'), function(mot){
				if(mot == 'CG'){
					pos='start';motName='HCG'
				} else {
					pos='end';motName='GCH'
				}
				tmp <- matchPattern(mot, GENOME[[as.character(seqnames(reg))]])

				return(
					resize(
						GRanges(seqnames = Rle(as.character(seqnames(reg))),
							ranges = IRanges(start(tmp), end = end(tmp)),
							strand="+",
							seqlengths=seqlengths(GENOME),
							seqinfo=seqinfo(GENOME),
							type=motName),
						1, fix=pos)
				)
			}))))
			#identify the GCG
			AllC[start(AllC) %in% start(AllC[duplicated(AllC)])]$type <- 'GCG'
			AllC <- AllC[!duplicated(AllC)]
		}

		#call methylation vectors
		methAln <- .getQuasRqMeth(SM_files, regExp, strandCollapsing=strandCollapsing, AllC=AllC)

		if(length(methAln[[1]]) > 0) { #control that at leas one read is found in the region
			selCs= methAln$strand=="+"
			if(cluster) {
				ids=unique(string.split(names(subsetByOverlaps(unlist(target_range), reg, type='within')), '[.]', 1))
				TFs=target_range[ids]
				sRs=.sortTF_clusters_reads_vect_nanopore(methAln=methAln, selCs=selCs, st=TFs, inMs=inMs, upMs=upMs, doMs=doMs)
			} else {
				TFs=subsetByOverlaps(target_range, reg, type='within')
				sRs=.sortTFreads_vect_nanopore(methAln=methAln, selCs=selCs, st=TFs, inMs=inMs, upMs=upMs, doMs=doMs)
			}
			return(sRs)
		} else {
			return(list())
		}
	}

	.sortTFreads_vect_nanopore <- function(methAln, selCs, st, inMs=c(-7,7), upMs=c(-35,-25), doMs=c(25,35)) {
		if(!suppressMessages(require("GenomicRanges", character.only = TRUE))) stop("Package not found: GenomicRanges")
		#create Crange
		Crange=IRanges(methAln[[2]][selCs], methAln[[2]][selCs])

			#create collecting intervals
			st=resize(st,1,fix='center')
		midP=start(st)

		#create the bin Grange
		inMP = MakeBins(RoI=st, viewPoint=midP, binAbsPos=inMs, binName='inMs', rmExtra=TRUE)
		upMP = MakeBins(RoI=st, viewPoint=midP, binAbsPos=upMs, binName='upMs', rmExtra=TRUE)
		doMP = MakeBins(RoI=st, viewPoint=midP, binAbsPos=doMs, binName='doMs', rmExtra=TRUE)

		#collapse bin in list
		bins=list(upMP=upMP, inMP=inMP, doMP=doMP)

		#find overlaping Cs
		ovs <- lapply(seq_along(bins), function(i){findOverlaps(query=ranges(bins[[i]]), subject=Crange)})

		#compute the methylation vectors
		Mvs <- lapply(seq_along(bins), function(i){methAln[[4]][selCs][subjectHits(ovs[[i]])]})

		#compute the readIDs
		IDvs <- lapply(seq_along(bins), function(i){paste(names(bins[[1]])[queryHits(ovs[[i]])],
			methAln[[1]][selCs][subjectHits(ovs[[i]])], sep='_')})

		#group Cs per region/per_read (need to rm NAs)
		g.Ms <- lapply(seq_along(bins), function(i){round(tapply(Mvs[[i]], IDvs[[i]], mean, na.rm=TRUE))})

		#identify NAs
		NAs.u.IDs.Ms <- lapply(g.Ms, function(i){unique(names(i[is.na(i)]))})

		#find unique IDs
		u.IDs <- lapply(seq_along(bins), function(i){
			uID <- sort(unique(IDvs[[i]]))
			return(uID[!uID %in% NAs.u.IDs.Ms[[i]]])
			})

		#intersect IDS
		intMat <- do.call(cbind, lapply(seq_along(bins), function(i){u.IDs[[1]] %in% u.IDs[[i]]}))
		ids <- seq_along(u.IDs[[1]])[rowSums(intMat) == ncol(intMat)] #makes sur the all bins are covered

		if(length(ids) > 0) {
			#binary met vectors
			sID <- u.IDs[[1]][ids]
			s.M <- lapply(seq_along(bins), function(i){g.Ms[[i]][sID]})

			patternMat <- do.call(cbind, s.M)
			colnames(patternMat) <- names(bins)
			pattern <- apply(patternMat, 1, function(x){paste0(as.character(x), collapse='')})

			# Name features
			read.id <- string.split.last(sID, '_')
			st.id <- unlist(lapply(seq_along(sID), function(i){gsub(paste0('_', read.id[i], '$'), '', sID[i])}))

			# TABULAR COUNTS
			#p.counts=table(st.id,pattern)

			# SORTED READ LISTS
			sR = split(read.id, paste(st.id, pattern, sep='_'))
			sRl = split(sR, gsub(paste(unique(paste0('_', string.split.last(names(sR), '_'), '$')), collapse='|'), '', names(sR)))
		} else {
			sRl = list()
		}
		return(sRl)
	}

	.sortTF_clusters_reads_vect_nanopore <- function(methAln, selCs, st, inMs=c(-7,7), upMs=c(-35,-25), doMs=c(25,35)){
		#create  Crange
		Crange=IRanges(methAln[[2]][selCs],methAln[[2]][selCs])

		#create collecting intervals
		#get center of each TF binding site
		TFcenter=resize(unlist(st),1,fix='center')
		inMP=GRanges(seqnames(TFcenter), IRanges(start(TFcenter)+inMs[1], start(TFcenter)+inMs[2]))
		names(inMP)=string.split(names(TFcenter),'\\.',2)
		inMPs=split(inMP,string.split(names(TFcenter),'\\.',1))

		#get borders of the clusters
		upMPs=unlist(GRangesList(lapply(st, function(i) {
			minV=order(start(i),decreasing=F)[1]
			return(GRanges(seqnames(i[minV]),IRanges(start(i[minV])+upMs[1],start(i[minV])+upMs[2])))
		})))

		doMPs=unlist(GRangesList(lapply(st, function(i) {
			maxV=order(start(i),decreasing=T)[1]
			return(GRanges(seqnames(i[maxV]),IRanges(start(i[maxV])+doMs[1],start(i[maxV])+doMs[2])))
		})))

		#generate dynamic bins depending on the composition of the TF cluster
			nb_TF=lengths(st)

			#group clusters by number of TFs
			#create dynamic bins
			#d_bins are grouped by binning type (number of TFs in the cluster)
			d_bins=lapply(seq_along(sort(unique(nb_TF))),function(i){
				nbTF=sort(unique(nb_TF))[i]
				TFrange=inMPs[lengths(inMPs)==nbTF]

				se=lapply(seq_along(TFrange),function(cl){
						lapply(seq(nbTF),function(pos){TFrange[[cl]][pos]})
					})

				inMPsd=lapply(seq(nbTF),function(pos){
						do.call(c,lapply(se, `[[`, pos))
					})

				cluster_id=names(st)[i]
				bins=GRangesList(c(list(upMPs[names(TFrange)]),inMPsd,list(doMPs[names(TFrange)])	))
				return(bins)
			})
			names(d_bins)=paste('clusterOf_',sort(unique(nb_TF)),'_TF',sep='')

		#iterate ober bin types
		sRls=lapply(seq_along(d_bins), function(bin_t){
			bins=d_bins[[bin_t]]

			#find overlaping Cs
			ovs <- lapply(seq_along(bins), function(i){findOverlaps(query=ranges(bins[[i]]), subject=Crange)})

			#compute the methylation vectors
			Mvs <- lapply(seq_along(bins), function(i){methAln[[4]][selCs][subjectHits(ovs[[i]])]})

			#compute the readIDs
			IDvs <- lapply(seq_along(bins), function(i){
				paste(names(bins[[1]])[queryHits(ovs[[i]])], methAln[[1]][selCs][subjectHits(ovs[[i]])], sep='_')
			})

			#group Cs per region/per_read
			g.Ms <- lapply(seq_along(bins), function(i){round(tapply(Mvs[[i]], IDvs[[i]], mean, na.rm=TRUE))})

			#identify NAs
			NAs.u.IDs.Ms <- lapply(g.Ms, function(i){unique(names(i[is.na(i)]))})

			#find unique IDs
			u.IDs <- lapply(seq_along(bins), function(i){
				uID <- sort(unique(IDvs[[i]]))
				return(uID[!uID %in% NAs.u.IDs.Ms[[i]]])
			})

			intMat <- do.call(cbind, lapply(seq_along(bins), function(i){u.IDs[[1]] %in% u.IDs[[i]]}))
			ids <- seq_along(u.IDs[[1]])[rowSums(intMat) == ncol(intMat)] #makes sur the all bins are covered

			if(length(ids) > 0) {
				#binary met vectors
				sID <- u.IDs[[1]][ids]
				s.M <- lapply(seq_along(bins), function(i){g.Ms[[i]][sID]})

				patternMat <- do.call(cbind, s.M)
				pattern=apply(patternMat,1,function(x){paste(as.character(x),sep='',collapse='')})

				read.id=string.split.last(sID, '_')
				st.id <- unlist(lapply(seq_along(sID), function(i){gsub(paste0('_', read.id[i], '$'), '', sID[i])}))

				#TABULAR COUNTS
				# p.counts=table(st.id,pattern)

				#SORTED READ LISTS
				sR=split(read.id,paste(st.id,pattern,sep='_'))
				sRl=split(sR, gsub(paste(unique(paste0('_', string.split.last(names(sR), '_'), '$')), collapse='|'), '', names(sR)))
			} else {
				sRl=list()
			}
			return(sRl)
		})
		#merge the results
		return(sRlsm=do.call(c, sRls))
	}

	#' Function to get QuasR::qMeth() like output from single molecule methylation call from Nanopore data
	#'
	#' BLABLAVLA
	#'
	#' @param SM_files data frame ....
	#' @param regExp
	#' @param flankedRegion genomic distance in base-pair to expand the regExp region single molecule methylation call. Defaults to 0.
	#' @param mbedColnames string containing the headers of the single-molecule bed file.
	#' @param modeGCG mode to keep ambiguous context methylation value. Will keep only the CpG (mode='cpg'), GpC (mode='gpc') or none (mode='strict') of methylation value at GCG context. Default to 'cpg'.
	#'
	#' @import GenomicRanges
	#' @importFrom S4Vectors queryHits subjectHits
	#' @importFrom IRanges IRanges reduce
	#'
	#' @return list with 4 elements:
	#' - aid: readID
	#' - Cid: Cytosine coordinates
	#' - strand: context strand
	#' - meth: cytosine methylation call (methylated=1, unmethylated=0, undetermined=NA)
	#'
	.getQuasRqMeth <- function(SM_files, regExp, flankedRegion=0, strandCollapsing=FALSE, AllC=NULL, mbedColnames=c("chrom","start","end","readname","mstring","scores","context"), modeGCG='cpg'){
		QuasRmat <- lapply(1:nrow(SM_files), function(i){
			region.tb <- getFromTabix(file=SM_files[i,1], regions=regExp, col_names=mbedColnames, tibble=FALSE)
			if(dim(region.tb)[1] == 0){
				return(list(data.frame(), GRanges()))
			} else {
				return(list(.mBedToQuasRmat(mBed=region.tb), GRanges(region.tb[,1:4])))
			}
			})
		readOrder <- sort(unlist(GRangesList(lapply(QuasRmat, function(i){i[[2]]}))))
		QuasRmat <- do.call(rbind, lapply(QuasRmat, function(i){i[[1]]}))

		if(nrow(QuasRmat) == 0){
			return(list(aid=c(), Cid=c(), strand=c(), meth=c()))
		} else {
			# order the matrix by reads mapping coord
				QuasRmat$aid <- factor(QuasRmat$aid, unique(readOrder$readname))
				QuasRmat <- QuasRmat[order(QuasRmat$aid, QuasRmat$Cid),]
				rownames(QuasRmat) <- seq(nrow(QuasRmat))
			# strand collapsing (prioritize CG)
				if(strandCollapsing & !is.null(AllC)){
					minusStrand <- !QuasRmat$Cid %in% start(AllC)
					GCi <- start(AllC[AllC$type  %in% c('GCH', 'GCG')])
					CGi <- start(AllC[AllC$type %in% c('HCG', 'GCG')])

					SelOvGC_minus <- ((QuasRmat$Cid+1) %in% GCi) & minusStrand
					SelOvCG_minus <- ((QuasRmat$Cid-1) %in% CGi) & minusStrand

					QuasRmat$Cid[SelOvCG_minus] <- QuasRmat$Cid[SelOvCG_minus] - 1
					QuasRmat$Cid[!SelOvCG_minus & SelOvGC_minus] <- QuasRmat$Cid[!SelOvCG_minus & SelOvGC_minus] + 1
				} else if(strandCollapsing & is.null(AllC)){
					stop('No reference cytosine object provided for the strand collapsing!')
				}

			# GCG are duplicated in case of separated context call
				QuasRmat <- .correctMatForGCGdup(mat=QuasRmat, mode=modeGCG)

			# reduce for only the windows of observation
				toKeep=QuasRmat$Cid >=  start(regExp) - flankedRegion & QuasRmat$Cid <=  end(regExp) + flankedRegion
				QuasRmat <- QuasRmat[toKeep,]

			return(list(
				aid=as.character(QuasRmat$aid),
				Cid=QuasRmat$Cid,
				strand=QuasRmat$strand,
				meth=QuasRmat$meth
			))
		}
	}

	#' Function that converts single-molecule methylation string (contained in mBED output files) to single-molecule cytosine methylation data frame
	#'
	#' @param mBed data frame containing methylBed reads out of Nanopore methylation call
	#'
	#' @return data frame with 4 columns:
	#' - ReadID (aid)
	#' - Cytosine coordinate (Cid)
	#' - strand (strand)
	#' - meth: translated single molecule methylation string into binary methylation (0=unmethylated, 1=methylated, NA=undetermined)
	#'
	.mBedToQuasRmat <- function(mBed){
		mat.l <- lapply(1:nrow(mBed), function(r){
			read <- mBed[r,]
			mstring = read$mstring
			call = strsplit(mstring, "[0-9]+")[[1]][-1]
			pos = cumsum(as.numeric(strsplit(mstring, "[a-z]")[[1]]))
			mat <- data.frame(
				aid=rep(read$readname, length(pos)),
				Cid=as.numeric(read$start) + pos + 1,
				strand=rep('+', length(pos)),
				meth=NA)
			mat$meth[which(call == "m")] = 1
			mat$meth[which(call == "u")] = 0
			return(mat)
		})
		return(do.call(rbind, mat.l))
	}

	#' Function to fix GCG context duplicates in single-molecule cytosine methylation data frame
	#'
	#' If multiple call found for one cytosine on the same molecule (not possible with our improved Nanopolish version):
	#' 	First, if one of them has undetermined status correct with the other methylation call.
	#' 	Second, keep only the CpG or GpC methylation value (mode='cpg' or 'gpc' respectively).
	#' 	If the mode is 'strict', all ambiguous cytosine in GCG with 2 methylation call are removed
	#'
	#' @param mat single-molecule cytosine methylation data frame
	#' @param mode mode to keep ambiguous context methylation value {cpg|gpc|strict}
	#'
	#' @return GCG corrected single-molecule cytosine methylation data frame with 4 columns:
	#' - ReadID (aid)
	#' - Cytosine coordinate (Cid)
	#' - strand (strand)
	#' - meth: translated single molecule methylation string into binary methylation (0=unmethylated, 1=methylated, NA=undetermined)
	#'
	.correctMatForGCGdup <- function(mat, mode='cpg'){
		dup <- as.numeric(rownames(mat[duplicated(paste(mat$aid, mat$Cid, sep='_')),])) #should be produced by the gpc call
		if(length(dup) > 0){
			# correct NAs with the other call if possible
				summary(is.na(mat[dup-1, 'meth']) == is.na(mat[dup, 'meth']))
				mat[dup-1, 'meth'] <- ifelse(is.na(mat[dup-1, 'meth']), mat[dup, 'meth'], mat[dup-1, 'meth'])
				mat[dup, 'meth'] <- ifelse(is.na(mat[dup, 'meth']), mat[dup-1, 'meth'], mat[dup, 'meth'])
			# correct value that are differential called between the cpg and gpc
				mat[dup-1, 'meth'] <- ifelse(
					Vectorize(isTRUE)(mat[dup, 'meth'] == mat[dup-1, 'meth']),
					mat[dup, 'meth'],
					if(mode == 'cpg'){
						mat[dup-1, 'meth']
					} else if(mode == 'gpc'){
						mat[dup, 'meth']
					} else if(mode == 'strict'){
						NA
					} else {
						stop('The mode of .correctMatForGCG() is not correct!\n')
					}
				)
			return(mat[-dup,])
		} else {
			return(mat)
		}
	}

	# Compute count matrix
	ComputeCount <- function(sRs.u, patternStrings, threads){
		Rcounts = mclapply(seq_along(sRs.u), function(i) {
			counts = unlist(lapply(sRs.u[[i]], length))
			oV = rep(0, length(patternStrings))
			names(oV) = patternStrings
			CountedPatterns <- string.split.last(names(counts), '_')
			oV[CountedPatterns] <- counts
			return(oV)
		}, mc.cores=threads)
		names(Rcounts) = names(sRs.u)
		return(do.call(rbind, Rcounts))
	}

	# Compute count matrix with cut-off
	getFrequencyMat <- function(data, cutoff){
		totC = 1 + apply(data, 1, function(x){sum(x, na.rm=T)})
		X <- apply(data, 2, function(x){
			fid = totC >= cutoff
			y = round(((x/totC) * 100), digits=3)
			y[!fid] <- NA
			return(y)
		})
		return(X)
	}

	# bin to classification assignment
	makeBinToClass <- function(patternStrings, mode){
		if(tolower(mode) %in% c('tf', 'single', 'hybrid')){
			states = list(
				nucleosome=patternStrings[c(1,seq_along(patternStrings)[!seq_along(patternStrings)%in%c(1,6,8)])],
				unbound=patternStrings[8],
				bound=patternStrings[6])
		} else if(tolower(mode) == 'std'){
			states=list(
				unassigned=patternStrings[c(4,6,7,10,13,14)],
				nucleosome=patternStrings[c(1,2,3,5,9)],
				unbound=patternStrings[c(8,16)],
				PIC = patternStrings[12],
				PIC.polII = patternStrings[11],
				polII=patternStrings[15]
			)
		} else if(tolower(mode) == 'downstream'){
			states=list(
			  unassigned=patternStrings[c(6,7,11)],
			  nucleosome=patternStrings[c(1:5,9,13)],
			  unbound=patternStrings[c(8,15,16)],
			  PIC = patternStrings[12],
			  PIC.polII = patternStrings[10],
			  polII=patternStrings[14]
			)
		} else if(grepl('bp', tolower(mode))){
			states=list(
				closed=patternStrings[1],
				accessible=patternStrings[2]
			)
		} else {
			warning('unrecognised mode to declare states...')
			states=list()
		}
		return(states)
	}

################## Co-occurence analysis  ##################

makeFeaturePairs <- function(TFBS, win, exclude, Pairmode, Freq, siteSub, sortedFeature, threads){
	dep.l <- c('GenomicRanges', 'magrittr')
	for (i in dep.l) {if(!suppressMessages(require(i, character.only = TRUE))) stop(paste0("Package not found: ", i))}

    ft = paste(names(TFBS), collapse='_')
	TFBS <- sort(TFBS, ignore.strand=TRUE)
    if(Pairmode != 'all'){
        thr <- as.numeric(gsub('freq', '', Pairmode))
    }

    sites <- mclapply(names(TFBS), function(view) {
        if(Pairmode == 'all'){
           out <- TFBS[[view]]
        } else {
            mat <- Freq[[view]]
            mat <- mat[rownames(mat) %in% names(TFBS[[view]]),]
            # collapse Freq if required
            if(length(sortedFeature[[view]]) > 1){
				mat <- rowSums(mat[,sortedFeature[[view]]], na.rm=F)
			} else {
				mat <- mat[,sortedFeature[[view]]]
			}
            ToKeep <- names(mat[ifelse(is.na(mat > thr), FALSE, mat > thr)])
            out <- TFBS[[view]][ToKeep,]
            out$freq <- mat[names(out)]
        }
        out$nature <- view
        if(!is.null(siteSub[[view]])){out <- out[out$name %in% siteSub[[view]]]}
        return(out)
    }, mc.cores = threads) %>%
        GRangesList() %>%
        unlist() %>%
        sort()

	if (length(sites) == 0) {
		message('No sites found after filtering the feature object.')
		return(NA)
	}

    if(length(TFBS) > 1){
        data <- sites[sites$nature =='TSS']
        strand(data) <- '*'
        subject = sites[sites$nature != 'TSS']
        linked <- mclapply(seq_along(data), function(i){
            query = trim(resize(data[i], win*2*1000, fix='center'))
            linkSites <- subject[subject %over% query]
            if(length(linkSites) == 0){
                return(NA)
            } else {
                toRM <- queryHits(findOverlaps(linkSites, resize(data[i], exclude[['TSS']]*2*1000, fix='center')))
                if(length(toRM) != 0){
                    linkSites <- linkSites[-toRM]
                }
                return(paste(names(linkSites), collapse="|"))
            }
        }, mc.cores = threads)
        rm(data,subject)

        sites$linkSite.pairID <- NA
        sites[sites$nature =='TSS']$linkSite.pairID <- unlist(linked)
        TFBSpair <- sites[sites$nature =='TSS' & !is.na(sites$linkSite.pairID)]
    } else {
        sites$linkSite.pairID <- mclapply(1:length(sites), function(i){
            data <- sites
            strand(data) <- '*'
            query = trim(resize(data[i], win*1000, fix='start')) #take only in 3' to avoid doublet
            subject = data[-i]

            linkSites <- subject[subject %over% query]
            if(length(linkSites) == 0){
                return(NA)
            } else {
                toRM <- queryHits(findOverlaps(linkSites, resize(data[i], exclude*1000, fix='start')))
                if(length(toRM) != 0){
                    linkSites <- linkSites[-toRM]
                }
                return(paste(names(linkSites), collapse="|"))
            }
        }, mc.cores=threads) %>%
        unlist()
        TFBSpair <- sites[!is.na(sites$linkSite.pairID),]
    }
	# sort the pairs by rank if available
	if('rank' %in% colnames(mcols(TFBSpair))){TFBSpair <- TFBSpair[order(TFBSpair$rank),]}
    # split the pairs by linkSite.pairID
	TFBSpair <- mclapply(1:length(TFBSpair), function(i){
		links = unlist(strsplit(TFBSpair[i]$linkSite.pairID, "[|]"))
		tmpGR <- TFBSpair[rep(i, length(links))]
		tmpGR$linkSite.pairID <- links
		tmpGR$linkSite.coor <- unlist(lapply(links, function(k){start(sites[k])}))
		tmpGR$linkSite.name <- unlist(lapply(links, function(k){sites[k]$name}))
		tmpGR$linkSite.strand <- unlist(lapply(links, function(k){as.character(strand(sites[k]))}))
		if('rank' %in% colnames(mcols(TFBSpair))){
			tmpGR$linkSite.rank <- unlist(lapply(links, function(k){sites[k]$rank}))
		}
        if('freq' %in% colnames(mcols(TFBSpair))){
			tmpGR$linkSite.freq <- unlist(lapply(links, function(k){sites[k]$freq}))
		}
		tmpGR$linkSite.pairDist <- unlist(lapply(links, function(k){start(sites[k])-start(tmpGR[1])}))
		return(tmpGR)
	}, mc.cores=threads) %>%
        GRangesList() %>%
        unlist() %>%
        sort(., ignore.strand=TRUE)
	TFBSpair$pairID <- paste0(ft, '_pairs_', 1:length(TFBSpair))
	return(list(sites=sites, TFBSpair=TFBSpair))
}

# Biological state declaration per feature nature (based on the ordered methylation bit-pattern of the sorting bins)
TFstate <- function(patternStrings, anc){
	return(
		list(
		nucleosome = patternStrings[anc %in% c("000","001","010","011","100","110")],
		unbound = patternStrings[anc %in% "111"],
		bound = patternStrings[anc %in% "101"])
	)
}

# mode is the TSS calling mode: 'std' (droso-like up-tata-inr-pol) or 'downstream' (mouse-like up-tata-pol-down)
TSSstate <- function(patternStrings, anc, mode){
	if(tolower(mode) %in% c('std', 'droso')){
		state =list(
			unassigned=patternStrings[anc %in% c("0011","0101","0110","1001","1100","1101")],
			nucleosome = patternStrings[anc %in% c("0000","0001","0010","0100","1000")],
			unbound = patternStrings[anc %in% c("0111","1111")],
			PIC = patternStrings[anc %in% "1011"],
			PIC.polII = patternStrings[anc %in% "1010"],
			polII=patternStrings[anc %in% "1110"])
	} else if(tolower(mode) %in% c('downstream', 'mouse')){
		state =list(
			unassigned=patternStrings[anc %in% c("0101","0110","1010")],
			nucleosome = patternStrings[anc %in% c("0000","0001","0010","0011","0100","1000","1100")],
			unbound = patternStrings[anc %in% c( "0111","1110","1111")],
			PIC = patternStrings[anc %in% "1011"],
			PIC.polII = patternStrings[anc %in% "1001"],
			polII=patternStrings[anc %in% "1101"])
	} else {
		state=list()
	}
	return(state)
}

DHSstate <- function(patternStrings, anc){
	return(
		list(
			closed = patternStrings[anc %in% "0"],
			accessible = patternStrings[anc %in% "1"])
	)
}

# Build the combined (TF1 x TF2) state list for a pair of feature natures (e.g. c('TF','TF'), c('TSS','TF'), c('TSS','DHS'))
MakePairStates <- function(pairType, mode){
	refBins=c(TF=3, TSS=4, DHS=1)
	Bins=c(refBins[pairType[1]], refBins[pairType[2]])

	allPos = expand.grid(rep(list(c(0,1)), sum(Bins)))
	patternStrings=names(table(apply(allPos,1,function(x){(paste(as.character((x)),collapse=''))})))

	TFs_states <- lapply(seq_along(Bins), function(n){
		if (n == 1){
			anc = substr(patternStrings, 1, Bins[1])
		} else {
			anc = substr(patternStrings, Bins[1]+1, sum(Bins))
		}

		if(names(Bins[n]) == 'TF'){
			return(TFstate(patternStrings, anc))
		} else if(names(Bins[n]) == 'TSS'){
			return(TSSstate(patternStrings, anc, mode[['TSS']]))
		} else  if(names(Bins[n]) == 'DHS'){
			return(DHSstate(patternStrings, anc))
		}
	})
	names(TFs_states) <- paste0('TF', seq_along(Bins))

	TF_TF_states = unlist(lapply(seq_along(TFs_states[['TF1']]), function(i){
		X <- lapply(seq_along(TFs_states[['TF2']]), function(j){
			TFs_states[['TF1']][[i]][TFs_states[['TF1']][[i]] %in% TFs_states[['TF2']][[j]]]
			})
		names(X)=paste0(pairType[1],"1_",names(TFs_states[['TF1']])[i],"_", pairType[2],"2_",names(TFs_states[['TF2']]))
		return(X)
	}), recursive=F)

	return(list(TF1=TFs_states[['TF1']], TF2=TFs_states[['TF2']], TF_TF=TF_TF_states))
}

# Preferred display/state ordering of the combined states per feature nature combo
getReadOrder <- function(featureNature){
	out = c(NA)
	if(featureNature == 'tf'){
		out=c("TF1_nucleosome_TF2_nucleosome","TF1_nucleosome_TF2_unbound","TF1_unbound_TF2_nucleosome",
			"TF1_unbound_TF2_unbound", "TF1_nucleosome_TF2_bound", "TF1_unbound_TF2_bound",
			"TF1_bound_TF2_nucleosome","TF1_bound_TF2_unbound", "TF1_bound_TF2_bound")
	} else if(featureNature == 'tss'){
		out=c("TSS1_unassigned_TSS2_unassigned", "TSS1_unassigned_TSS2_nucleosome", "TSS1_nucleosome_TSS2_unassigned",
		      "TSS1_unassigned_TSS2_unbound",  "TSS1_nucleosome_TSS2_unbound",
		      "TSS1_nucleosome_TSS2_nucleosome", "TSS1_unbound_TSS2_unassigned", "TSS1_unbound_TSS2_nucleosome",
		      "TSS1_unassigned_TSS2_PIC", "TSS1_nucleosome_TSS2_PIC", "TSS1_unbound_TSS2_PIC",
		      "TSS1_PIC_TSS2_unassigned", "TSS1_PIC_TSS2_nucleosome", "TSS1_PIC_TSS2_unbound", "TSS1_PIC_TSS2_PIC",
		      "TSS1_unbound_TSS2_unbound",
		      "TSS1_unassigned_TSS2_polII", "TSS1_unassigned_TSS2_PIC.polII", "TSS1_nucleosome_TSS2_PIC.polII", "TSS1_nucleosome_TSS2_polII",
		      "TSS1_unbound_TSS2_polII", "TSS1_unbound_TSS2_PIC.polII", "TSS1_PIC_TSS2_PIC.polII", "TSS1_PIC_TSS2_polII",
		      "TSS1_polII_TSS2_unassigned", "TSS1_PIC.polII_TSS2_unassigned", "TSS1_polII_TSS2_nucleosome", "TSS1_PIC.polII_TSS2_nucleosome",
		      "TSS1_polII_TSS2_unbound", "TSS1_PIC.polII_TSS2_unbound", "TSS1_PIC.polII_TSS2_PIC","TSS1_polII_TSS2_PIC",
		      "TSS1_PIC.polII_TSS2_PIC.polII","TSS1_PIC.polII_TSS2_polII", "TSS1_polII_TSS2_PIC.polII", "TSS1_polII_TSS2_polII")
	} else if(featureNature == 'dhs'){
		out=c('DHS1_closed_DHS2_closed', 'DHS1_closed_DHS2_accessible', 'DHS1_accessible_DHS2_closed', 'DHS1_accessible_DHS2_accessible')
	} else if(featureNature %in% c('tf-tss', 'tss-tf')){
		out=c("TSS1_unassigned_TF2_nucleosome", "TSS1_unassigned_TF2_unbound", "TSS1_unassigned_TF2_bound",
			"TSS1_nucleosome_TF2_nucleosome", "TSS1_nucleosome_TF2_unbound", "TSS1_nucleosome_TF2_bound",
			"TSS1_unbound_TF2_nucleosome",  "TSS1_unbound_TF2_unbound", "TSS1_unbound_TF2_bound",
			"TSS1_PIC_TF2_nucleosome", "TSS1_PIC_TF2_unbound", "TSS1_PIC_TF2_bound",
			"TSS1_PIC.polII_TF2_nucleosome", "TSS1_PIC.polII_TF2_unbound", "TSS1_PIC.polII_TF2_bound",
			"TSS1_polII_TF2_nucleosome", "TSS1_polII_TF2_unbound", "TSS1_polII_TF2_bound")
	} else if(featureNature %in% c('tss-dhs', 'dhs-tss')){
		out=c("TSS1_unassigned_DHS2_closed", "TSS1_unassigned_DHS2_accessible", "TSS1_nucleosome_DHS2_closed",
			"TSS1_nucleosome_DHS2_accessible", "TSS1_unbound_DHS2_closed", "TSS1_unbound_DHS2_accessible",
			"TSS1_PIC_DHS2_closed", "TSS1_PIC_DHS2_accessible", "TSS1_PIC.polII_DHS2_closed",
			"TSS1_PIC.polII_DHS2_accessible", "TSS1_polII_DHS2_closed", "TSS1_polII_DHS2_accessible")
	}
	return(out)
}

# Collapse the fine-grained combined states into the coarse biological categories used for the Fisher tests
ReducecountStates <- function(count, type, pseudocount=0){
	out = c()
	if(type=='TF'){
		out = c(
			bTF_bTF=sum(count[c('TF1_bound_TF2_bound')]),
			bTF_uTF=sum(count[c('TF1_bound_TF2_unbound', 'TF1_bound_TF2_nucleosome')]),
			uTF_bTF=sum(count[c('TF1_unbound_TF2_bound', 'TF1_nucleosome_TF2_bound')]),
			uTF_uTF=sum(count[c('TF1_unbound_TF2_unbound')]),
			nuc_TF=sum(count[c('TF1_nucleosome_TF2_nucleosome', 'TF1_nucleosome_TF2_unbound', 'TF1_unbound_TF2_nucleosome')]),

			Ac_Ac=sum(count[c('TF1_bound_TF2_bound', 'TF1_bound_TF2_unbound', 'TF1_unbound_TF2_bound', 'TF1_unbound_TF2_unbound')]),
			Ac_noAc=sum(count[c('TF1_bound_TF2_nucleosome', 'TF1_unbound_TF2_nucleosome')]),
			noAc_Ac=sum(count[c('TF1_nucleosome_TF2_bound', 'TF1_nucleosome_TF2_unbound')]),
			noAc_noAc=sum(count[c('TF1_nucleosome_TF2_nucleosome')])
			)
		out = out + pseudocount
	} else if(type=='TSS'){
			out = c(
				bPolII_bPolII=sum(count[c("TSS1_polII_TSS2_polII", "TSS1_polII_TSS2_PIC.polII", "TSS1_PIC.polII_TSS2_polII", "TSS1_PIC.polII_TSS2_PIC.polII")]),
				bPolII_uPolII=sum(count[c("TSS1_polII_TSS2_unbound", "TSS1_polII_TSS2_PIC", "TSS1_polII_TSS2_nucleosome", "TSS1_PIC.polII_TSS2_unbound", "TSS1_PIC.polII_TSS2_PIC", "TSS1_PIC.polII_TSS2_nucleosome")]),
				uPolII_bPolII=sum(count[c("TSS1_unbound_TSS2_polII", "TSS1_PIC_TSS2_polII", "TSS1_nucleosome_TSS2_polII", "TSS1_unbound_TSS2_PIC.polII", "TSS1_PIC_TSS2_PIC.polII", "TSS1_nucleosome_TSS2_PIC.polII")]),
				uPolII_uPolII=sum(count[c("TSS1_unbound_TSS2_unbound", "TSS1_PIC_TSS2_unbound", "TSS1_unbound_TSS2_PIC", "TSS1_PIC_TSS2_PIC")]),
				nuc_PolII=sum(count[c("TSS1_nucleosome_TSS2_nucleosome", "TSS1_nucleosome_TSS2_unbound", "TSS1_unbound_TSS2_nucleosome", "TSS1_nucleosome_TSS2_PIC", "TSS1_PIC_TSS2_nucleosome")]),

				bPIC_bPIC=sum(count[c("TSS1_PIC_TSS2_PIC", "TSS1_PIC_TSS2_PIC.polII", "TSS1_PIC.polII_TSS2_PIC", "TSS1_PIC.polII_TSS2_PIC.polII")]),
				bPIC_uPIC=sum(count[c("TSS1_PIC_TSS2_unbound", "TSS1_PIC_TSS2_polII", "TSS1_PIC_TSS2_nucleosome",  "TSS1_PIC.polII_TSS2_nucleosome", "TSS1_PIC.polII_TSS2_unbound", "TSS1_PIC.polII_TSS2_polII")]),
				uPIC_bPIC=sum(count[c("TSS1_unbound_TSS2_PIC", "TSS1_polII_TSS2_PIC", "TSS1_nucleosome_TSS2_PIC", "TSS1_nucleosome_TSS2_PIC.polII", "TSS1_unbound_TSS2_PIC.polII", "TSS1_polII_TSS2_PIC.polII")]),
				uPIC_uPIC=sum(count[c("TSS1_unbound_TSS2_unbound", "TSS1_polII_TSS2_unbound", "TSS1_unbound_TSS2_polII", "TSS1_polII_TSS2_polII")]),
				nuc_PIC=sum(count[c("TSS1_nucleosome_TSS2_nucleosome", "TSS1_nucleosome_TSS2_unbound", "TSS1_unbound_TSS2_nucleosome", "TSS1_nucleosome_TSS2_polII", "TSS1_polII_TSS2_nucleosome")]),

				Ac_Ac=sum(count[c("TSS1_polII_TSS2_polII", "TSS1_polII_TSS2_PIC.polII", "TSS1_PIC.polII_TSS2_polII", "TSS1_PIC.polII_TSS2_PIC.polII", "TSS1_unbound_TSS2_unbound", "TSS1_PIC_TSS2_unbound", "TSS1_unbound_TSS2_PIC",  "TSS1_PIC_TSS2_PIC", "TSS1_PIC_TSS2_PIC.polII", "TSS1_polII_TSS2_unbound", "TSS1_polII_TSS2_PIC","TSS1_PIC.polII_TSS2_unbound", "TSS1_PIC.polII_TSS2_PIC", "TSS1_unbound_TSS2_polII", "TSS1_PIC_TSS2_polII", "TSS1_unbound_TSS2_PIC.polII")]),
				Ac_noAc=sum(count[c("TSS1_polII_TSS2_nucleosome", "TSS1_unbound_TSS2_nucleosome", "TSS1_PIC_TSS2_nucleosome", "TSS1_PIC.polII_TSS2_nucleosome")]),
				noAc_Ac=sum(count[c("TSS1_nucleosome_TSS2_polII", "TSS1_nucleosome_TSS2_unbound", "TSS1_nucleosome_TSS2_PIC", "TSS1_nucleosome_TSS2_PIC.polII")]),
				noAc_noAc=sum(count[c("TSS1_nucleosome_TSS2_nucleosome")])
				)
		out = out + pseudocount
	} else if(type %in% c('TSS-TF', 'TF-TSS')){
		out = c(
			bPolII_bTF=sum(count[c("TSS1_polII_TF2_bound", "TSS1_PIC.polII_TF2_bound")]),
			bPolII_uTF=sum(count[c("TSS1_polII_TF2_unbound", "TSS1_polII_TF2_nucleosome", "TSS1_PIC.polII_TF2_unbound", "TSS1_PIC.polII_TF2_nucleosome")]),
			uPolII_bTF=sum(count[c("TSS1_unbound_TF2_bound", "TSS1_PIC_TF2_bound", "TSS1_nucleosome_TF2_bound")]),
			uPolII_uTF=sum(count[c("TSS1_unbound_TF2_unbound", "TSS1_PIC_TF2_unbound")]),
			`nuc_PolII-TF`=sum(count[c("TSS1_nucleosome_TF2_nucleosome", "TSS1_nucleosome_TF2_unbound", "TSS1_unbound_TF2_nucleosome",  "TSS1_PIC_TF2_nucleosome")]),

			bPIC_bTF=sum(count[c("TSS1_PIC_TF2_bound", "TSS1_PIC.polII_TF2_bound")]),
			bPIC_uTF=sum(count[c("TSS1_PIC_TF2_unbound", "TSS1_PIC_TF2_nucleosome", "TSS1_PIC.polII_TF2_unbound", "TSS1_PIC.polII_TF2_nucleosome")]),
			uPIC_bTF=sum(count[c("TSS1_polII_TF2_bound", "TSS1_unbound_TF2_bound","TSS1_nucleosome_TF2_bound")]),
			uPIC_uTF=sum(count[c("TSS1_unbound_TF2_unbound", "TSS1_polII_TF2_unbound")]),
			`nuc_PIC-TF`=sum(count[c("TSS1_nucleosome_TF2_nucleosome", "TSS1_nucleosome_TF2_unbound", "TSS1_unbound_TF2_nucleosome", "TSS1_polII_TF2_nucleosome")]),

			Ac_Ac=sum(count[c("TSS1_PIC_TF2_bound", "TSS1_PIC.polII_TF2_bound", "TSS1_PIC_TF2_unbound", "TSS1_PIC.polII_TF2_unbound",
				"TSS1_polII_TF2_bound", "TSS1_unbound_TF2_bound", "TSS1_unbound_TF2_unbound", "TSS1_polII_TF2_unbound")]),
			Ac_noAc=sum(count[c("TSS1_PIC_TF2_nucleosome", "TSS1_PIC.polII_TF2_nucleosome", "TSS1_unbound_TF2_nucleosome", "TSS1_polII_TF2_nucleosome")]),
			noAc_Ac=sum(count[c("TSS1_nucleosome_TF2_bound", "TSS1_nucleosome_TF2_unbound")]),
			noAc_noAc=sum(count[c("TSS1_nucleosome_TF2_nucleosome")])
		)
		out = out + pseudocount
	} else if(type=='DHS'){
		out = c(
			Ac_Ac=sum(count['DHS1_accessible_DHS2_accessible']),
			Ac_noAc=sum(count['DHS1_accessible_DHS2_closed']),
			noAc_Ac=sum(count['DHS1_closed_DHS2_accessible']),
			noAc_noAc=sum(count['DHS1_closed_DHS2_closed'])
			)
		out = out + pseudocount
	} else if(type %in% c('TSS-DHS', 'DHS-TSS')){
		out = c(
			bPolII_bDHS=sum(count[c("TSS1_polII_DHS2_accessible", "TSS1_PIC.polII_DHS2_accessible")]),
			bPolII_uDHS=sum(count[c("TSS1_polII_DHS2_closed", "TSS1_PIC.polII_DHS2_closed")]),
			uPolII_bDHS=sum(count[c("TSS1_unbound_DHS2_accessible", "TSS1_PIC_DHS2_accessible", "TSS1_nucleosome_DHS2_accessible")]),
			uPolII_uDHS=sum(count[c("TSS1_unbound_DHS2_closed", "TSS1_PIC_DHS2_closed")]),
			`nuc_PolII-DHS`=sum(count[c("TSS1_nucleosome_DHS2_closed")]),

			bPIC_bDHS=sum(count[c("TSS1_PIC_DHS2_accessible", "TSS1_PIC.polII_DHS2_accessible")]),
			bPIC_uDHS=sum(count[c("TSS1_PIC_DHS2_closed", "TSS1_PIC.polII_DHS2_closed")]),
			uPIC_bDHS=sum(count[c("TSS1_polII_DHS2_accessible", "TSS1_unbound_DHS2_accessible","TSS1_nucleosome_DHS2_accessible")]),
			uPIC_uDHS=sum(count[c("TSS1_unbound_DHS2_closed", "TSS1_polII_DHS2_closed")]),
			`nuc_PIC-DHS`=sum(count[c("TSS1_nucleosome_DHS2_closed")]),

			Ac_Ac=sum(count[c("TSS1_PIC_DHS2_accessible", "TSS1_PIC.polII_DHS2_accessible", "TSS1_polII_DHS2_accessible", "TSS1_unbound_DHS2_accessible")]),
			Ac_noAc=sum(count[c("TSS1_PIC_DHS2_closed", "TSS1_PIC.polII_DHS2_closed", "TSS1_unbound_DHS2_closed", "TSS1_polII_DHS2_closed")]),
			noAc_Ac=sum(count[c("TSS1_nucleosome_DHS2_accessible")]),
			noAc_noAc=sum(count[c("TSS1_nucleosome_DHS2_closed")])
		)
		out = out + pseudocount
	}
	return(out)
}

# Fisher's exact test for co-occupancy (coop) and co-accessibility (coacc) on the reduced 2x2 contingency tables
MakeFisher <- function(reduceStates, type){
	if(type == 'TF'){
		ft=list(rep(type,2))
		name.l = c('TF_coop', 'TF_coacc')
	} else if(type == 'TSS'){
		ft=list(rep('PolII',2), rep('PIC',2))
		name.l = c('PolII_coop', 'PIC_coop', 'TSS_coacc')
	} else if(type %in% c('TSS-TF', 'TF-TSS')){
		ft=list(c('PolII','TF'), c('PIC','TF'))
		name.l = c("PolII-TF_coop","PIC-TF_coop", 'TSS-TF_coacc')
	} else if(type == 'DHS'){
		ft=list()
		name.l = c('DHS-DHS_coacc')
	} else if(type %in% c('TSS-DHS', 'DHS-TSS')){
		ft=list(c('PolII','DHS'), c('PIC','DHS'))
		name.l = c("PolII-DHS_coop","PIC-DHS_coop", 'TSS-DHS_coacc')
	} else {
		return(list())
	}
	out <- lapply(ft, function(n){
		contT <- matrix(c(reduceStates[paste0('b', n[1],'_b',n[2])],
			reduceStates[paste0('u', n[1],'_b',n[2])],
			reduceStates[paste0('b', n[1],'_u',n[2])],
			reduceStates[paste0('u', n[1],'_u',n[2])] +
			reduceStates[paste0('nuc_', ifelse(n[1] == n[2], n[1], paste(n[1],n[2], sep='-')))]
			), 2, 2, byrow = F, dimnames=list(
				paste0(n[1], c('_1 bound', '_1 unbound')),
				paste0(n[2], c('_2 bound', '_2 unbound'))
		))
		Ftest <- fisher.test(contT)
		score.coop <- round(contT[1,1]/(contT[1,1]+contT[2,1]+contT[1,2]), 5)

		return(list(
			cont=contT, score=score.coop, odd=round(Ftest$estimate, 4),
			pvalue=Ftest$p.value, p.adjust=NA, conf.int=round(Ftest$conf.int, 4),
			cov=sum(contT), cov.adjust=sum(contT[1,1]+contT[2,1]+contT[1,2]))
		)
	})

	contT <- matrix(c(reduceStates['Ac_Ac'],
		reduceStates['noAc_Ac'],
		reduceStates['Ac_noAc'],
		reduceStates['noAc_noAc']
		), 2, 2, byrow = F, dimnames=list(
		paste0(type, c('_1 Acc', '_1 noAcc')),
		paste0(type, c('_2 Acc', '_2 noAcc'))
	))
	Ftest <- fisher.test(contT)
	score <- round(contT[1,1]/(contT[1,1]+contT[2,1]+contT[1,2]), 5)
	out <- append(out, list(list(
			cont=contT, score=score, odd=round(Ftest$estimate, 4),
			pvalue=Ftest$p.value, p.adjust=NA, conf.int=round(Ftest$conf.int, 4),
			cov=sum(contT), cov.adjust=sum(contT[1,1]+contT[2,1]+contT[1,2])))
	)
	names(out) <- name.l
	return(out)
}

#' Compute the per-pair contingency tables (Fisher tests) for a set of feature pairs
#'
#' @param TFBSpair GRanges of feature pairs, output of makeFeaturePairs()$TFBSpair
#' @param smA Single-molecule vector list (qread output of smf_sort.R) for the reference feature (site name keyed)
#' @param smB Single-molecule vector list for the linked feature (same object as smA for homotypic pairs)
#' @param TF_TF_states Combined state list, output of MakePairStates()$TF_TF
#' @param pairType String describing the pair nature for ReducecountStates/MakeFisher ('TF', 'TSS', 'DHS', 'TSS-TF', 'TSS-DHS'...)
#' @param minCov Minimal number of single molecules covering both features of a pair to test it
#' @param threads Number of threads
#'
#' @return Named list (by 'siteA:siteB') of Fisher test results per pair, with coor/Pair_name/PairID metadata attached
computePairContingency <- function(TFBSpair, smA, smB, TF_TF_states, pairType, minCov=5, threads=2){
	dep.l <- c('parallel')
	for (i in dep.l) {if(!suppressMessages(require(i, character.only = TRUE))) stop(paste0("Package not found: ", i))}

	conT.L <- mclapply(seq_along(TFBSpair), function(k){
		pr <- TFBSpair[k]
		siteA <- names(pr)
		siteB <- pr$linkSite.pairID

		read_sortA <- smA[[siteA]]
		read_sortB <- smB[[siteB]]
		if(is.null(read_sortA) || is.null(read_sortB)){return(NULL)}

		nSM1 <- unlist(read_sortA, use.names=FALSE)
		nSM2 <- unlist(read_sortB, use.names=FALSE)
		covReadID <- nSM1[nSM1 %in% nSM2]
		nSM <- length(covReadID)
		if(nSM < minCov){return(NULL)}

		Pair <- paste(siteA, siteB, sep=':')
		Pair_name <- paste(pr$name, pr$linkSite.name, sep=':')
		if(length(Pair_name) == 0){Pair_name <- Pair}
		coor <- paste0(seqnames(pr), ':', start(pr), '-', pr$linkSite.coor)

		# map each covered read to its sorting bin/pattern for feature A and feature B
		binsA <- read_sortA
		names(binsA) <- string.split.last(names(binsA), '_')
		dfA <- do.call(rbind, lapply(seq_along(binsA), function(i){
			reads <- binsA[[i]][binsA[[i]] %in% covReadID]
			if(length(reads) == 0){return(NULL)}
			data.frame(readID=reads, sortingBins=names(binsA)[i], stringsAsFactors=FALSE)
		}))
		if(is.null(dfA) || nrow(dfA) == 0){return(NULL)}
		rownames(dfA) <- dfA$readID

		binsB <- read_sortB
		names(binsB) <- string.split.last(names(binsB), '_')
		dfB <- do.call(rbind, lapply(seq_along(binsB), function(i){
			reads <- binsB[[i]][binsB[[i]] %in% covReadID]
			if(length(reads) == 0){return(NULL)}
			data.frame(readID=reads, sortingBins=names(binsB)[i], stringsAsFactors=FALSE)
		}))
		if(is.null(dfB) || nrow(dfB) == 0){return(NULL)}
		rownames(dfB) <- dfB$readID

		covReadID <- covReadID[covReadID %in% rownames(dfA) & covReadID %in% rownames(dfB)]
		if(length(covReadID) < minCov){return(NULL)}

		sortingBins <- paste0(dfA[covReadID, 'sortingBins'], dfB[covReadID, 'sortingBins'])
		names(covReadID) <- sortingBins
		read_sort <- split(covReadID, names(covReadID))
		read_sort <- lapply(read_sort, as.vector)

		read_sort.e <- vector('list', length(unlist(TF_TF_states)))
		names(read_sort.e) <- unlist(TF_TF_states)
		read_sort.e[names(read_sort)] <- read_sort
		read_sort <- read_sort.e

		counts <- unlist(lapply(read_sort, length))
		grouped.counts <- unlist(lapply(seq_along(TF_TF_states), function(i){sum(counts[TF_TF_states[[i]]])}))
		grouped.counts[is.na(grouped.counts)] <- 0
		names(grouped.counts) <- names(TF_TF_states)

		if(sum(grouped.counts[grep('unassigned', names(grouped.counts), invert=TRUE)]) < minCov){return(NULL)}

		reduceStates <- ReducecountStates(grouped.counts, type=pairType, pseudocount=1)
		out <- MakeFisher(reduceStates=reduceStates, type=pairType)

		out <- list(append(out, list(coor=coor, Pair_name=Pair_name, PairID=pr$pairID)))
		names(out) <- Pair
		return(out)
	}, mc.cores=threads)

	conT.L <- conT.L[!sapply(conT.L, is.null)]
	conT.L <- unlist(conT.L, recursive=FALSE)
	return(conT.L)
}

# Adjust p-values (BH) across all pairs, independently for each contingency comparison type (e.g. TF_coop, TF_coacc)
adjustContingencyPvalues <- function(conT.L, threads=2){
	if(length(conT.L) == 0){return(conT.L)}
	comNames <- c('coor', 'Pair_name', 'PairID')
	sortFT <- names(conT.L[[1]])
	sortFT <- sortFT[!sortFT %in% comNames]
	pairNames <- names(conT.L)

	for(type in sortFT){
		pval <- unlist(lapply(conT.L, function(i){i[[type]]$pvalue}), use.names=FALSE)
		adj.pval <- p.adjust(pval, method='BH')
		conT.L <- mclapply(seq_along(conT.L), function(i){
			x <- conT.L[[i]]
			x[[type]][['p.adjust']] <- adj.pval[i]
			return(x)
		}, mc.cores=threads)
		names(conT.L) <- pairNames
	}
	return(conT.L)
}

##################### Phiscore Utils  ######################

	rm_outliers <- function(matrix, lowThr, HighThr) {
		cytMeans <- colMeans(matrix, na.rm=T)
		toRm <- which(cytMeans <= lowThr | cytMeans >= HighThr)
		out <- matrix[,-toRm]
		return(out)
	}

	getCMethMatrix <- function(SM_files, range) {
		if(!suppressMessages(require("data.table", character.only = TRUE))) stop("Package not found: data.table")
		Cs = .getQuasRqMeth(SM_files, range)

		# use data.table to get a 1,0 matrix of methylation profiles
		all.cids = unique(Cs$Cid) # get all possible C locations

		# make the data.table object
		dt = data.table(meth = Cs$meth, aid = Cs$aid, cid = Cs$Cid)
		if(length(dt) > 0){
			# this function converts cids to columns
			myfun <- function(x, all.cids) {
				vec = rep(-1, length(all.cids))
				names(vec) = as.character(all.cids)
				b = as.list((vec))
				b[as.character(x$cid)] = as.double(x$meth)
				return(b)
			}
			dtm = dt[, myfun(.SD, all.cids), by = aid]
			ronames = dtm$aid
			dtm[, aid:=NULL] # remove unwanted row
			CpGm = as.matrix(dtm)
			CpGm[CpGm == -1] = NA # put NAs
			rownames(CpGm) = ronames
			return(CpGm)
		} else {
			return(matrix(ncol = 0, nrow = 0))
		}
	}
