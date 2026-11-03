// Download the BSgenome
process BSGENOME {
    label 'SMF_bulk'

    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.SMF}"
    module { params.use_module ? "${params.tools.module.R}" : '' }

    input:
        val BSgenome

    output:
        path BSgenome       , emit: BSgenome
        path "versions.yml" , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    Rscript --vanilla ${workflow.projectDir}/bin/BSgenome_build.R \
    --genome ${BSgenome} \
    --taskMeta ${task.process}
    """
}

// Make the reference object for the input's motif
process REF_SITES {
    label 'SMF_bulk'
    label 'process_high'

    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.SMF}"
    module { params.use_module ? "${params.tools.module.R}" : '' }

    input:
        val motifString
        path BSgenome

    output:
        path "*.qs"         , emit: qs
        path "versions.yml" , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    Rscript --vanilla ${workflow.projectDir}/bin/smf_bulk_makeRef.R \
    --genome ${params.BSgenome} \
    --mode ${motifString} \
    --threads ${task.cpus} \
    --taskMeta ${task.process}
    """
}

// Extract Methylation in R object
process SMF_EXTRACT {
    label 'SMF_bulk'
    label 'process_medium'
    label 'process_high_memory'

    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.SMF}"
    module { params.use_module ? "${params.tools.module.R}" : '' }

    input:
        val motifString
        path BSgenome
        path refqs
        val filter_read_met
        path "*"

    output:
        path "BulkSMF_*.qs" , emit: bulkSignal
        path "qc.tsv"       , emit: qc
        path "versions.yml" , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    Rscript --vanilla ${workflow.projectDir}/bin/smf_bulk.R \
    --genome ${params.BSgenome} \
    --expName ${params.sample_name} \
    --mode ${motifString} \
    --threads ${task.cpus} \
    --taskMeta ${task.process}

    if [[ $filter_read_met ]]; then
        for qsfile in BulkSMF_*.qs; do
            mv "\${qsfile}" "\${qsfile/.qs/_methFiltering_min_${params.min_read_met_rate}_max_${params.max_read_met_rate}.qs}"
        done
    fi
    """
}

// publish stat in the QC output dir
process SMF_PUB_STAT {
    label 'SMF_bulk'

    input:
        path qc
        val filter_read_met

    output:
        path "*_qc.tsv", emit: tsv

    when:
    task.ext.when == null || task.ext.when

    script:
    argsFile = filter_read_met ? "${params.sample_name}_methFiltering_min_${params.min_read_met_rate}_max_${params.max_read_met_rate}_qc.tsv" : "${params.sample_name}_qc.tsv"
    """
    cat ${qc} > ${argsFile}
    """
}

// Make the reference object for the input's motif
process SMF_BULK_QC {
    label 'SMF_bulk'
    label 'QC'
    label 'process_high'
    label 'process_high_memory'

    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.SMF}"
    module { params.use_module ? "${params.tools.module.R}" : '' }

    input:
        val motifString
        path qs
        path qc

    output:
        path "${params.sample_name}_plots_minCov_*" , emit: folder
        path "versions.yml"                         , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    refData_arg = params.refData ? "--refData ${params.refData}" : ''
    """
    Rscript --vanilla ${workflow.projectDir}/bin/smf_bulk_qc.R \
    --InputQs ${qs} \
    --stat ${qc} \
    --expName ${params.sample_name} \
    --mode ${motifString} ${refData_arg} \
    --threads ${task.cpus} \
    --minCoverage ${params.min_cytosine_coverage} \
    --refMinCoverage ${params.ref_min_cytosine_coverage} \
    --taskMeta ${task.process}
    """
}

// Make the reference object for the input's motif
process SMF_BW {
    label 'SMF_bulk'
    label 'process_low'

    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.SMF}"
    module { params.use_module ? "${params.tools.module.R}" : '' }

    input:
        val motifString
        path BSgenome
        path qs

    output:
        path "${params.sample_name}_BW_minCov_*", emit: folder
        path "versions.yml"                     , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    Rscript --vanilla ${workflow.projectDir}/bin/smf_bulk_bw.R \
    --genome ${params.BSgenome} \
    --InputQs ${qs} \
    --expName ${params.sample_name} \
    --mode ${motifString} \
    --keepMeth ${params.keep_methylation_signal} \
    --threads ${task.cpus} \
    --minCoverage ${params.min_cytosine_coverage} \
    --taskMeta ${task.process}
    """
}

// Sort read according to binding patterns
process SM_SORT {
    tag "${featureNat}_${sortingMode}"
    label 'SMF_sort'
    label 'process_high'
    label 'process_high_memory'
    label 'error_retry'

    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.SMF}"
    module { params.use_module ? "${params.tools.module.R}" : '' }

    input:
        tuple val(featureNat), path(feature), val(sortingMode)
        val motifString
        path BSgenome
        tuple path(mbed), path(tbi)


    output:
        tuple val(featureNat), val(sortingMode), path("SMvec_*.qs"),    emit: SMvec
        path "bin_used_*.qs", optional: true,                           emit: Bins
        tuple val(featureNat), path(feature), val(sortingMode),         emit: Feature
        path "versions.yml",                                            emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    Rscript --vanilla ${workflow.projectDir}/bin/smf_sort.R \
    --genome ${params.BSgenome} \
    --featureFile ${feature} \
    --context ${motifString}\
    --expName ${params.sample_name} \
    --featureNature ${featureNat} \
    --mode ${sortingMode} \
    --threads ${task.cpus} \
    --taskMeta ${task.process}
    """
}

process TF_HYBRID_SORT {
    label 'SMF_sort'
    label 'process_high'
    label 'process_high_memory'
    label 'error_retry'

    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.SMF}"
    module { params.use_module ? "${params.tools.module.R}" : '' }

    input:
        tuple path(singleSM), path(clusterSM)
        tuple path(feature), val(featureNat)

    output:
        tuple val(featureNat), val('hybrid'), path("SMvec_*.qs"), emit: SMvec
        path('Plot_hybrid_TF_sorting_SM_merging')               , emit: folder
        path "versions.yml"                                     , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    Rscript --vanilla ${workflow.projectDir}/bin/smf_mergeTFsort.R \
    --featureFile ${feature} \
    --expName ${params.sample_name} \
    --minCoverage ${params.min_cytosine_coverage} \
    --threads ${task.cpus} \
    --taskMeta ${task.process}
    """
}

// extract binding frequency from the single molecule vector
process SM_EXTRACT {
     tag "${featureNat}_${sortingMode}"
    label 'SMF_sort_extract'
    label 'process_high'

    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.SMF}"
    module { params.use_module ? "${params.tools.module.R}" : '' }

    input:
        tuple val(featureNat2), val(sortingMode2), path(smvec)
        tuple val(featureNat), path(feature), val(sortingMode)
        val motifString

    output:
        path "pattern_*.qs",                                                                    emit: pattern
        path "states_*.qs", optional: true,                                                     emit: state
        tuple val(featureNat), val(sortingMode), path("states_frequency_*.qs"), optional: true, emit: freq
        path "versions.yml",                                                                    emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    Rscript --vanilla ${workflow.projectDir}/bin/smf_sort_extract.R \
    --inputSMvec ${smvec} \
    --featureFile ${feature} \
    --context ${motifString}\
    --expName ${params.sample_name} \
    --featureNature ${featureNat} \
    --mode ${sortingMode} \
    --minCoverage ${params.min_cytosine_coverage} \
    --threads ${task.cpus} \
    --taskMeta ${task.process}
    """
}

// Create the feature pair objects (homotypic or heterotypic) used for the co-occupancy analysis
process COOC_MAKEPAIRS {
    tag "${comboNat}"
    label 'SMF_coop'
    label 'process_high'
    label 'process_high_memory'

    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.SMF}"
    module { params.use_module ? "${params.tools.module.R}" : '' }

    input:
        tuple val(comboNat), val(comboKey), val(natures), path(feature), val(modes), path(freq), path(smvec)

    output:
        tuple val(comboNat), val(comboKey), val(modes), path("*pairs*.qs"), path(smvec), emit: pairs
        path "versions.yml",                                                            emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    Rscript --vanilla ${workflow.projectDir}/bin/smf_cooc_makePairs.R \
    --featureFile ${feature.join(',')} \
    --freqFile ${freq.join(',')} \
    --featureNature ${comboNat} \
    --mode ${modes.join(',')} \
    --pairmode ${params.coop_pairmode} \
    --rightwindow ${params.coop_rightwindow} \
    --threads ${task.cpus} \
    --taskMeta ${task.process}
    """
}

// Compute the co-occupancy / co-accessibility contingency tables (Fisher's exact test) for each feature pair
process COOC_CONTINGENCY {
    tag "${comboNat}"
    label 'SMF_coop'
    label 'process_high'
    label 'process_high_memory'
    label 'error_retry'

    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.SMF}"
    module { params.use_module ? "${params.tools.module.R}" : '' }

    input:
        tuple val(comboNat), val(comboKey), val(modes), path(pairs), path(smvec)

    output:
        tuple val(comboNat), val(comboKey), path("*_contingency_*.qs"), emit: contingency
        path "versions.yml",                                           emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def smvecA = smvec instanceof List ? smvec[0] : smvec
    def smvecB = smvec instanceof List && smvec.size() > 1 ? smvec[1] : smvecA
    """
    Rscript --vanilla ${workflow.projectDir}/bin/smf_cooc.R \
    --pairsFile ${pairs} \
    --smvecA ${smvecA} \
    --smvecB ${smvecB} \
    --featureNature ${comboNat} \
    --mode ${modes.join(',')} \
    --expName ${comboKey} \
    --minCov ${params.coop_min_cov} \
    --threads ${task.cpus} \
    --taskMeta ${task.process}
    """
}

// Fold the contingency object of a comparison into a single master table (.qs), applying feature-nature
// specific filtering (e.g. 10-90% accessibility for DHS). This is the final output of the coop analysis.
process COOC_MASTERTABLE {
    tag "${comboNat}"
    label 'SMF_coop'
    label 'process_medium'

    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.SMF}"
    module { params.use_module ? "${params.tools.module.R}" : '' }

    input:
        tuple val(comboNat), val(comboKey), path(pairs), path(contingency)

    output:
        tuple val(comboNat), val(comboKey), path("*_masterTable.qs"), emit: masterTable
        path "versions.yml",                                         emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    Rscript --vanilla ${workflow.projectDir}/bin/smf_cooc_masterTable.R \
    --pairsFile ${pairs} \
    --inputCont ${contingency} \
    --featureNature ${comboNat} \
    --expName ${comboKey} \
    --sigThreshold ${params.coop_sig_threshold} \
    --minCovAdjust ${params.coop_min_cov_adjust} \
    --minAccFreq ${params.coop_dhs_min_acc_freq} \
    --maxAccFreq ${params.coop_dhs_max_acc_freq} \
    --threads ${task.cpus} \
    --taskMeta ${task.process}
    """
}

// Generate generic (organism-agnostic) QC plots from all the co-occupancy contingency tables of the run
process COOC_QC {
    label 'QC'
    label 'process_medium'

    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.SMF}"
    module { params.use_module ? "${params.tools.module.R}" : '' }

    input:
        path contingency

    output:
        path "*_coop_qc_plots.pdf",         optional: true, emit: plots
        path "*_coop_summary_metrics.tsv",  optional: true, emit: summary
        path "versions.yml",                                emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    Rscript --vanilla ${workflow.projectDir}/bin/smf_cooc_qc.R \
    --inputCont ${contingency.join(',')} \
    --expName ${params.sample_name} \
    --threads ${task.cpus} \
    --taskMeta ${task.process}
    """
}

// // extract binding frequency from the single molecule vector
// process SMF_SM_SORTING_QC {
//     tag "${featureNat}_${sortingMode}"
//     label 'SMF_sort_extract'
//     label 'process_high'

//     conda "${params.tools.conda.SMF}"
//     module { params.use_module ? "${params.tools.module.R}" : '' }

//     input:
//         tuple val(featureNat2), val(sortingMode2), path(smvec)
//         tuple val(featureNat), path(feature), val(sortingMode)
//         val motifString

//     output:
//         path "pattern_*.qs",                                                                    emit: pattern
//         path "states_*.qs", optional: true,                                                     emit: state
//         tuple val(featureNat), val(sortingMode), path("states_frequency_*.qs"), optional: true, emit: freq
//         path "versions.yml",                                                                    emit: versions

//     when:
//     task.ext.when == null || task.ext.when

//     script:
//     """
//     Rscript --vanilla ${workflow.projectDir}/bin/smf_sort_extract.R \
//     --inputSMvec ${smvec} \
//     --featureFile ${feature} \
//     --context ${motifString}\
//     --expName ${params.sample_name} \
//     --featureNature ${featureNat} \
//     --mode ${sortingMode} \
//     --minCoverage ${params.min_cytosine_coverage} \
//     --threads ${task.cpus} \
//     --taskMeta ${task.process}
//     """
// }



