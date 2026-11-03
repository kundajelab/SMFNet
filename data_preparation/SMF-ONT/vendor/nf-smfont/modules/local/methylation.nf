// Extract methylation information from the BAM into methyl BED format (timp lab format)
process methylBED_modBAM {
    label "SMF_extract_methyl"
    label 'process_multi_threads'
    label 'error_retry'

    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.methylTools}"

    input:
        tuple path(cram), val(chunk_idx)
        path ref
        val motif
        val fileTag

    output:
        tuple path("${chunk_idx}_calls_llr_${params.llr}_${fileTag}.tsv.gz"), val(chunk_idx), emit: mbed
        path "versions.yml"                                                                 , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    motif_arg = motif.collect{ "-m ${it}" }.join(' ')
    """
    samtools view -@ ${task.cpus} -o ${chunk_idx}.bam##idx##${chunk_idx}.bam.bai -O BAM --no-PG --write-index --reference ${ref} ${cram}
    BAM_to_mBED.py -i ${chunk_idx}.bam -l ${params.llr} -r ${ref} ${motif_arg} -t ${task.cpus - 1} | bgzip > ${chunk_idx}_calls_llr_${params.llr}_${fileTag}.tsv.gz

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        samtools: \$(echo \$(samtools --version 2>&1) | sed 's/^.*samtools //; s/Using.*\$//')
        Python: \$(echo \$(python --version 2>&1 | sed -n 1p | sed 's/Python //g'))
    \$(cat python_packages.txt)
        bgzip: \$(echo \$(bgzip --version 2>&1 | sed -n 1p | sed 's/bgzip (htslib) //g'))
    END_VERSIONS
    """
}

process MBED_READSTATS {
    label "SMF_extract_methyl"
    label 'process_multi_threads'

    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.methylTools}"

    input:
        tuple path(mbed), val(chunk_idx)

    output:
       path "${chunk_idx}_read_stats.tsv.gz", emit: rstat
       path "versions.yml"                  , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    readStats_from_mBED.py -v -i ${mbed} | \
    awk 'NR == 1; NR > 1 {{ print \$0 | "LC_ALL=C sort -S2G --parallel=${task.cpus - 2} -k1,1d -k2,2n" }}' | \
    bgzip > ${chunk_idx}_read_stats.tsv.gz

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        Python: \$(echo \$(python --version 2>&1 | sed -n 1p | sed 's/Python //g'))
    \$(cat python_packages.txt)
        awk: \$(echo \$(awk --version 2>&1 | sed -n 1p | sed 's/GNU Awk //g' | cut -d " " -f 1))
        sort: \$(echo \$(sort --version 2>&1 | sed -n 1p | sed 's/sort (GNU coreutils) //g'))
        bgzip: \$(echo \$(bgzip --version 2>&1 | sed -n 1p | sed 's/bgzip (htslib) //g'))
    END_VERSIONS
    """
}

process READSTATS_REPORT {
    label "SMF_extract_methyl"
    label "process_high"
    label 'error_retry'
    //label 'error_retry_ignore' // ignore errors in this process and continue pipeline

    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.SMF}"

    input:
        path readstat
        path BSgenome
        val fileTag
        val motifString

    output:
       path "ReadStat_*_plots"      , emit: folder
       path "ReadStat_*.Rmd"        , emit: md
       path "Report_ReadStat_*.html", emit: html
       path "versions.yml"          , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    Rscript --vanilla ${workflow.projectDir}/bin/readStats_report.R \
    --genome ${params.BSgenome} \
    --expName ${params.sample_name} \
    --filterTag ${fileTag} \
    --mode ${motifString} \
    --threads ${task.cpus} \
    --taskMeta ${task.process}
    """
}

process MBED_FREQ {
    label "SMF_extract_methyl"
    label 'process_multi_threads'

    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.methylTools}"

    input:
        tuple path(mbed), val(chunk_idx)
        path ref
        val std_collapse

    output:
       path "${chunk_idx}_freqs*.tsv.gz", emit: freq
       path "versions.yml"              , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    collapse_arg = std_collapse ? "-r ${ref} -c" : ""
    filetag = std_collapse ? "freqs_std_collapsed" : "freqs"
    """
    freq_from_mBED.py ${collapse_arg} -v -i ${mbed} | \
    awk 'NR == 1; NR > 1 {{ print \$0 | "LC_ALL=C sort -S2G --parallel=${task.cpus - 2} -k1,1d -k2,2n" }}' | \
    bgzip > ${chunk_idx}_${filetag}.tsv.gz

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        Python: \$(echo \$(python --version 2>&1 | sed -n 1p | sed 's/Python //g'))
    \$(cat python_packages.txt)
        awk: \$(echo \$(awk --version 2>&1 | sed -n 1p | sed 's/GNU Awk //g' | cut -d " " -f 1))
        sort: \$(echo \$(sort --version 2>&1 | sed -n 1p | sed 's/sort (GNU coreutils) //g'))
        bgzip: \$(echo \$(bgzip --version 2>&1 | sed -n 1p | sed 's/bgzip (htslib) //g'))
    END_VERSIONS
    """
}

process FT_READ_ID {
    label "SMF_extract_methyl"
    label 'process_multi_threads'

    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.methylTools}"

    input:
        tuple path(mbed), val(chunk_idx)
        tuple val(min_rate), val(max_rate)

    output:
       tuple path(mbed), path("read_ID_${chunk_idx}.gz"), val(chunk_idx)     , emit: reads_ID
       path "methFiltering_min_${min_rate}_max_${max_rate}_${chunk_idx}.json", emit: FT_stats
       path "read_ID_${chunk_idx}.gz"                                        , emit: reads_ID2
       path "versions.yml"                                                   , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    zcat ${mbed} | \
    methyl_filter_mBED.py -l ${min_rate} -u ${max_rate} -r methFiltering_min_${min_rate}_max_${max_rate}_${chunk_idx}.json | \
    bgzip -@ ${task.cpus - 2} > read_ID_${chunk_idx}.gz

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        zcat: \$(echo \$(zcat --version 2>&1 | sed -n 1p | sed 's/zcat (gzip) //g'))
        Python: \$(echo \$(python --version 2>&1 | sed -n 1p | sed 's/Python //g'))
    \$(cat python_packages.txt)
        bgzip: \$(echo \$(bgzip --version 2>&1 | sed -n 1p | sed 's/bgzip (htslib) //g'))
    END_VERSIONS
    """
}

process FT_MBED {
    label "SMF_extract_methyl"
    label 'process_single'
    label 'process_multi_threads'
    label 'error_retry'

    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.methylTools}"

    input:
        tuple path(mbed), path(readID), val(chunk_idx)
        tuple val(min_rate), val(max_rate)

    output:
       tuple path("${chunk_idx}_methFiltered_min_${min_rate}_max_${max_rate}.tsv.gz"), val(chunk_idx), emit: filt_mbed
       path "versions.yml"                                                           , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    methyl_filter_mBED_by_readID.py -i ${mbed} -r ${readID} | \
    LC_ALL=C sort --parallel=${task.cpus - 2} -k1,1 -k2,2n | \
    bgzip > "${chunk_idx}_methFiltered_min_${min_rate}_max_${max_rate}.tsv.gz"

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        Python: \$(echo \$(python --version 2>&1 | sed -n 1p | sed 's/Python //g'))
    \$(cat python_packages.txt)
        sort: \$(echo \$(sort --version 2>&1 | sed -n 1p | sed 's/sort (GNU coreutils) //g'))
        bgzip: \$(echo \$(bgzip --version 2>&1 | sed -n 1p | sed 's/bgzip (htslib) //g'))
    END_VERSIONS
    """
}

// process insilicoBS {
//     label "SMF_extract_methyl"
//     label 'process_high'
//     label 'error_retry'

//     container = "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
//     conda "${params.tools.conda.methylTools}"

//     input:
//         tuple path(cram), path(mbed), val(chunk_idx)
//         path ref
//         val motif
//         val fileTag

//     output:
//        tuple path("${chunk_idx_calls_llr_${params.llr}_${fileTag}.tsv.gz"), val(chunk_idx)

//     script:
//     motif_arg = motif.collect{ "-m ${it}" }.join(' ')
//     """
//     samtools index ${cram}
//     zcat ${mbed} | LC_ALL=C sort -S2G --parallel=${task.cpus - 2} -k1,1 -k2,2n | bgzip > ${chunk_idx}_sorted.tsv.gz
//     tabix -p bed ${chunk_idx}_sorted.tsv.gz
//     inSilico_BS.py -a ${cram} -b ${chunk_idx}_sorted.tsv.gz



//     """
// }

