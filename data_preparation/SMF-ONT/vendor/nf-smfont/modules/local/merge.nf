// merge all basecall
process MERGE_BASECALLS {
    label 'ONT_basecalling'
    label 'Merge_Basecalls'
    label 'process_high'
    label 'process_long'

    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.samtools}"
    module { params.use_module ? "${params.tools.module.samtools}" : '' }

    input:
        path calls
        val tag
        tuple val(align_ext), val(index_ext)
        tuple val(mapping), path(ref)

    output:
        tuple path("${params.sample_name}.${tag}.${align_ext}"), path("${params.sample_name}.${tag}.${align_ext}.${index_ext}"), emit: calls
        path "versions.yml"                                                                                                    , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def mapping_spe = mapping ? "--reference ${ref}" : ''
    """
    samtools merge --threads ${task.cpus} \
    -o ${params.sample_name}.${tag}.${align_ext}##idx##${params.sample_name}.${tag}.${align_ext}.${index_ext} \
    -O ${align_ext} --no-PG --write-index ${mapping_spe} ${calls}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        samtools: \$(echo \$(samtools --version 2>&1) | sed 's/^.*samtools //; s/Using.*\$//')
    END_VERSIONS
    """
}

// merge all basecall
process MERGE_SUMMARIES {
    label 'ONT_basecalling'
    label 'Merge_Basecalls'
    label 'process_simple'

    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    // conda "${params.tools.conda.samtools}"
    // module { params.use_module ? "${params.tools.module.samtools}" : '' }

    input:
        path sum
        val tag

    output:
        path "merged_summary.tsv", emit: calls
        path "versions.yml"      , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    first = sum[0]
    """
    files=($sum)

    sed -n 1p \${files[0]} > merged_summary.tsv

    for f in \${files[@]}; do
        sed 1d \$f >> merged_summary.tsv
    done

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        cat: \$(echo \$(cat --version | sed -n 1p | sed 's/cat (GNU coreutils) //g'))
        sed: \$(echo \$(sed --version | sed -n 1p | sed 's/sed (GNU sed) //g'))
    END_VERSIONS
    """
}

// Merge Fastq from each basecall
process MERGE_FQ {
    label 'ONT_basecalling'
    label 'Merge_Basecalls'
    label 'process_long'

    input:
        path fqs
        val tag

    output:
        path "${params.sample_name}.${tag}.fq.gz", emit: fastq
        path "versions.yml"                      , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    zcat ${fqs} | gzip > ${params.sample_name}.${tag}.fq.gz

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        gzip: \$(echo \$(gzip --version 2>&1 | sed -n 1p | sed 's/gzip //g'))
    END_VERSIONS
    """
}

// merge methyl BED files (timp lab format)
process MERGE_MBED {
    label 'SMF_extract_methyl'
    label 'Merge_mBED'
    label 'process_high'
    label 'process_long'

    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.samtools}"
    module { params.use_module ? "${params.tools.module.samtools}:${params.tools.module.tabix}" : '' }

    input:
        path beds
        val tag
        val llr
        val motifTag

    output:
        tuple path("${params.sample_name}_methcalls_*.bed.gz"), path("${params.sample_name}_methcalls_*.bed.gz.tbi"), emit: mbed
        path "versions.yml"                                                                                         , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    llrTag = llr ? "_llr_${llr}" : ''
    """
    zcat ${beds} | LC_ALL=C sort -S2G --parallel=${task.cpus - 2} -k1,1 -k2,2n | bgzip > ${params.sample_name}_methcalls${llrTag}_mot_${motifTag}_${tag}.bed.gz
    sleep 30
    tabix -p bed ${params.sample_name}_methcalls${llrTag}_mot_${motifTag}_${tag}.bed.gz

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        zcat: \$(echo \$(zcat --version 2>&1 | sed -n 1p | sed 's/zcat (gzip) //g'))
        sort: \$(echo \$(sort --version 2>&1 | sed -n 1p | sed 's/sort (GNU coreutils) //g'))
        bgzip: \$(echo \$(bgzip --version 2>&1 | sed -n 1p | sed 's/bgzip (htslib) //g'))
        tabix: \$(echo \$(tabix --version 2>&1 | sed -n 1p | sed 's/tabix (htslib) //g'))
    END_VERSIONS
    """
}

// merge read stats BED files
process MERGE_READSTATS {
    label 'SMF_extract_methyl'
    label 'Merge_mBED'
    label 'process_high'
    label 'process_long'

    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.methylTools}"
    module { params.use_module ? "${params.tools.module.samtools}:${params.tools.module.tabix}" : '' }

    input:
        path readStats
        val tag
        val llr
        val motifTag

    output:
        tuple path("${params.sample_name}_read_stats_*.bed.gz"), path("${params.sample_name}_read_stats_*.bed.gz.tbi"), emit: rstat
        path "versions.yml"                                                                                           , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    llrTag = llr ? "_llr_${llr}" : ''
    """
    merge_presorted_BED.py -i ${readStats} -s 1 | bgzip -@ ${task.cpus - 1} > ${params.sample_name}_read_stats${llrTag}_mot_${motifTag}_${tag}.bed.gz
    sleep 30
    tabix -p bed -S 1 ${params.sample_name}_read_stats${llrTag}_mot_${motifTag}_${tag}.bed.gz

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        Python: \$(echo \$(python --version 2>&1 | sed -n 1p | sed 's/Python //g'))
    \$(cat python_packages.txt)
        tabix: \$(echo \$(tabix --version 2>&1 | sed -n 1p | sed 's/tabix (htslib) //g'))
    END_VERSIONS
    """
}

// Aggregate methylation frequencies
process MERGE_FREQ {
    label 'SMF_extract_methyl'
    label 'Merge_mBED'
    label 'process_high'
    label 'process_long'

    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.methylTools}"

    input:
        path freqs
        val tag
        val llr
        val motifTag
        val std_collapse

    output:
        tuple path("${params.sample_name}_freqs_*.bed.gz"), path("${params.sample_name}_freqs_*.bed.gz.tbi"), emit: freq
        path "versions.yml"                                                                                 , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    llrTag = llr ? "_llr_${llr}" : ''
    filetag = std_collapse ? "freqs_std_collapsed" : "freqs"
    """
    merge_presorted_BED.py -i ${freqs} -s 1 | \
    aggregate_frequencies_BED.py -s 1 | \
    bgzip -@ ${task.cpus - 2} > ${params.sample_name}_${filetag}${llrTag}_mot_${motifTag}_${tag}.bed.gz
    sleep 30
    tabix -p bed -S 1 ${params.sample_name}_${filetag}${llrTag}_mot_${motifTag}_${tag}.bed.gz

    # grep '????\$' python_packages.txt | awk '{split(\$0, s, ": "); split(s[1], n, "python-"); print n[2]}'
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        Python: \$(echo \$(python --version 2>&1 | sed -n 1p | sed 's/Python //g'))
    \$(cat python_packages.txt)
        tabix: \$(echo \$(tabix --version 2>&1 | sed -n 1p | sed 's/tabix (htslib) //g'))
    END_VERSIONS
    """
}
