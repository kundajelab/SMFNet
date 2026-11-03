// index BAM or CRAM
process INDEXCALLS{
    label 'ONT_basecalling'

    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.samtools}"
    module { params.use_module ? "${params.tools.module.samtools}" : '' }

    input:
        path reads

    output:
        tuple path(reads), path("${reads.baseName}.${reads.extension}.*"), emit: reads
        path "versions.yml"                                              , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    samtools index ${reads}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        samtools: \$(echo \$(samtools --version 2>&1) | sed 's/^.*samtools //; s/Using.*\$//')
    END_VERSIONS
    """
}

// determine if input SAMs/BAMs/CRAMs are mapped and contain methyl info
process PARSEBAM {
    label 'BAM_input'
    // executor 'local'

    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.samtools}"
    module { params.use_module ? "${params.tools.module.samtools}" : '' }

    input:
        path reads
        path ref

    output:
        env mapping, emit: mapping
        env methyl, emit: methyl

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    input=(${reads})
    mapping_test=()
    methyl_test=()
    for r in \${input[@]}; do
        SQ_len=\$(samtools view -H \$r | grep -c '^@SQ')
        if [[ \${SQ_len} > 0 ]]; then
            mapping_test+=(true)
            ref_contig=\$(cat ${ref} | grep -c '^>')
            if [[ ! \${SQ_len} -eq \${ref_contig} ]]; then
                printf "The number reference contigs wihtin the BAM provided is different than the number of contig in the reference genome file..."
                exit 1
            fi
        else
            mapping_test+=(false)
        fi

        if [[ \$(samtools view \$r | head -n 1000 | grep -c 'MM:') -eq 1000 ]]; then
            methyl_test+=(true)
        else
            methyl_test+=(false)
        fi
    done

    if [[ \$(printf "%s\n" "\${mapping_test[@]}" | sort -u | wc -l) -ne 1 ]]; then
        printf "All Bam files provided should present the same carateristics (all mapped or unmapped)"
        exit 1
    elif [[ \${mapping_test[*]} == *true* ]]; then
        mapping=true
    else
        mapping=false
    fi

    if [[ \$(printf "%s\n" "\${methyl_test[@]}" | sort -u | wc -l) -ne 1 ]]; then
        printf "All Bam files provided should present methylation information encoded in MM/ML tags"
        exit 1
    elif [[ \${methyl_test[*]} == *true* ]]; then
        methyl=true
    else
        methyl=false
    fi
    """
}

// split merged input BAMs in chunks
process SPLIT_BAM {
    label 'BAM_input'
    label 'process_medium'
    label 'process_long'

    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.bamutils}"
    module { params.use_module ? "${params.tools.module.picard}" : '' }

    input:
        tuple path(merged_bam), path(merged_bam_idx)
        tuple val(mapping), path(ref)

    output:
        path "${params.sample_name}_split_*.bam", emit: bams
        path "versions.yml"                     , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def mapping_spe = mapping ? "--REFERENCE_SEQUENCE ${ref}" : ''
    def avail_mem = (task.memory.mega*0.8).intValue()
    def numReads = 4000 * 25
    """
    export LC_ALL=C
    picard \\
        -Xmx${avail_mem}M \\
        SplitSamByNumberOfReads \\
        --INPUT ${merged_bam} \\
        --OUTPUT ./ ${mapping_spe} \\
        --SPLIT_TO_N_READS ${numReads} \\
        --OUT_PREFIX ${params.sample_name}_split

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        picard: \$(echo \$(picard SplitSamByNumberOfReads --version 2>&1) | grep -o 'Version:.*' | cut -f2- -d:)
    END_VERSIONS
    """
}

// convert each splited BAM in cram and assign chunk number
process ASSIGN_CHUNK {
    label 'BAM_input'

    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.samtools}"
    module { params.use_module ? "${params.tools.module.samtools}" : '' }

    input:
        path(bam)
        tuple val(mapping), path(ref)

    output:
        tuple path("${params.sample_name}_*.cram"), path("${params.sample_name}_*.cram.crai"), env(chunk_idx), emit: bam
        path "versions.yml"                                                                                  , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    chunk_idx=${bam.baseName}
    chunk_idx=\${chunk_idx/${params.sample_name}_split_/}
    samtools view -@ ${task.cpus} \
    -o ${params.sample_name}_\${chunk_idx}.cram -O CRAM \
    --no-PG --write-index --reference ${ref} ${bam}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        samtools: \$(echo \$(samtools --version 2>&1) | sed 's/^.*samtools //; s/Using.*\$//')
    END_VERSIONS
    """
}

// filter BAM file on read names
process FILTER_BAM {
    label 'SMF_extract_methyl'
    label 'Merge_Basecalls'
    label 'process_high'
    label 'error_retry'

    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.methylTools}"

    input:
        tuple path(call), path(call_id)
        tuple val(align_ext), val(index_ext)
        path qnames
        path ref
        val fileTag

     output:
        tuple path("${params.sample_name}.pass_${fileTag}.${align_ext}"), path("${params.sample_name}.pass_${fileTag}.${align_ext}.${index_ext}"), emit: merged_bam
        path "versions.yml"                                                                                                                      , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    zcat ${qnames} > qnames_list.txt
    samtools view -@ ${task.cpus} -N qnames_list.txt \
    -o ${params.sample_name}.pass_${fileTag}.${align_ext}##idx##${params.sample_name}.pass_${fileTag}.${align_ext}.${index_ext} \
    -O ${align_ext} --write-index --reference ${ref} ${call}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        zcat: \$(echo \$(zcat --version 2>&1 | sed -n 1p | sed 's/zcat (gzip) //g'))
        samtools: \$(echo \$(samtools --version 2>&1) | sed 's/^.*samtools //; s/Using.*\$//')
    END_VERSIONS
    """
}
