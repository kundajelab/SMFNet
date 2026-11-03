// Extract run info fron BAMs
process SUMCALLS {
    label 'ONT_basecalling'
    label 'gpu'
    label 'process_high'

    accelerator 1
    container "${params.tools.container.dorado_repo}:${params.tools.container.dorado_sha}"
    module { params.use_module ? "${params.tools.module.dorado}" : '' }

    input:
        tuple path(call), path(call_id)
        val group
        val align_ext
    
    output:
        path("${params.sample_name}.${group}_run.summary.txt")  , emit: sum
        path "versions.yml"                                     , emit: versions
    
    when:
    task.ext.when == null || task.ext.when
    
    script:
    """
    if [[ ${align_ext} == "cram" ]]; then
        samtools view -b --no-PG -@ ${task.cpus} ${call} --write-index \
        -o tmpBAM.bam##idx##tmpBAM.bam.bai
    else
        mv ${call} tmpBAM.bam && mv ${call_id} tmpBAM.bam.bai
    fi
    
    dorado summary tmpBAM.bam > ${params.sample_name}.${group}_run.summary.txt

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        dorado: \$(dorado -v)
    END_VERSIONS
    """
}

// Plot run metrix (basecalling and mapping)
process pycoQC {
    label 'QC'
    label 'error_retry_ignore'
    label 'process_high'

    container "${params.tools.container.pycoqc_repo}:${params.tools.container.pycoq_sha}"
    conda "${params.tools.conda.pycoQC}"

    input:
        tuple path(call), path(call_id)
        path sum
        val group
        val align_ext
        tuple val(mapping), path(ref)
   
    output:
        path "pycoQC_report_${params.sample_name}.${group}.html", emit: html
        path "pycoQC_report_${params.sample_name}.${group}.json", emit: json
        path "versions.yml"                                     , emit: versions
    
    when:
    task.ext.when == null || task.ext.when

    script:
    def mapping_spe = mapping ? "--reference ${ref}" : ''
    """
    LIST=(${call})
    for (( i = 0; i < \${#LIST[@]}; i++ )); do
        if [[ ${align_ext} == "cram" ]]; then
            samtools view -b --no-PG -@ ${task.cpus} ${call} --write-index \
            -o tmpBAM_\${i}.bam##idx##tmpBAM_\${i}.bam.bai ${mapping_spe}
        else
            mv \${LIST[\${i}]} tmpBAM_\${i}.bam && mv \${LIST[\${i}]}.bai tmpBAM_\${i}.bam.bai
        fi
    done
    
    pycoQC -q -f ${sum} -a tmpBAM_*.bam \
    --report_title 'SMF-ONT PycoQC report of ${params.sample_name} ${group} reads' \
    -o pycoQC_report_${params.sample_name}.${group}.html \
    -j pycoQC_report_${params.sample_name}.${group}.json

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        pycoQC: \$(pycoQC --version| sed 's/pycoQC v//g')
    END_VERSIONS
    """
}

// Collect methylation rate filtering statistic and report it
process REPORT_FILTERING {
    label "SMF_extract_methyl"
    label 'QC'

    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.methylTools}"

    input:
        path jsons
        tuple val(min_rate), val(max_rate)

    output:
       path "${params.sample_name}_methFiltered_min_${min_rate}_max_${max_rate}.json", emit: jsn
       path "versions.yml"                                                           , emit: versions
    
    script:
    //need to include little ploting + debug repprotint last one
    """
    report_filtering.py -o ${params.sample_name}_methFiltered_min_${min_rate}_max_${max_rate}.json -i ${jsons}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python --version | sed 's/Python //g')
    END_VERSIONS
    """
}
