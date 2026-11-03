// Calculate how much chunk to run
process GENOMIC_CHUNK {
    label "Phiscore"
    label 'process_high'
    
    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.SMF}"
    module { params.use_module ? "${params.tools.module.R}" : '' }
    
    input:
        path BSgenome

    output:
        path "chunk.txt"    , emit: chunk
        path "chr.txt"      , emit: chr
        path "versions.yml" , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    Rscript --vanilla ${workflow.projectDir}/bin/smf_phiscore_slidingWin.R \
    --genome ${params.BSgenome} \
    --tileWindows ${params.sliding_win_size} \
    --threads ${task.cpus} \
    --taskMeta ${task.process}
    """
}

// Calculate Phiscore between each cytosines within the genomic chunk
process PHISCORE_CHUNK {
    label "Phiscore"
    label 'process_high'
    label 'process_long'
    label 'process_high_memory'
    label 'array'
    
    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.SMF}"
    module { params.use_module ? "${params.tools.module.R}" : '' }
    
    input:
        tuple path(mbed), path(tbi)
        val chunk
        path BSgenome

    output:
        path "*.qs"         , emit: qs
        path "*.bed.gz"     , optional: true, emit: bed
        path "versions.yml" , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    pseudo_arg = params.pseudocount ? "--pseudocount " : ''
    outlier_arg = params.outliersFilter ? "--outliersFilter " : ''
    bed_arg = params.output_bed ? "--bedFile " : ''
    extra_arg = pseudo_arg + outlier_arg + bed_arg
    """
    Rscript --vanilla ${workflow.projectDir}/bin/smf_phiscore.R \
    --genome ${params.BSgenome} \
    --expName ${params.sample_name} \
    --chunk ${chunk} \
    --tileWindows ${params.sliding_win_size} ${extra_arg} \
    --threads ${task.cpus} \
    --taskMeta ${task.process}
    """
}

// merge Phiscore per chromosome
process MERGE_PHI {
    label "Phiscore"
    label 'process_high'
    label 'process_long'
    label 'process_high_memory'
    
    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.SMF}"
    module { params.use_module ? "${params.tools.module.R}" : '' }
    
    input:
        path qs
        val chr
        path BSgenome

    output:
        path "Phiscore_${params.sample_name}_${chr}.qs" , emit: qs
        path "versions.yml"                             , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    Rscript --vanilla ${workflow.projectDir}/bin/smf_phiscore_merge.R \
    --genome ${params.BSgenome} \
    --expName ${params.sample_name} \
    --chr ${chr} \
    --tileWindows ${params.sliding_win_size} \
    --threads ${task.cpus} \
    --taskMeta ${task.process}
    """
}

// merge Phiscore BED files
process MERGE_PHISCORE_BED {
    label 'Phiscore'
    label 'process_high'
    label 'process_long'

    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.samtools}"
    module { params.use_module ? "${params.tools.module.samtools}:${params.tools.module.tabix}" : '' }

    input:
        path beds

    output:
        tuple path("Phiscore_${params.sample_name}.bed.gz"), path("Phiscore_${params.sample_name}.bed.gz.tbi"), emit: bed
        path "versions.yml"                                                                                   , emit: versions

    when:
    task.ext.when == null || task.ext.when
    
    script:
    """
    zcat ${beds} | LC_ALL=C sort -S2G --parallel=${task.cpus - 2} -k1,1 -k2,2n | bgzip > Phiscore_${params.sample_name}.bed.gz
    sleep 30
    tabix -p bed Phiscore_${params.sample_name}.bed.gz

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        zcat: \$(echo \$(zcat --version 2>&1 | sed -n 1p | sed 's/zcat (gzip) //g'))
        sort: \$(echo \$(sort --version 2>&1 | sed -n 1p | sed 's/sort (GNU coreutils) //g'))
        bgzip: \$(echo \$(bgzip --version 2>&1 | sed -n 1p | sed 's/bgzip (htslib) //g'))
        tabix: \$(echo \$(tabix --version 2>&1 | sed -n 1p | sed 's/tabix (htslib) //g'))
    END_VERSIONS
    """
}
