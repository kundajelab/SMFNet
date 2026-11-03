// Conversion Fast5 file to pod5 for compatibility with old run
process FAST5_TO_POD5 {
    label 'ONT_fast5_convertion'
    label 'error_retry'

    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.fast5converter}"

    input:
        path fast5

    output:
        path "${fast5.baseName}.pod5"   , emit: pod5
        path "versions.yml"             , emit: versions

    script:
    """
    pod5 convert fast5 -o ${fast5.baseName}.pod5 -t ${task.cpus} --strict ${fast5}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        \$(pod5 --version | sed 's/ version//g')
    END_VERSIONS
    """
}

// Split pod5 files to reduce too big files and optimize basecalling chunks
process SPLIT_POD5 {
    label 'process_single'
    stageOutMode 'copy'
    label 'array'

    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.fast5converter}"

    input:
        path pod5
        val batch_size

    output:
        // path "${pod5.simpleName}_${chunk_idx}_split_*.pod5", emit: pod5
        path "${pod5.simpleName}_split_*.pod5", emit: pod5
        path "versions.yml", emit: versions

    script:
    // def outPod5 = fast5.toString().replaceAll(".fast5", ".pod5")
    """
    fileName=\$(basename "$pod5" .pod5)
    readIDs_file=\${fileName}_readIDs.tsv

    # extracting readIDs
    pod5 view $pod5 --ids > \${readIDs_file}

    # Count the number of rows in the file excluding the header (first line) and empty lines
    num_rows=\$(awk 'NR > 1 && NF > 0 {count++} END {print count}' "\${readIDs_file}")

    if [[ \$num_rows -gt $batch_size ]]; then
        awk  -v batch_size="${batch_size}" 'BEGIN{FS="\t";OFS="\t"}{
            if(NR == 1){
                print\$0, "subgroup"
            } else {
                batch_number = int((NR-1)/batch_size) + 1;
                print \$0, "subgp" batch_number
            }
        }' "\${readIDs_file}" > tmp_\${fileName}.tsv
        mv tmp_\${fileName}.tsv \$readIDs_file

        pod5 subset $pod5 --output ./ --table \$readIDs_file --columns subgroup --template "\${fileName}_split_{subgroup}.pod5"

        echo "The input file ($pod5) has \$num_rows rows, which is more than the batch size of $batch_size. Splitted into smaller pod5 files." ""

    else
        cp $pod5 "\${fileName}_split_1.pod5"
        echo "The input file ($pod5) has \$num_rows rows, which is less than or equal to the batch size of $batch_size. No splitting required." ""
    fi

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        \$(pod5 --version | sed 's/ version//g')
    END_VERSIONS
    """
}

// Basecalling process using Dorado
process DORADO {
    label 'ONT_basecalling'
    label 'gpu'
    label 'process_high'
    label 'error_retry'
    label 'array'

    accelerator 1
    container "${params.tools.container.dorado_repo}:${params.tools.container.dorado_sha}"
    module { params.use_module ? "${params.tools.module.dorado}:${params.tools.module.samtools}" : '' }

    input:
        tuple val(chunk_idx), path('*')
        tuple val(basecaller_cfg), path(custom_basecaller_model)
        tuple val(mapping), path(ref), path(mmi_ref)
        tuple val(methyl), val(remora_cfg), path(custom_remora_model)
        // tuple val(methyl), val(remora_cfg), val(custom_basecaller_dir), path('*')

    output:
        tuple path("${chunk_idx}.cram"), path("${chunk_idx}.cram.crai"), val(chunk_idx), emit: cram
        path "*_summary.tsv"                                                           , emit: tsv, optional: true
        path "versions.yml"                                                            , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def model_arg = basecaller_cfg == 'custom' ? "${custom_basecaller_model}" : "\${DRD_MODELS_PATH}/${basecaller_cfg}"
    def basecaller_args = params.basecaller_args ?: ''

    def mapping_arg = mapping ? "--reference ${mmi_ref}" : ''

    def remora_extra =  params.remora_args ?: ''
    def remora_model_args = remora_cfg == 'custom' ? "${custom_remora_model}" : "\${DRD_MODELS_PATH}/${remora_cfg}"
    def remora_args = methyl ? "--modified-bases-models ${remora_model_args} ${remora_extra}" : ''

    // if(remora_cfg == "custom"){
    //     remora_model_args = custom_basecaller_dir
    //         .tokenize(',')
    //         .collect{ "--modified-bases-models ${it}" }
    //         .join(' ')
    // } else {
    //     remora_model_args = remora_cfg
    //         .tokenize(',')
    //         .collect{ "--modified-bases-models \${DRD_MODELS_PATH}/${it}" }
    //         .join(' ')
    // }

    def ref_arg = mapping ? "--reference ${ref}" : ''
    def sorting_arge = mapping ? "samtools sort -@ ${task.cpus.intdiv(4)} --no-PG | " : ''
    def postProc_arg = sorting_arge + "samtools view -@ ${task.cpus.intdiv(4)} --no-PG -O CRAM -o ${chunk_idx}.cram --write-index " + ref_arg

    """
    set +e
    source /opt/nvidia/entrypoint.d/*-gpu-driver-check.sh # runtime driver check msg
    set -e
    echo "GPU used (CUDA_VISIBLE_DEVICES: \${CUDA_VISIBLE_DEVICES})"
    dorado basecaller -v \
        ${model_arg} . \
        ${remora_args} \
        --device ${params.cuda_device} \
        ${basecaller_args} \
        ${mapping_arg} | \
        ${postProc_arg}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        dorado: \$(dorado -v)
        samtools: \$(echo \$(samtools --version 2>&1) | sed 's/^.*samtools //; s/Using.*\$//')
    END_VERSIONS
    """
}

process DORADO_ALIGN {
    label 'ONT_basecalling'
    label 'process_high'

    accelerator 1
    container "${params.tools.container.dorado_repo}:${params.tools.container.dorado_sha}"
    module { params.use_module ? "${params.tools.module.dorado}:${params.tools.module.samtools}" : '' }

    input:
        tuple val(chunk_idx), path(bam), path(bai)
        tuple val(mapping), path(ref), path(mmi_ref)

    output:
        tuple path("${chunk_idx}.cram"), path("${chunk_idx}.cram.crai"), val(chunk_idx), emit: cram
        path "versions.yml"                                                                                                        , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def align_thread = task.cpus.intdiv(2)
    def postProc_thread = task.cpus.intdiv(4)
    """
    set +e
    source /opt/nvidia/entrypoint.d/*-gpu-driver-check.sh # runtime driver check msg
    set -e
    dorado aligner -v -t ${align_thread} ${mmi_ref} ${bam} \
        samtools sort -@ ${postProc_thread}  --no-PG | \
        samtools view -@ ${postProc_thread}  --no-PG -O CRAM -o ${chunk_idx}.cram --write-index --reference ${ref}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        dorado: \$(dorado -v)
        samtools: \$(echo \$(samtools --version 2>&1) | sed 's/^.*samtools //; s/Using.*\$//')
    END_VERSIONS
    """
}

process BONITO_MODELS {
    label 'ONT_basecalling'
    label 'gpu'

    accelerator 1
    conda "${params.tools.conda.bonito}"

    output:
        path "versions.yml", emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    echo "GPU used: \${CUDA_VISIBLE_DEVICES}\n"
    bonito download --models ## necessary for the first time conda usage

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bonito: \$(bonito -v | sed 's/bonito //g')
    END_VERSIONS
    """
}

process BONITO {
    label 'ONT_basecalling'
    label 'gpu'
    label 'process_high'
    label 'process_long'
    label 'error_retry'
    label 'array'

    accelerator 1
    container "${params.tools.container.bonito_repo}:${params.tools.container.bonito_sha}"
    // conda "${params.tools.conda.bonito}"
    module { params.use_module ? "${params.tools.module.CUDA}:${params.tools.module.samtools}" : '' }

    input:
        tuple val(chunk_idx), path('*')
        tuple val(basecaller_cfg), path(custom_basecaller_model)
        tuple val(mapping), path(ref), path(mmi_ref)
        tuple val(methyl), val(remora_cfg), path(custom_remora_model)
        // tuple val(methyl), val(remora_cfg), val(custom_basecaller_dir), path('*')

    output:
        tuple path("${chunk_idx}.cram"), path("${chunk_idx}.cram.crai"), val(chunk_idx), emit: cram
        path "versions.yml"                                                            , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def model_arg = basecaller_cfg == 'custom' ? "${custom_basecaller_model}" : "${basecaller_cfg}"
    def basecaller_args = params.basecaller_args ?: ''

    def mapping_arg = mapping ? "--alignment-threads ${task.cpus.intdiv(6)} --reference ${mmi_ref}" : ''

    def remora_extra =  params.remora_args ?: ''
    def remora_model_args = remora_cfg == 'custom' ? "${custom_remora_model}" : "${remora_cfg}"
    def remora_args = methyl ? "--modified-base-model ${remora_model_args} ${remora_extra}" : ''

    def ref_arg = mapping ? "--reference ${ref}" : ''
    def sorting_arge = mapping ? "samtools sort -@ ${task.cpus.intdiv(4)} --no-PG | " : ''
    def postProc_arg = sorting_arge + "samtools view -@ ${task.cpus.intdiv(4)} --no-PG -O CRAM -o ${chunk_idx}.cram --write-index " + ref_arg

    """
    set +e
    source /opt/nvidia/entrypoint.d/*-gpu-driver-check.sh # runtime driver check msg
    set -e
    echo "GPU used (CUDA_VISIBLE_DEVICES: \${CUDA_VISIBLE_DEVICES})"
    bonito basecaller -v \
        ${model_arg} ./ ${remora_args} \
        --device 'cuda' \
        ${basecaller_args} \
        ${mapping_arg} | \
        ${postProc_arg}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        bonito: \$(bonito -v | sed 's/bonito //g')
        samtools: \$(echo \$(samtools --version 2>&1) | sed 's/^.*samtools //; s/Using.*\$//')
    END_VERSIONS
    """
}

process make_mmi {
    label 'ONT_basecalling'
    label 'process_medium'

    container { params.basecaller == 'dorado' ? "${params.tools.container.dorado_repo}:${params.tools.container.dorado_sha}" : "${params.tools.container.bonito_repo}:${params.tools.container.bonito_sha}" }
    conda "${params.tools.conda.minimap2}"
    module { params.use_module ? "${params.tools.module.minimap2}" : '' }

    input:
        path ref

    output:
        path "ref.mmi"     , emit: mmi
        path "versions.yml", emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    minimap2 -t ${task.cpus} -x lr:hq -d ref.mmi ${ref}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        minimap2: \$(minimap2 --version)
    END_VERSIONS
    """
}

// Remove non-primary (256) and supplemental (2048) alignments (unmapped [4] can be removed with -F 2308)
process FILTER_BAM {
    label 'ONT_basecalling'
    label 'process_single'
    label 'process_multi_threads'
    label 'error_retry'

    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.samtools}"
    module { params.use_module ? "${params.tools.module.samtools}" : '' }

    input:
        tuple path(cram), path(cram_idx), val(chunk_idx)
        path ref

    output:
        tuple path("${chunk_idx}_primaryAlign.cram"), path("${chunk_idx}_primaryAlign.cram.crai"), val(chunk_idx), emit: cram
        path "versions.yml"                                                                                      , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    """
    samtools view -@ ${task.cpus} -F 2304 -o ${chunk_idx}_primaryAlign.cram -O CRAM --no-PG --write-index --reference ${ref} ${cram}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        samtools: \$(echo \$(samtools --version 2>&1) | sed 's/^.*samtools //; s/Using.*\$//')
    END_VERSIONS
    """
}

// Split inot pass and fail groups based on the min qScrore parameter
process QS_FILTER {
    label 'ONT_basecalling'
    label 'process_single'
    label 'process_multi_threads'
    label 'error_retry'

    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.samtools}"
    module { params.use_module ? "${params.tools.module.samtools}" : '' }

    input:
        tuple path(cram), path(cram_idx), val(chunk_idx)
        tuple val(mapping), path(ref)

    output:
        tuple path("${cram.baseName}.pass.cram"), val(chunk_idx), emit: pass
        tuple path("${cram.baseName}.fail.cram"), val(chunk_idx), emit: fail
        path "versions.yml"                                     , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def mapping_spe = mapping ? "--reference ${ref}" : ''
    """
    samtools view -@ ${task.cpus} -e '[qs] >= ${params.qscore_filter}' --output ${cram.baseName}.pass.cram --unoutput ${cram.baseName}.fail.cram -O CRAM --no-PG ${mapping_spe} ${cram}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        samtools: \$(echo \$(samtools --version 2>&1) | sed 's/^.*samtools //; s/Using.*\$//')
    END_VERSIONS
    """
}

//extract fastq from merged BAMs
process CALL2FQ {
    label "ONT_basecalling"

    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.samtools}"
    module { params.use_module ? "${params.tools.module.samtools}:${params.tools.module.htslib}" : '' }

    input:
        path(cram)
        val tag
        tuple val(mapping), path(ref)

    output:
        path "${cram.baseName}.${tag}.fq.gz", emit: fastq
        path "versions.yml"                 , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def ref_arg = mapping ? "--reference ${ref}" : ''
    """
    ### IF CRAM CONTAINS SECONDARY ALIGNEMENT THOSE WILL BE WRITTEN INTO THE FASTQ FILE!!!
    samtools bam2fq ${ref_arg} ${cram} | bgzip > ${cram.baseName}.${tag}.fq.gz

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        samtools: \$(echo \$(samtools --version 2>&1) | sed 's/^.*samtools //; s/Using.*\$//')
    END_VERSIONS
    """
}

//Fix "." implicit syntax in hte MM tag produce by Dorado to fix forward analysis
process MMTAGFIX {
    label 'ONT_basecalling'
    label 'process_single'
    label 'process_multi_threads'
    // label 'error_retry'

    // container = "${params.tools.container.modkit_repo}:${params.tools.container.modkit_sha}"
    container "${params.tools.container.smf_utils_repo}:${params.tools.container.smf_utils_sha}"
    conda "${params.tools.conda.samtools}"

    input:
        tuple path(cram), path(cram_idx), val(chunk_idx)
        tuple val(mapping), path(ref)

    output:
        tuple path("${chunk_idx}_MMfix.cram"), path("${chunk_idx}_MMfix.cram.crai"), val(chunk_idx), emit: cram
        path "versions.yml"                                                                                , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def mapping_spe = mapping ? "--reference ${ref}" : ''
    """
    ${workflow.projectDir}/bin/modkit update-tags ${cram} ${chunk_idx}_MMfix.bam --no-implicit-probs -t ${task.cpus}
    samtools view -@ ${task.cpus} -o ${chunk_idx}_MMfix.cram -O CRAM --write-index --no-PG ${mapping_spe} ${chunk_idx}_MMfix.bam
    rm ${chunk_idx}_MMfix.bam

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        modkit:
        samtools: \$(echo \$(samtools --version 2>&1) | sed 's/^.*samtools //; s/Using.*\$//')
    END_VERSIONS
    """
}
