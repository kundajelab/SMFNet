#!/usr/bin/env nextflow
include { FILTER_BAM;QS_FILTER } from '../../modules/local/basecalling'
include { MERGE_BASECALLS as MERGE_INPUT_BAMS} from '../../modules/local/merge'
include { MERGE_CALLS as MERGE_CALLS_BAM_PASS; MERGE_CALLS as MERGE_CALLS_BAM_FAIL; FASTQ as FASTQ_PASS; FASTQ as FASTQ_FAIL} from './wf_basecalling'
include { INDEXCALLS;SPLIT_BAM;ASSIGN_CHUNK } from '../../modules/local/BAM_utils'

workflow BAM_INPUT {
    take:
        reads
        mapping
        output_fastq

    main:
        ch_versions = Channel.empty()

        ref = file("${projectDir}/assets/OPTIONAL_FILE3")
        if (mapping){
            if (!params.ref) {
                throw new Exception(colors.red + "You must provide a reference genome with --ref /Path/to/genome.fa when --mapping used" + colors.reset)
            }
            ref = file(params.ref, checkIfExists: true)
        }
        if(reads.collect().size() == 1){
            INDEXCALLS(reads.collect())
            input=INDEXCALLS.out.reads
            ch_versions = ch_versions.mix(INDEXCALLS.out.versions.first())
        } else {
            MERGE_INPUT_BAMS(reads.collect(), 'input_merge', tuple('bam', 'bai'), tuple(mapping, ref))
            input=MERGE_INPUT_BAMS.out.calls
            ch_versions = ch_versions.mix(MERGE_INPUT_BAMS.out.versions.first())
        }
        
        SPLIT_BAM(input, tuple(mapping, ref))
        ch_versions = ch_versions.mix(SPLIT_BAM.out.versions.first())
        bams=SPLIT_BAM.out.bams.flatten().buffer(size: 1)

        ASSIGN_CHUNK(bams, tuple(mapping, ref))
        ch_versions = ch_versions.mix(ASSIGN_CHUNK.out.versions.first())

        if(mapping){
            FILTER_BAM(ASSIGN_CHUNK.out.bam, ref)
            QS_FILTER(FILTER_BAM.out.cram, tuple(mapping, ref))
            ch_versions = ch_versions
                .mix(FILTER_BAM.out.versions.first())
                .mix(QS_FILTER.out.versions.first())
        } else {
            QS_FILTER(ASSIGN_CHUNK.out.bam,  tuple(mapping, ref))
            ch_versions = ch_versions.mix(QS_FILTER.out.versions.first())
        }

        if (output_fastq){
            log.warn "IF CALLS CONTAIN SECONDARY ALIGNEMENT THOSE WILL BE WRITTEN INTO THE FINAL FASTQ FILE (duplication)!!!"
            FASTQ_PASS(QS_FILTER.out.pass, "pass", mapping, ref)
            ch_versions = ch_versions.mix(FASTQ_PASS.out.versions)
            FASTQ_FAIL(QS_FILTER.out.fail, "fail", mapping, ref)
            // ch_versions = ch_versions.mix(FASTQ_FAIL.out.versions)
        }

        MERGE_CALLS_BAM_PASS(QS_FILTER.out.pass.collect{it[0]}, "pass", mapping, ref)
        ch_versions = ch_versions.mix(MERGE_CALLS_BAM_PASS.out.versions)
        MERGE_CALLS_BAM_FAIL(QS_FILTER.out.fail.collect{it[0]}, "fail", mapping, ref)
        ch_versions = ch_versions.mix(MERGE_CALLS_BAM_FAIL.out.versions)
        
    emit:
        crams_pass = QS_FILTER.out.pass
        merged_pass = MERGE_CALLS_BAM_PASS.out.calls.map { [calls: it] } // Wrap value with 'calls' key for consistency in downstream analysis
        versions = ch_versions
}
