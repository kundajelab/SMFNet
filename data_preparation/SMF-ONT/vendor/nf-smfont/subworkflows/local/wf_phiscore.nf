#!/usr/bin/env nextflow
include { GENOMIC_CHUNK;PHISCORE_CHUNK;MERGE_PHI } from '../../modules/local/phiscore'
if(params.output_bed){
    include { MERGE_PHISCORE_BED } from '../../modules/local/phiscore'
}

workflow PHISCORE {
    take:
        mBED
        BSgenome

    main:
        ch_versions = Channel.empty()

        GENOMIC_CHUNK(BSgenome)
        ch_versions = ch_versions.mix(GENOMIC_CHUNK.out.versions)
        PHISCORE_CHUNK(mBED, GENOMIC_CHUNK.out.chunk.splitText().map{it -> it.trim()}, BSgenome)
        ch_versions = ch_versions.mix(PHISCORE_CHUNK.out.versions.first())
        MERGE_PHI(
            PHISCORE_CHUNK.out.qs.collect(),
            GENOMIC_CHUNK.out.chr.splitText().map{it -> it.trim()},
            BSgenome
        )
        ch_versions = ch_versions.mix(MERGE_PHI.out.versions)

        if(params.output_bed){
            MERGE_PHISCORE_BED(PHISCORE_CHUNK.out.bed.collect())
            ch_versions = ch_versions.mix(MERGE_PHISCORE_BED.out.versions)
        }

    emit:
       phiscore = MERGE_PHI.out.qs
       versions = ch_versions
}