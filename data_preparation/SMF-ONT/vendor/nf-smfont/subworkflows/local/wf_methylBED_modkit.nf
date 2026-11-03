#!/usr/bin/env nextflow
include { methylBED_modBAM; MBED_READSTATS; MBED_FREQ; READSTATS_REPORT } from '../../modules/local/methylation'
include { MERGE_MBED; MERGE_READSTATS; MERGE_FREQ } from '../../modules/local/merge'

if(params.filter_read_met_rate){
    include { FT_READ_ID;FT_MBED; MBED_READSTATS as MBED_READSTATS_FT; MBED_FREQ as MBED_FREQ_FT; READSTATS_REPORT as READSTATS_REPORT_FT} from '../../modules/local/methylation'
    include { MERGE_MBED as MERGE_MBED_FT; MERGE_READSTATS as MERGE_READSTATS_FT; MERGE_FREQ as MERGE_FREQ_FT } from '../../modules/local/merge'
    include { FILTER_BAM } from '../../modules/local/BAM_utils'
    include { REPORT_FILTERING} from '../../modules/local/qc'
}

workflow METHYLBED_MODKIT {
    take:
        crams
        motifs
        ref
        reads
        readstats
        motifsString
        BSgenome
        colors

    main:
        ch_versions = Channel.empty()

        if (!params.ref) {
            throw new Exception(colors.red + "You must provide a reference genome with --ref /Path/to/genome.fa when --mapping used" + colors.reset)
        } 
        ref = file(params.ref, checkIfExists: true)

        motStr=motifs.join('-')
        
        methylBED_modBAM(crams, ref, motifs, 'methUnfiltered')
        mbeds = methylBED_modBAM.out.mbed
        ch_versions = ch_versions.mix(methylBED_modBAM.out.versions.first())

        if(readstats){
            MBED_READSTATS(mbeds)
            ch_versions = ch_versions.mix(MBED_READSTATS.out.versions.first())
            MERGE_READSTATS(MBED_READSTATS.out.rstat.collect(), 'methUnfiltered', params.llr, motStr)
            ch_versions = ch_versions.mix(MERGE_READSTATS.out.versions)
            READSTATS_REPORT(MERGE_READSTATS.out.rstat, BSgenome, 'methUnfiltered', motifsString)
            ch_versions = ch_versions.mix(READSTATS_REPORT.out.versions)
        } else {
            rstat=null
        }

        MBED_FREQ(mbeds, ref, params.collapse_strand_frequencies)
        ch_versions = ch_versions.mix(MBED_FREQ.out.versions.first())

        MERGE_MBED(mbeds.collect{it[0]}, 'methUnfiltered', params.llr, motStr)
        ch_versions = ch_versions.mix(MERGE_MBED.out.versions)

        MERGE_FREQ(MBED_FREQ.out.freq.collect(), 'methUnfiltered', params.llr, motStr, params.collapse_strand_frequencies)
        ch_versions = ch_versions.mix(MERGE_FREQ.out.versions)

        // Filtering read based on read methylation rate and repeat processes
        if(params.filter_read_met_rate){
            filterTag="methFiltered_min_${params.min_read_met_rate}_max_${params.max_read_met_rate}"

            // determine output extentions
            def align_ext = "cram"
            def index_ext = "crai"
            if (params.output_bam) {
                align_ext = "bam"
                index_ext = "bai"
            }

            FT_READ_ID(mbeds, tuple(params.min_read_met_rate, params.max_read_met_rate))
            ch_versions = ch_versions.mix(FT_READ_ID.out.versions.first())

            FT_MBED(FT_READ_ID.out.reads_ID, tuple(params.min_read_met_rate, params.max_read_met_rate))
            ch_versions = ch_versions.mix(FT_MBED.out.versions.first())

            REPORT_FILTERING(FT_READ_ID.out.FT_stats.collect(), tuple(params.min_read_met_rate, params.max_read_met_rate))
            stat_ft = REPORT_FILTERING.out.jsn
            ch_versions = ch_versions.mix(REPORT_FILTERING.out.versions)
            
            if(readstats){
                MBED_READSTATS_FT(FT_MBED.out.filt_mbed)
                //ch_versions = ch_versions.mix(MBED_READSTATS_FT.out.versions.first())

                MERGE_READSTATS_FT(MBED_READSTATS_FT.out.rstat.collect(),
                    filterTag,
                    params.llr,
                    motStr,
                )
                rstat_ft = MERGE_READSTATS_FT.out.rstat
                //ch_versions = ch_versions.mix(MERGE_READSTATS_FT.out.versions.first())
                READSTATS_REPORT_FT(rstat_ft, BSgenome, filterTag, motifsString)
                //ch_versions = ch_versions.mix(READSTATS_REPORT_FT.out.versions.first())
            } else {
                rstat_ft=null
            }
            MBED_FREQ_FT(FT_MBED.out.filt_mbed, ref, params.collapse_strand_frequencies)
            //ch_versions = ch_versions.mix(MBED_FREQ_FT.out.versions.first())

            MERGE_MBED_FT(
                FT_MBED.out.filt_mbed.collect{it[0]},
                filterTag,
                params.llr,
                motStr,
            )
            mbed_ft = MERGE_MBED_FT.out.mbed
            //ch_versions = ch_versions.mix(MERGE_MBED_FT.out.versions.first())

            MERGE_FREQ_FT(
                MBED_FREQ_FT.out.freq.collect(),
                filterTag,
                params.llr,
                motStr,
                params.collapse_strand_frequencies,
            )
            freq_ft = MERGE_FREQ_FT.out.freq
            //ch_versions = ch_versions.mix(MERGE_FREQ_FT.out.versions.first())

            FILTER_BAM(
                reads,
                tuple(align_ext, index_ext),
                FT_READ_ID.out.reads_ID2.collect(),
                ref,
                filterTag
            )
            ch_versions = ch_versions.mix(FILTER_BAM.out.versions)
        } else {
            mbed_ft = null
            rstat_ft = null
            freq_ft = null
            stat_ft = null
        }

    emit:
        mbed = MERGE_MBED.out.mbed
        rstat = MERGE_READSTATS.out.rstat
        freq = MERGE_FREQ.out.freq
        mbed_ft
        rstat_ft
        freq_ft
        stat_ft
        versions = ch_versions
}
