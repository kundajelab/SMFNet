/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT LOCAL MODULES/SUBWORKFLOWS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

//
// SUBWORKFLOW: Consisting of a mix of local and nf-core/modules
//
include { FAST5_TO_POD5                   } from '../modules/local/basecalling'
include { SPLIT_POD5                      } from '../modules/local/basecalling'
include { PARSEBAM                        } from '../modules/local/BAM_utils'
include { BSGENOME                        } from '../modules/local/smf'

include { logColours                      } from '../subworkflows/nf-core/utils_nfcore_pipeline'
include { BASECALLING                     } from '../subworkflows/local/wf_basecalling'
include { MERGE_CALLS as MERGE_CALLS_PASS } from '../subworkflows/local/wf_basecalling'
include { MERGE_CALLS as MERGE_CALLS_FAIL } from '../subworkflows/local/wf_basecalling'
include { MERGE_CALLS as MERGE_CALLS_ALL  } from '../subworkflows/local/wf_basecalling'
include { FASTQ as FASTQ_PASS             } from '../subworkflows/local/wf_basecalling'
include { FASTQ as FASTQ_FAIL             } from '../subworkflows/local/wf_basecalling'
include { BAM_INPUT                       } from '../subworkflows/local/wf_bam_inputs'
include { METHYLBED_MODKIT                } from '../subworkflows/local/wf_methylBED_modkit'
include { METHYLBED_MODBAM                } from '../subworkflows/local/wf_methylBED'
include { BULKSMF; SMF_SORT; COOP         } from '../subworkflows/local/wf_SMF'
include { PHISCORE                        } from '../subworkflows/local/wf_phiscore'

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT NF-CORE MODULES/SUBWORKFLOWS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

//
// MODULE: Installed directly from nf-core/modules
//
include { MULTIQC                } from '../modules/nf-core/multiqc/main'
include { paramsSummaryMap       } from 'plugin/nf-validation'
include { paramsSummaryMultiqc   } from '../subworkflows/nf-core/utils_nfcore_pipeline'
include { softwareVersionsToYAML } from '../subworkflows/nf-core/utils_nfcore_pipeline'
include { methodsDescriptionText } from '../subworkflows/local/utils_nfcore_smfont_pipeline'

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    RUN MAIN WORKFLOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

workflow SMFONT {
    take:
    ch_samplesheet // channel: samplesheet read in from --input

    main:
    ch_versions = Channel.empty()
    ch_multiqc_files = Channel.empty()

    Map colors = logColours(params.monochrome_logs)

    /*
    Input parameters global check
    */

    if (!params.basecall && !params.mapping && !params.methyl && !params.metBED && !params.metFreq && !params.SMsorting && !params.coop && !params.nucl_reg && params.input_ext.toLowerCase() != 'fast5') {
        throw new Exception(colors.red + "No work to be done! Choose one or more workflows to run from [--basecall, --mapping, --methyl, --metBED, --metFreq, --SMsorting --coop --phiscore]" + colors.reset)
    } else {
        basecall = false
        if(params.basecall){basecall=true}
        mapping = false
        if(params.mapping){mapping=true}
        methyl =false
        if(params.methyl){methyl=true}
        metBED = false
        if(params.metBED){metBED=true}
        metFreq = false
        if(params.metFreq){metFreq=true}
        SMsorting = false
        if(params.SMsorting){SMsorting=true}
        coop = false
        if(params.coop){coop=true}
        phiscore = false
        if(params.nucl_reg){phiscore=true}
    }

    /*
    TODO: later
        => input: samplesheet (csv or tsv) with folder | sample_name | input_ext column
        To allow multiple samples to be processed in one run
        example:
        sample,dirs,ext
        S1,/path/to/pod5/files1/:/path/to/pod5/files2/,pod5:fast5
        S2,/path/to/fast5/files/,fast5
        S3,/path/to/bam/files/,bam
        S4,/path/to/cram/files/,cram
    */

    if (!params.input_folder) {
        throw new Exception(colors.red + "Missing required input_folder argument, use --input_folder" + colors.reset)
    } else {
        output_fastq=false
        if(params.output_fastq){output_fastq=true}

        if (!params.input_ext) {
            throw new Exception(colors.red + "Missing required input_ext argument, use --input_ext" + colors.reset)
        } else {
            input_ch = Channel.fromPath(params.input_folder.tokenize(',').collect{ "${it}/**.${params.input_ext}" } , checkIfExists: true, maxDepth: 0)
        }

        if (params.input_ext.toLowerCase() ==~ /bam|cram/){
            if(params.basecall){
                log.warn "Basecalling is not supported on .${params.input_ext} files. Skipping basecalling..."
            }
            basecall=false

            mmi_ref = file("${projectDir}/assets/OPTIONAL_FILE2")
            ref = file("${projectDir}/assets/OPTIONAL_FILE3")
            if (mapping){
                if (!params.ref) {
                    throw new Exception(colors.red + "You must provide a reference genome with --ref /Path/to/genome.fa when --mapping used" + colors.reset)
                }
                ref = file(params.ref, checkIfExists: true)
            }

            PARSEBAM(input_ch.collect(), ref)

            mapping=PARSEBAM.out.mapping
            if(!mapping){ref = file("${projectDir}/assets/OPTIONAL_FILE3")}
            methyl=PARSEBAM.out.methyl

            BAM_INPUT(input_ch, mapping, output_fastq)
            ch_versions = ch_versions.mix(BAM_INPUT.out.versions)

            crams_pass=BAM_INPUT.out.crams_pass
            merged_pass=BAM_INPUT.out.merged_pass
        } else if (params.input_ext.toLowerCase() ==~ /pod5|fast5/){

            // define batch size for splitting pod5 files
            batch_size = 4001

            if (params.input_ext.toLowerCase() == 'fast5') {
                FAST5_TO_POD5(input_ch)
                ch_versions = ch_versions.mix(FAST5_TO_POD5.out.versions.first())
                SPLIT_POD5(FAST5_TO_POD5.out.pod5, batch_size)
            } else {
                SPLIT_POD5(input_ch, batch_size)
            }

            Integer chunk_idx = 0
            split_pod5_ch = SPLIT_POD5.out.pod5
                .flatten()
                .toSortedList()
                .flatten()
                .buffer(size:25, remainder: true)
                .map { tuple(chunk_idx++, it) }
            ch_versions = ch_versions.mix(SPLIT_POD5.out.versions.first())
        } else {
            throw new Exception(colors.red + "${params.input_ext} not supported, use one of --input_ext pod5|fast5|bam|cram" + colors.reset)
        }
    }


    /*
    Base calling using Dorado or Bonito
    */

    if(basecall){
        BASECALLING(split_pod5_ch, mapping, methyl, colors)
        crams_pass = BASECALLING.out.crams_pass
        ch_versions = ch_versions.mix(BASECALLING.out.versions)

        mmi_ref = file("${projectDir}/assets/OPTIONAL_FILE2")
        ref = file("${projectDir}/assets/OPTIONAL_FILE3")
        if (mapping){
            if (!params.ref) {
                throw new Exception(colors.red + "You must provide a reference genome with --ref /Path/to/genome.fa when --mapping used" + colors.reset)
            }
            ref = file(params.ref, checkIfExists: true)
        }
        if (output_fastq){
            log.warn "IF CALLS CONTAIN SECONDARY ALIGNEMENT THOSE WILL BE WRITTEN INTO THE FINAL FASTQ FILE (duplication)!!!"
            FASTQ_PASS(crams_pass, "pass", mapping, ref)
            ch_versions = ch_versions.mix(FASTQ_PASS.out.versions)
            FASTQ_FAIL(BASECALLING.out.crams_fail, "fail", mapping, ref)
            ch_versions = ch_versions.mix(FASTQ_FAIL.out.versions)
        }

        merged_pass=MERGE_CALLS_PASS(crams_pass.collect{it[0]}, "pass", mapping, ref)
        ch_versions = ch_versions.mix(merged_pass.versions)

        merged_fail=MERGE_CALLS_FAIL(BASECALLING.out.crams_fail.collect{it[0]}, "fail", mapping, ref)
        ch_versions = ch_versions.mix(merged_fail.versions)

        // ch_multiqc_files = ch_multiqc_files
        //    .mix(MERGE_CALLS_PASS.out.pycoqc.collect())
        //    .mix(MERGE_CALLS_FAIL.out.pycoqc.collect())

        // merged_all = merged_pass
        //     .merge(merged_fail)
        //     .collect()
        //     .filter(/.*bam$/)
        //     .dump(tag: 'toto')
        // MERGE_CALLS_ALL(, "All", mapping, ref, null)
    }

    /*
    Extract Methylation from the BAM
    */

    if(metFreq || params.readstats){
        if(params.BSgenome_path){
            BSgenome = file(params.BSgenome_path, checkIfExists: true)
        } else {
            BSGENOME(params.BSgenome)
            BSgenome = BSGENOME.out.BSgenome
            ch_versions = ch_versions.mix(BSGENOME.out.versions)
        }
    } else {
        BSgenome = null
    }

    if(metBED){
        if(! mapping || ! methyl){
            throw new Exception(colors.red + "mBED can be generated only on mapped and methyl-called reads..." + colors.reset)
        }
        def supMots = ['CG', 'GC', 'A']
        motifsBED=params.motifs
            .tokenize(',')
            .collect { it.toLowerCase() == 'cpg' ? 'CG' : it }
            .collect { it.toLowerCase() == 'gpc' ? 'GC' : it }
            .collect { it.toLowerCase() == 'a' ? 'A' : it }
            .collect { it.toLowerCase() == 'c' ? 'C' : it }
            .each {
                if(it !in supMots){
                    throw new Exception(colors.red + "${it} motif not supported" + colors.reset)
                }
                if(it == 'C'){
                    log.warn "${it} motif not supported. Skipping..."
                }
            }

            // .each {
            //     if(it !in supMots){
            //         log.warn "${it} motif not supported. Skipping..."
            //     }
            // }

        // motifs.retainAll { it in supMots }

        // if (motifs.isEmpty()){
        //     throw new Exception(colors.red + "No supported motif provided to extract methylation. Supported motifs: ${supMots.join(' ')}" + colors.reset)
        // }

        modkit = false
        if(params.modkit){
            modkit=true
            mBEDdata = METHYLBED_MODKIT(
                crams_pass,
                motifsBED,
                ref,
                merged_pass.calls,
                params.readstats,
                params.motifs,
                BSgenome,
                colors,
            )
        } else {
            mBEDdata = METHYLBED_MODBAM(
                crams_pass,
                motifsBED,
                ref,
                merged_pass.calls,
                params.readstats,
                params.motifs,
                BSgenome,
                colors,
            )
        }
        //ch_versions = ch_versions.mix(mBEDdata.versions)
    }


    /*
    SMF analysis in 3 parts:
        - Bulk Methylation signal: QCs, comparison and visualisation
        - Single Molecule sorting
        - Occupancy and Co-occupancy of TF GW analysis
    */


    if(metFreq){
        if(!params.collapse_strand_frequencies){
             throw new Exception(colors.red + "SMFONT analysis is supported only on strand collapsed data..." + colors.reset)
        } else if (!metBED){
             throw new Exception(colors.red + "SMF bulk analysis requires mBED step to work" + colors.reset)
        }

        if(params.filter_read_met_rate){
            inputFreq=mBEDdata.freq.merge(mBEDdata.freq_ft).collect()
        } else {
            inputFreq=mBEDdata.freq.collect()
        }

        BULKSMF(
            inputFreq,
            params.motifs,
            BSgenome,
        )
        ch_versions = ch_versions.mix(BULKSMF.out.versions)
    }

    if(SMsorting){
        if(!params.features){
             throw new Exception(colors.red + "To perform SMFONT single molecule sorting, a 'features' argument is required." + colors.reset)
        } else if (!params.featureNature){
             throw new Exception(colors.red + "The nature of the features is not provided. ['TF', 'TSS' or 'DHS']" + colors.reset)
        } else if (!metBED){
             throw new Exception(colors.red + "SMFONT single molecule sorting requires mBED step to work" + colors.reset)
        }

        supFeatureType = ['TF', 'TSS', 'DHS']
        featureL = params.featureNature.tokenize(',').collect{ it.toUpperCase() }

        featureNat_ch = Channel.fromList(
            featureL
            .each {
                if(it !in supFeatureType){
                    throw new Exception(colors.red + "${it} feature type is not supported!" + colors.reset)
                }
            }
        )

        // checking if correct Feature nature and settings:
            // TODO: remove this when merginf feature form the same type is dev...
            featureNat_ch
                .count()
                .collect{
                    if(it > 3) {throw new Exception(colors.red + "More than three feature types is not supported yet!" + colors.reset)}
                }

           // Adjust the hybrid settings according to user's input
           hybridSorting = 'TF' in featureL && 'TSS' in featureL ? params.TF_hybrid_sorting : false

            // featureNat_ch.unique()
            //     .count()
            //     .merge(count)
            //     .reduce{ if(a != b) {throw new Exception(colors.red + "Providing multiple time the same feature type is not supported yet!" + colors.reset)} }
            //     .collect{
            //         if(it != count.collect()) {throw new Exception(colors.red + "Providing multiple time the same feature type is not supported yet!" + colors.reset)}
            //     }

        if(params.filter_read_met_rate){
            inputmBED=mBEDdata.mbed_ft
        } else {
            inputmBED=mBEDdata.mbed
        }

        SMF_SORT(
            inputmBED,
            params.motifs,
            BSgenome,
            Channel.fromPath(params.features.tokenize(',').collect() , checkIfExists: true, maxDepth: 0),
            featureNat_ch,
            hybridSorting
        )
        ch_versions = ch_versions.mix(SMF_SORT.out.versions)

    }

    if(params.coop){
        if(!params.features){
             throw new Exception(colors.red + "To perform SMFONT single molecule sorting, a 'features' argument is required." + colors.reset)
        } else if (!params.featureNature){
             throw new Exception(colors.red + "The nature of the features is not provided. ['TF', 'TSS' or 'DHS']" + colors.reset)
        } else if (!metBED){
             throw new Exception(colors.red + "SMFONT co-occurance analysis requires mBED step to work" + colors.reset)
        } else if (!SMsorting){
             throw new Exception(colors.red + "SMFONT co-occurance analysis requires SMFONT single molecule sorting step to work" + colors.reset)
        }

        featureByNature = featureNat_ch.merge(
            Channel.fromPath(params.features.tokenize(',').collect(), checkIfExists: true, maxDepth: 0)
        )

        COOP(
            featureByNature,
            SMF_SORT.out.smvec,
            SMF_SORT.out.freq,
        )
        ch_versions = ch_versions.mix(COOP.out.versions)
    }


    // SMFONT analysis of nucleosomal array patterns
    if(phiscore){
        if (!metBED){
            throw new Exception(colors.red + "Phi score analysis requires mBED step to work" + colors.reset)
        }
        if(mBEDdata.mbed_ft){
            inputmBED=mBEDdata.mbed_ft
        } else {
            inputmBED=mBEDdata.mbed
        }

        PHISCORE(
            inputmBED,
            BSgenome,
        )
        ch_versions = ch_versions.mix(PHISCORE.out.versions)
    }

    //
    // Collate and save software versions
    //
    softwareVersionsToYAML(ch_versions)
        .collectFile(storeDir: "${params.outdir}/pipeline_info", name: 'nf_core_pipeline_software_mqc_versions.yml', sort: true, newLine: true)
        .set { ch_collated_versions }

    //
    // MODULE: MultiQC
    //

    ch_multiqc_config                     = Channel.fromPath("$projectDir/assets/multiqc_config.yml", checkIfExists: true)
    ch_multiqc_custom_config              = params.multiqc_config ? Channel.fromPath(params.multiqc_config, checkIfExists: true) : Channel.empty()
    ch_multiqc_logo                       = params.multiqc_logo ? Channel.fromPath(params.multiqc_logo, checkIfExists: true) : Channel.empty()

    summary_params                        = paramsSummaryMap(workflow, parameters_schema: "nextflow_schema.json")
    ch_workflow_summary                   = Channel.value(paramsSummaryMultiqc(summary_params))

    ch_multiqc_custom_methods_description = params.multiqc_methods_description ? file(params.multiqc_methods_description, checkIfExists: true) : file("$projectDir/assets/methods_description_template.yml", checkIfExists: true)
    ch_methods_description                = Channel.value(methodsDescriptionText(ch_multiqc_custom_methods_description))

    ch_multiqc_files                      = ch_multiqc_files.mix(ch_workflow_summary.collectFile(name: 'workflow_summary_mqc.yaml'))
    ch_multiqc_files                      = ch_multiqc_files.mix(ch_collated_versions)
    ch_multiqc_files                      = ch_multiqc_files.mix(ch_methods_description.collectFile(name: 'methods_description_mqc.yaml', sort: false))

    MULTIQC (
        ch_multiqc_files.collect(),
        ch_multiqc_config.toList(),
        ch_multiqc_custom_config.toList(),
        ch_multiqc_logo.toList(),
        [],
        []
    )

    emit:
    multiqc_report = MULTIQC.out.report.toList() // channel: /path/to/multiqc_report.html
    versions       = ch_versions                 // channel: [ path(versions.yml) ]
}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
