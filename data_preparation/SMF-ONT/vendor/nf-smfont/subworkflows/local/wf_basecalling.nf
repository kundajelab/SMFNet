#!/usr/bin/env nextflow
include { DORADO;BONITO;BONITO_MODELS;make_mmi;FILTER_BAM;QS_FILTER;CALL2FQ as CALL2FQ_PASS; CALL2FQ as CALL2FQ_FAIL; MMTAGFIX } from '../../modules/local/basecalling'
include { MERGE_BASECALLS as MERGE_BASECALLS_PASS; MERGE_BASECALLS as MERGE_BASECALLS_FAIL; MERGE_BASECALLS as MERGE_BASECALLS_ALL; MERGE_FQ as MERGE_FQ_PASS; MERGE_FQ as MERGE_FQ_FAIL } from '../../modules/local/merge'
include { SUMCALLS; pycoQC as PYCOQC } from '../../modules/local/qc'

workflow BASECALLING {
    take:
        reads
        mapping
        methyl
        colors

    main:
        ch_versions = Channel.empty()

        // Ensure basecaller config is set
        if (!params.basecaller_cfg && !params.basecaller_model_path) {
            throw new Exception(colors.red + "You must provide a basecaller profile with --basecaller_cfg <profile>" + colors.reset)
        }
        
        custom_basecaller_model = file("${projectDir}/assets/OPTIONAL_FILE1")
        if (params.basecaller_cfg.toLowerCase() == "custom"){
            if(!params.basecaller_model_path){
                 throw new Exception(colors.red + "You have selected a custom basecalling model but have not provided the path of the custom model with --basecaller_model_path" + colors.reset)
            }
            custom_basecaller_model = file(params.basecaller_model_path, type: "dir", checkIfExists: true)
            log.warn "Custom basecaller model used: ${params.basecaller_model_path}"
        }

        // Ensure mapping config is set
        mmi_ref = file("${projectDir}/assets/OPTIONAL_FILE2")
        ref = file("${projectDir}/assets/OPTIONAL_FILE3")
        if (mapping){
            if (!params.ref) {
                throw new Exception(colors.red + "You must provide a reference genome with --ref /Path/to/genome.fa when --mapping used" + colors.reset)
            } 
            ref = file(params.ref, checkIfExists: true)
            if(!params.ref_mmi){
                make_mmi(ref)
                mmi_ref = make_mmi.out.mmi
                ch_versions = ch_versions.mix(make_mmi.out.versions.collect())
            } else {
                mmi_ref = file(params.ref_mmi, checkIfExists: true)
            }
        }

        // Ensure modbase config is set
        // custom_modbase_model = Channel.fromPath("${projectDir}/assets/OPTIONAL_FILE4")
        custom_modbase_model = file("${projectDir}/assets/OPTIONAL_FILE4")
        //custom_modbase_dir = 'OPTIONAL_FILE4'
        if (methyl){
            if (!params.remora_cfg && !params.remora_model_path) {
                throw new Exception(colors.red + "You must provide a remora profile with --remora_cfg <profile> when --methyl used" + colors.reset)
            }

            if (params.remora_cfg.toLowerCase() == "custom"){
                if(!params.remora_model_path){
                    throw new Exception(colors.red + "You have selected a custom modbasecalling model but have not provided the path of the custom model with --remora_model_path" + colors.reset)
                }
                custom_modbase_model = file(params.remora_model_path, type: "dir", checkIfExists: true)
                // custom_modbase_model = Channel.fromPath(params.remora_model_path.tokenize(','), type: "dir", checkIfExists: true)
                log.warn "Custom remora model used: ${params.remora_model_path}"
            }
            // custom_modbase_dir = custom_modbase_model.map{ it.getName() }.reduce{ a, b -> "$a,$b" }.collect()
        }

        // Basecalling
        if(params.basecaller.toLowerCase() == "dorado") {
            DORADO(
                reads,
                tuple(params.basecaller_cfg, custom_basecaller_model),
                tuple(mapping, ref, mmi_ref),
                tuple(methyl, params.remora_cfg, custom_modbase_model),
            )
            // tuple(methyl, params.remora_cfg, custom_modbase_dir, custom_modbase_model),
            // if(methyl){
            //     called_crams=MMTAGFIX(tmp, tuple(mapping, ref))
            //     ch_versions = ch_versions.mix(tmp.out.versions.first())
            // } else {
            //     called_crams=tmp
            // }
            called_crams=DORADO.out.cram
            ch_versions = ch_versions.mix(DORADO.out.versions.first())
        } else if (params.basecaller.toLowerCase() == "bonito"){
            // if(conda.enabled) {
            //     BONITO_MODELS()
            // }
            called_crams=BONITO(
                reads,
                tuple(params.basecaller_cfg, custom_basecaller_model),
                tuple(mapping, ref, mmi_ref),
                tuple(methyl, params.remora_cfg, custom_modbase_model),
            )
            called_crams=BONITO.out.cram
            ch_versions = ch_versions.mix(BONITO.out.versions.first())
        } else {
            throw new Exception(colors.red + "Basecaller ${params.basecaller} is not supported, --basecaller <dorado|bonito>" + colors.reset)
        }

        if(mapping){
            FILTER_BAM(called_crams, ref)
            QS_FILTER(FILTER_BAM.out.cram,  tuple(mapping, ref))
            ch_versions = ch_versions
                .mix(FILTER_BAM.out.versions.first())
                .mix(QS_FILTER.out.versions.first())
        } else {
            QS_FILTER(called_crams,  tuple(mapping, ref))
            ch_versions = ch_versions.mix(QS_FILTER.out.versions.first())
        }

    emit:
        crams_pass = QS_FILTER.out.pass
        crams_fail = QS_FILTER.out.fail
        versions = ch_versions  
}

workflow MERGE_CALLS {
    take:
        crams
        group
        mapping
        ref

    main:
        ch_versions = Channel.empty()

        // determine output extentions
        def align_ext = "cram"
        def index_ext = "crai"
        if (params.output_bam) {
            align_ext = "bam"
            index_ext = "bai"
        }

        if(group == 'pass'){
            call_merged = MERGE_BASECALLS_PASS(crams, group, tuple(align_ext, index_ext), tuple(mapping, ref))
        } else if (group == 'fail') {
            call_merged = MERGE_BASECALLS_FAIL(crams, group, tuple(align_ext, index_ext), tuple(mapping, ref))
        } else {
            crams.removeAll {it -> it.contains(".bai")}
            call_merged = MERGE_BASECALLS_ALL(crams, group, tuple(align_ext, index_ext), tuple(mapping, ref))
        }
        ch_versions = ch_versions.mix(call_merged.versions.collect())

        SUMCALLS(call_merged.calls, group, align_ext)
        ch_versions = ch_versions.mix(SUMCALLS.out.versions.collect())

        PYCOQC(call_merged.calls, SUMCALLS.out.sum, group, align_ext, tuple(mapping, ref))
        ch_versions = ch_versions.mix(PYCOQC.out.versions.collect())

    emit:
        calls = call_merged.calls
        pycoqc = PYCOQC.out.html
        versions = ch_versions
}

workflow FASTQ {
    take:
        crams
        group
        mapping
        ref

    main:
        ch_versions = Channel.empty()
        
        if(group == 'pass'){
            CALL2FQ_PASS(crams, group, tuple(mapping, ref))
            MERGE_FQ_PASS(CALL2FQ_PASS.out.fastq.collect(), group)
            ch_versions = ch_versions
                .mix(CALL2FQ_PASS.out.versions.first())
                .mix(MERGE_FQ_PASS.out.versions.collect())
        } else if (group == 'fail') {
            CALL2FQ_FAIL(crams, group, tuple(mapping, ref))
            MERGE_FQ_FAIL(CALL2FQ_FAIL.out.fastq.collect(), group)
            ch_versions = ch_versions
                .mix(CALL2FQ_FAIL.out.versions.first())
                .mix(MERGE_FQ_FAIL.out.versions.collect())
        }

    emit:
        versions = ch_versions
}









