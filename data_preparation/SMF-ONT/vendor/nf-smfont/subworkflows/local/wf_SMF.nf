#!/usr/bin/env nextflow
include { REF_SITES;SMF_EXTRACT;SMF_BULK_QC;SMF_PUB_STAT;SMF_BW;SM_SORT;SM_EXTRACT;COOC_MAKEPAIRS;COOC_CONTINGENCY;COOC_MASTERTABLE;COOC_QC } from '../../modules/local/smf'

if(params.TF_hybrid_sorting){
   include { TF_HYBRID_SORT;SM_EXTRACT as SM_EXTRACT_HYBRID } from '../../modules/local/smf'
}

workflow BULKSMF {
    take:
        freqBED
        motifs
        BSgenome

    main:
        ch_versions = Channel.empty()

        REF_SITES(motifs, BSgenome)
        ch_versions = ch_versions.mix(REF_SITES.out.versions)

        SMF_EXTRACT(motifs, BSgenome, REF_SITES.out.qs, params.filter_read_met_rate, freqBED)
        ch_versions = ch_versions.mix(SMF_EXTRACT.out.versions)

        SMF_PUB_STAT(SMF_EXTRACT.out.qc, params.filter_read_met_rate)

        SMF_BULK_QC(motifs, SMF_EXTRACT.out.bulkSignal, SMF_EXTRACT.out.qc)
        ch_versions = ch_versions.mix(SMF_BULK_QC.out.versions)

        SMF_BW(motifs, BSgenome, SMF_EXTRACT.out.bulkSignal)
        ch_versions = ch_versions.mix(SMF_BW.out.versions)

    emit:
        bulk = SMF_EXTRACT.out.bulkSignal
        versions = ch_versions
}

workflow SMF_SORT {
    take:
        mBED
        motifs
        BSgenome
        feature
        nature
        hybridSorting

    main:
        ch_versions = Channel.empty()

        sorint_opt = Channel.of(['single', 'TF'], ['cluster', 'TF'], ['downstream', 'TSS'], ["${params.bins_size}", 'DHS'])
        feature_input = feature.merge(nature).combine(sorint_opt, by: 1)

        SM_SORT(
            feature_input,
            motifs,
            BSgenome,
            mBED,
        )
        ch_versions = ch_versions.mix(SM_SORT.out.versions.first())

        SM_EXTRACT(SM_SORT.out.SMvec, SM_SORT.out.Feature, motifs)
        ch_versions = ch_versions.mix(SM_EXTRACT.out.versions.first())

        if(hybridSorting){
            SM_SORT.out.SMvec
                .branch{
                    TSS: it[0] == 'TSS'
                    TF: it[0] == 'TF'
                }
                .set{SMvec}

            SM_EXTRACT.out.freq
                .branch{TSS: it[0] == 'TSS'}
                .set{freq}

            TF_HYBRID_SORT(SMvec.TF.collect{it[2]}, feature.merge(nature).filter{it[1] == 'TF'})
            ch_versions = ch_versions.mix(TF_HYBRID_SORT.out.versions)

            SM_EXTRACT_HYBRID(
                TF_HYBRID_SORT.out.SMvec,
                feature.merge(nature).filter{it[1] == 'TF'}.combine(Channel.of(['hybrid', 'TF']), by: 1),
                motifs
            )
            ch_versions = ch_versions.mix(SM_EXTRACT_HYBRID.out.versions)

            outSM = SMvec.TSS.concat(TF_HYBRID_SORT.out.SMvec)
            outFreq = freq.TSS.concat(SM_EXTRACT_HYBRID.out.freq)
        } else {
            outSM = SM_SORT.out.SMvec // need to remove cluster SM if TF without hybrid mode ?
            outFreq = SM_EXTRACT.out.freq
        }

        // plotting comparison if required
        // //SMF_SM_SORTING_QC()

    emit:
        smvec = outSM
        freq = outFreq
        versions = ch_versions
}

workflow COOP {
    take:
        feature // channel: tuple(val(featureNat), path(featureFile))
        smvec   // channel: tuple(val(featureNat), val(sortingMode), path(smvec))    <- SMF_SORT.out.smvec
        freq    // channel: tuple(val(featureNat), val(sortingMode), path(freq))     <- SMF_SORT.out.freq

    main:
        ch_versions = Channel.empty()

        // join every sorted (nature, mode) single-molecule vector / frequency matrix with its original feature file
        typed = smvec
            .combine(freq, by: [0, 1])
            .combine(feature, by: 0)
            .map { nat, mode, smvecPath, freqPath, featureFile ->
                tuple(nat, mode, featureFile, freqPath, smvecPath)
            }

        // Build the full combo list deterministically from the collected (nature, mode) records, rather than
        // chaining .filter()/.combine() on the reactive channel: with 3 natures this guarantees exactly
        // homotypic TF-TF/TSS-TSS/DHS-DHS + heterotypic TSS-TF/TSS-DHS.
        combo_ch = typed.toList().flatMap { items ->
            def homotypic = items.collect { nat, mode, featureFile, freqPath, smvecPath ->
                tuple(nat, "${nat}__${mode}", [nat], [featureFile], [mode], [freqPath], [smvecPath])
            }

            def tssItems = items.findAll { it[0] == 'TSS' }
            def tfItems  = items.findAll { it[0] == 'TF'  }
            def dhsItems = items.findAll { it[0] == 'DHS' }

            // heterotypic combo: TSS-TF (only produced if both natures were sorted)
            def tssTF = tssItems.collectMany { t -> tfItems.collect { f ->
                tuple('TSS-TF', "TSS-TF__${t[1]}_${f[1]}", [t[0], f[0]], [t[2], f[2]], [t[1], f[1]], [t[3], f[3]], [t[4], f[4]])
            } }

            // heterotypic combo: TSS-DHS (only produced if both natures were sorted)
            def tssDHS = tssItems.collectMany { t -> dhsItems.collect { d ->
                tuple('TSS-DHS', "TSS-DHS__${t[1]}_${d[1]}", [t[0], d[0]], [t[2], d[2]], [t[1], d[1]], [t[3], d[3]], [t[4], d[4]])
            } }

            homotypic + tssTF + tssDHS
        }

        COOC_MAKEPAIRS(combo_ch)
        ch_versions = ch_versions.mix(COOC_MAKEPAIRS.out.versions)

        COOC_CONTINGENCY(COOC_MAKEPAIRS.out.pairs)
        ch_versions = ch_versions.mix(COOC_CONTINGENCY.out.versions)

        // Re-join each comparison's pairs (carries per-feature freq/strand) with its own contingency table
        pairsForMaster = COOC_MAKEPAIRS.out.pairs.map { comboNat, comboKey, modes, pairs, smvec ->
            tuple(comboNat, comboKey, pairs)
        }
        masterTable_ch = pairsForMaster.combine(COOC_CONTINGENCY.out.contingency, by: [0, 1])

        COOC_MASTERTABLE(masterTable_ch)
        ch_versions = ch_versions.mix(COOC_MASTERTABLE.out.versions)

        COOC_QC(COOC_CONTINGENCY.out.contingency.map { it[2] }.collect())
        ch_versions = ch_versions.mix(COOC_QC.out.versions)

    emit:
        pairs       = COOC_MAKEPAIRS.out.pairs
        contingency = COOC_CONTINGENCY.out.contingency
        masterTable = COOC_MASTERTABLE.out.masterTable
        versions    = ch_versions
}
