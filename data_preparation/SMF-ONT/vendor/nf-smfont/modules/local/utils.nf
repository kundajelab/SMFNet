#!/usr/bin/env nextflow
import groovy.json.JsonBuilder

// Extract the input parameters into a JSON
process getParams {
    label "SMFONT_pipeline"
    cpus 1
    output:
        path "params.json"
    script:
        def paramsJSON = new JsonBuilder(params).toPrettyString()
    """
    # Output nextflow params object to JSON
    echo '$paramsJSON' > params.json
    """
}


