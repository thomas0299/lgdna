# Geometric Weights Inference
This code aims to infer geometric weights for an unweighted, undirected network, to be run on CPU.

## Disclaimer

This code is provided **exclusively for the purpose of academic peer review** in connection with the submitted manuscript.  
It is **not intended for public use, distribution, or reproduction**.

To comply with double-blind review standards:
- All identifying information (including author names, affiliations, email addresses, and version control history) has been removed.
- Any external references that could compromise anonymity have been omitted or anonymized.

Reviewers are kindly asked to respect the confidentiality and anonymity of this submission, and to refrain from redistributing, citing, or using this material outside the review process.

A full, public version of the code will be made available upon acceptance of the manuscript.

## Setting up the environment

`cd geometric-weights-inference_supp_cpu`

`conda env create -f environment.yml`

OR

`conda env update -f environment.yml --prune`

`conda activate gwi-cpu`


## Example
**geometric_weights_inference(
       graph=graph,
       weightings=["RA2"],
    )**

The graph can either be a `graph_tool.Graph` object or a `scipy.csr_matrix`.
It returns a `Network` object with intermediat results as well as the final weightings stored in `weighting_results` dictionary, keys being the method.


## Weightings

| Name | Formula|
|--------------|-----------|
| RA2_{ij} || (1 + e_{i} + e_{j} + e_{i} * e_{j}) / (1 + CN_{ij}) |
| RA2num_{ij} || 1 + e_{i} + e_{j} + e_{i} * e_{j} |
| CND_{ij} || 1 / (1 + CN_{ij}) |
