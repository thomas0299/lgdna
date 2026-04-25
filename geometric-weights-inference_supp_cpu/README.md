### Geometric Weights Inference

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
