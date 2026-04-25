# Repulsion Attraction Dismantling
This code aims to dismantle networks using the Repulsion Attraction rule and other centrality measures, using CPU.

## Disclaimer

This code is provided **exclusively for the purpose of academic peer review** in connection with the submitted manuscript.  
It is **not intended for public use, distribution, or reproduction**.

To comply with double-blind review standards:
- All identifying information (including author names, affiliations, email addresses, and version control history) has been removed.
- Any external references that could compromise anonymity have been omitted or anonymized.

Reviewers are kindly asked to respect the confidentiality and anonymity of this submission, and to refrain from redistributing, citing, or using this material outside the review process.

A full, public version of the code will be made available upon acceptance of the manuscript.


## Setting up the environment

`cd RAdismantling_supp_cpu`

`conda env create -f environment.yml`

OR

`conda env update -f environment.yml --prune`

`conda activate ra-dismantling-cpu`


You also have to install the Geometric Weights Inference package:
```bash
pip install -e ../geometric-weights-inference_supp_cpu
```

and this package:
```bash
pip install -e ../RADismantling_supp_cpu
```

## Datasets

A representative network `covert_FIFA2015` has been provided to test the code

## Usage

### Dismantling

To run the dismantling algorithm framework, you can use the following `dismantler.py` script.

```bash
usage: dismantler.py [-h] [-o OUTPUT] -l LOCATION [-t THRESHOLD] [-H [ ...]] [-HF HEURISTICS_FILE]
                     [-SD] [-DD] [-R] [-RT {R1,R2,R3}]
                     [-Ft TEST_FILTER] [-j JOBS]
                     [-mnv MAX_NUM_VERTICES] [-miv MIN_NUM_VERTICES]
                     [-mne MAX_NUM_EDGES] [-mie MIN_NUM_EDGES]
                     [-v {INFO,DEBUG,WARNING,ERROR}]

options:
  -h, --help            Show this help message and exit.
  -o OUTPUT, --output OUTPUT
                        Output DataFrame file location to store the results.
                        Default: ./out/df/heuristics.csv
  -l LOCATION, --location LOCATION
                        Path to the dataset (directory).
  -t THRESHOLD, --threshold THRESHOLD
                        Dismantling target threshold. Fraction of the network size.
                        Default: 0.1
  -H [...], --heuristics [ ...]
                        Space-separated list of heuristics to test.
                        Allowed values: ['RA2_sum', 'RA2num_sum', 'CND_sum', 'betweenness_centrality', 'domirank', 'pagerank', 'degree', 'fitness_centrality', 'resilience_centrality', ',eigenvector_centrality'].
  -HF HEURISTICS_FILE, --heuristics_file HEURISTICS_FILE
                        Path to file containing heuristics names (space or newline separated).
  -SD, --static_dismantling
                        Static computation of heuristics (no recomputation).
  -DD, --dynamic_dismantling
                        Enables recomputation of heuristics after each removal.
  -R, --reinsertion     Performs reinsertion at the end of the attack.
  -RT {R1,R2,R3}, --reinsertion_type {R1,CI,R3}
                        Reinsertion type to use at the end of the attack.
  -Ft TEST_FILTER, --test_filter TEST_FILTER
                        Filter test folders (default: '*').
  -j JOBS, --jobs JOBS  Number of jobs to run in parallel.
  -mnv MAX_NUM_VERTICES, --max_num_vertices MAX_NUM_VERTICES
                        Filter networks by maximum number of vertices.
  -miv MIN_NUM_VERTICES, --min_num_vertices MIN_NUM_VERTICES
                        Filter networks by minimum number of vertices.
  -mne MAX_NUM_EDGES, --max_num_edges MAX_NUM_EDGES
                        Filter networks by maximum number of edges.
  -mie MIN_NUM_EDGES, --min_num_edges MIN_NUM_EDGES
                        Filter networks by minimum number of edges.
  -v {INFO,DEBUG,WARNING,ERROR}, --verbose {INFO,DEBUG,WARNING,ERROR}
                        Set verbosity level (case insensitive).

```

For example:

```bash
python ra_dismantling/dismantler.py \
       -l ../dataset/ATLAS/ \
       -Ft "covert_FIFA2015" \
       -H  RA2_sum eigenvector_centrality \
       -DD \
       --output out/df/ra2.csv \
       -j 1
```

You can also run dismantling on all the networks from the covert subfield using `-Ft "covert*"`


### Reinsertion

There are three reinsertion techniques, you need to rerun the dismantling command with the R and RT commands specified, with the output file the original file where the dismantling results were stored. For example:
```bash
python ra_dismantling/dismantler.py \
       -l ../dataset/ATLAS/ \
       -Ft "covert_FIFA2015" \
       -H  RA2_sum eigenvector_centrality \
       -DD \
       -R \
       -RT R1 \
       --output out/df/ra2.csv \
       -j 1
```

### Output

All output files will be stored in a DataFrame with the following columns:
- `network`: the network name
- `heuristic`: the heuristic name
- `slcc_peak_at`: the size of the SLCC at the peak of the second largest component (used to capture the percolation of the network)
- `lcc_size_at_peak`: the size of the LCC at the peak of the second largest component
- `slcc_size_at_peak`: the size of the SLCC at the peak of the second largest component
- `removals`: the list of removal steps (information about the node removed and the size of the LCC and SLCC)
- `static`: whether the heuristic is static or not
- `r_auc`: the area under the dismantling curve
- `rem_num`: the number of removals
- `r_auc_n`: the area under the dismantling curve, normalised for the size (number of nodes) in the network
- `time`: time, in nanoseconds, taken to run this dismantling
- `date`: timestamp of dismantling
- `threshold`: the threshold used in the dismantling
- `reinsertion`: whether a reinsertion step was performed at the end of the attack
- `reinsertion_type`: reinsertion method
- `reinserted_nodes`: nodes that were reinserted in the reinsertion phase


CAREFUL: When choosing the resinsertion step argument, the list of removals and "reinserted nodes" will be the original indices for the reinserted_method, so uncomparable to the the removals with no reinsertion step. There is a function ``recover_original_indices`. to convert the removals from non-reinsertion runs to their original indices.