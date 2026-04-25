This repository is associated with the following paper and should be cited accordingly:

> **Latent Geometry-Driven Network Automata for Complex Network Dismantling**  
> Thomas Adler, Marco Grassia, Ziheng Liao, Giuseppe Mangioni, Carlo Vittorio Cannistraci  
> *The Fourteenth International Conference on Learning Representations (ICLR), 2026*  
> https://openreview.net/forum?id=yz29QCGVzC

```bibtex
@inproceedings{
adler2026latent,
title={Latent Geometry-Driven Network Automata for Complex Network Dismantling},
author={Thomas Adler and Marco Grassia and Ziheng Liao and Giuseppe Mangioni and Carlo Vittorio Cannistraci},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=yz29QCGVzC}
}
```

# Latent Geometry Driven Network Automata

## Usage
There are two versions of the code, one for CPU and one for GPU.

For CPU code, follow the instructions in `RADismantling_supp_cpu`, available methods are:  RA2_sum, RA2num_sum, CND_sum, betweenness_centrality, domirank, pagerank, degree, fitness_centrality, resilience_centrality, eigenvector_centrality.

For GPU code, `RADismantling_supp_gpu`, available methods are: RA2_sum, RA2num_sum, CND_sum, betweenness_centrality

For each CPU and GPU version a separate environment should be created with their correct dependencies.

## Engineering Network Robustness

The `create_robust_networks.ipynb` is a simple code to engineer a more robust network as detailed in the main text and save it. One can then dismantle that network as the other to compare the change in dismantling performance.

## Results

Below are the results for all available methods for the `covert_FIFA2015` network, run on CPU.



Method|AUC|Removals|
|--------------|-----------|-----------|
RA2|0.04303|31|
RA2_num|0.04502|33|
CND|0.04298|33|
NBC|0.03885|32|
Domirank-0.5|0.0485|38|
PR|0.04697|38|
Degree|0.04809|36|
Fitness|0.04746|38|
Resilience|0.05|39|
Eigenvector|0.0953|94|
RA2-R1|0.0427|30|
RA2_num-R1|0.04441|31|
NBC-R1|0.04049|30|
CND-R1|0.04216|30|
Degree-R1|0.04495|32|
Domirank-0.5-R1|0.04456|32|
PR-R1|0.04358|32|
Fitness-R1|0.04469|32|
Resilience-R1|0.04587|32|
Eigenvector-R1|0.05393|32|
RA2-R2|0.0427|30|
RA2_num-R2|0.04441|31|
NBC-R2|0.04049|30|
CND-R2|0.04218|30|
Degree-R2|0.04495|32|
Domirank-0.5-R2|0.04456|32|
Fitness-R2|0.04469|32|
PR-R2|0.04358|32|
Resilience-R2|0.04587|32|
Eigenvector-R2|0.05419|33|
RA2-R3|0.0427|30|
RA2_num-R3|0.04441|31|
NBC-R3|0.04049|30|
CND-R3|0.04216|30|
Degree-R3|0.04495|32|
Domirank-0.5-R3|0.04461|32|
Fitness-R3|0.04469|32|
PR-R3|0.04358|32|
Resilience-R3|0.04587|32|
Eigenvector-R3|0.05452|33|