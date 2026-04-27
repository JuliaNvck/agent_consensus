# Mathematical Foundations: Multi-Agent LLM Consensus

This document defines the mathematical formulas and algorithmic objectives required for the pipeline. Do not deviate from these definitions when implementing the logic.

## Module 1: Reliability Filtering

### Top-K Probability Mass (TopKMass)
The core generation-process signal used for admission control. It calculates the average probability mass concentrated in the top-5 tokens over a sliding window of $W$ tokens.

$$TopKMass(W) = \frac{1}{W} \sum_{w \in W} \sum_{i \in top-5} P(t_i | context)$$

* **$W$**: The sliding generation window size (constant: 64 tokens).
* **$w \in W$**: Each token step within the sliding window.
* **$P(t_i | context)$**: The discrete probability of the $i$-th ranked token in the model's next-token distribution.

## Module 2: Robust Semantic Aggregation

### Geometric Median Objective (Stage 1)
Calculated over the sentence embeddings of the admitted agents' outputs. Unlike the arithmetic mean, the geometric median minimizes the sum of Euclidean distances to all points, bounding adversarial centroid shift to $O(f/N)$.

$$y^* = \text{argmin}_{y} \sum_{j=1}^{M} ||x_j - y||_2$$

* **$M$**: The total number of agents that passed the Module 1 filter (where $M \le N$).
* **$x_j$**: The embedding vector of the $j$-th admitted agent.
* **$y$**: The candidate centroid vector.
* **Implementation Requirement**: This objective must be minimized iteratively using the Weiszfeld algorithm via `scipy.optimize`.