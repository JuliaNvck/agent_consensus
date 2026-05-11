# Review: FedEAT: A Robustness Optimization Framework for Federated LLMs

**Citation:** Pang, Y., Wu, X., Chen, W., Jin, H., & Zhang, X. (2025). *FedEAT: A Robustness Optimization Framework for Federated LLMs.* arXiv:2502.11863. ICML 2025.

---

### Understanding of the Work

Federated LLMs face two simultaneous threat surfaces: adversarial inputs at inference time and malicious client updates at training time. FedEAT addresses each at its natural layer. The client-side component is embedding-space adversarial training (EAT): rather than searching over discrete token sequences (which requires exponential beam search or noisy Gumbel-softmax approximations), each client runs PGD directly on continuous embeddings to generate perturbations subject to a norm constraint, training on a mix of clean and adversarial examples before uploading LoRA deltas to the server. The server-side component is geometric median aggregation via the Weiszfeld algorithm, which minimizes the sum of L2 distances to client updates rather than squared distances, giving a theoretical breakdown point of 1/2 against arbitrary outlier updates. FedEAT reduces Attack Success Rate from 0.259 to 0.252 (PHI-3-MINI) and from 0.176 to 0.144 (ZEPHYR-7B) over FedAVG, with minimal utility loss on clean benchmarks; ablation (Table 5) shows both EAT-only and GM-only generally improve over FedAVG.

---

### Strengths and Weaknesses

**Strengths.** Embedding-space EAT is the paper's most original contribution: continuous embeddings make PGD directly applicable with exact gradients, bypassing the combinatorial search or gradient estimation error required for token-space adversarial training. The ablation (Table 5) gives evidence that both components contribute independently, validating the combination as principled rather than redundant. The threat taxonomy (Table 1) usefully structures the problem by decomposing FL-inherited, LLM-inherited, and joint robustness challenges, which is an organizing contribution for the field.

**Weaknesses.** The most significant flaw is that GM's stated purpose (defending against malicious client updates) is never tested; no Byzantine clients appear in any experiment, leaving GM empirically unvalidated for its primary threat. GEMMA outputs "yes" universally on yes/no tasks, producing ASR=0 trivially, yet is included in the aggregate Average row, distorting the reported averages. Only FedAVG is used as a baseline, leaving FedEAT unpositioned against a mature robust FL literature that includes Krum, trimmed mean, and FLTrust. A single global perturbation norm is applied across all architectures despite documented per-model sensitivity differences acknowledged in Appendix A.1.

---

### Proposed New Ideas for Improvement

**Idea 1: Pre-Aggregation Admission Control Gate.** Before the Weiszfeld step, filter client updates via two signals: L2 norm of each LoRA delta (rejecting updates above the 95th percentile of benign norms from a clean bootstrapping round) and cosine similarity to the previous-round consensus direction (rejecting directionally adversarial updates). Coordinated Byzantine attacks, where all malicious clients push the same poisoned delta, form a tight anomalous cluster in parameter space detectable by both signals before aggregation even runs. This is directly validated in my project's coordinated-attack experiment, where a two-layer pre-aggregation gate reduced the effective Byzantine fraction admitted to the geometric median step. A BFT liveness fallback is essential: if the admitted pool falls below 2f+1 clients, revert to full-pool aggregation and flag the round as low-confidence, preventing the gate from degrading the system under legitimate high-heterogeneity conditions.

**Idea 2: Per-Architecture Adaptive ε.** Run a pre-training calibration sweep over ε values per model architecture on a held-out slice, selecting the ε that maximizes robustness gain minus a weighted utility cost. The PHI-3-MINI/ZEPHYR-7B divergence in results (0.007 vs. 0.032 ASR improvement) is a direct consequence of applying a single ε to models with different embedding geometries; a 7B model and a 2B model have different characteristic embedding norms, so the same Euclidean perturbation means a different fractional distortion for each. Per-architecture calibration adds no communication overhead since ε is a local hyperparameter.

**Idea 3: Run the Missing Byzantine Injection Experiment.** The paper's GM component needs the experiment it never ran: inject malicious clients at varying fractions under two regimes, coordinated (identical poisoned LoRA deltas, forming a tight cluster in parameter space) and uncoordinated (independent random poisons, forming a dispersed distribution). Compare FedAVG, EAT-only, GM-only, and FedEAT across both regimes. GM should dominate under coordinated attack and offer no advantage under uncoordinated attack, yielding a precise, mechanistic scope-of-defense claim grounded in geometric reasoning rather than aggregate ASR averages.
