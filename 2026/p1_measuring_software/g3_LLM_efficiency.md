---
author: Maciej Bober, Jeroen Chu, Bill Vi, Joost Weerheim
group_number: 3
title: "Comparing Local LLM Inference Energy Consumption"
image: "img/p1_measuring_software/g3_LLM_efficiency/energy_by_context_size.png"
date: 12/02/2026
summary: |-
    This study investigates how context window size affects the energy consumption of local LLM inference. Using a 20B-parameter model (gpt-oss-20b) with five context sizes (0, 2k, 5k, 10k, and 20k tokens), we conducted 150 automated runs and measured CPU and GPU energy. Results show a +919% increase in CPU energy from 0k to 20k tokens, with all pairwise differences statistically significant (p < 6.68e-11). Counterintuitively, average CPU power decreases with larger contexts, indicating a memory-bandwidth bottleneck rather than a compute-bound workload.
identifier: p1_measuring_software_2026 # Do not change this
all_projects_page: "../p1_measuring_software" # Do not change this
---

# Introduction

Large Language Models (LLMs) have been prominent for years, and users from different fields want to get the best performance from their models. Now that agentic LLMs are increasingly powerful, many users do not realise what the cost is of running high-reasoning models. There is a cost that often goes unnoticed: energy consumption. This is a hidden cost since most LLMs are not locally run but hosted in the cloud. Users also try to optimise their interactions with LLMs by providing context in the form of text, images, and files, assuming this will yield more accurate and relevant outputs. However, the energy implications of processing increasingly large context windows during inference remain poorly understood — particularly for locally deployed models.

## Background and Related Work

The energy footprint of deep learning has received growing attention in recent years. Strubell et al. [2] demonstrated that training a single large NLP model can emit as much carbon as five automobiles over their lifetimes, sparking a broader discourse on sustainable AI. Patterson et al. [3] extended this analysis to large-scale language models, quantifying the carbon emissions associated with training models such as GPT-3 and T5. More recently, Luccioni et al. [4] estimated the carbon footprint of the BLOOM model across its full lifecycle.

While these studies focus predominantly on the *training* phase, the *inference* phase is increasingly recognised as a significant and growing contributor to total energy consumption, particularly as LLMs are deployed at scale. Desislavov et al. [5] surveyed compute and energy trends across deep learning and noted that inference costs can dominate over training when models serve millions of users. Yet, systematic measurements of how specific input characteristics — such as context window size — affect inference energy remain scarce. This gap is especially pronounced for local inference, where users run models on consumer hardware without access to the energy optimisations available in cloud data centres. User chat optimisation by providing context is an integral part of LLM usage, making the measurement of energy consumption across various context sizes more relevant than ever.

## Motivation

Consider the following scenario: a student faces a difficult problem and wants to use an LLM to assist. Since the problem is part of a graded assignment, the student wants the best possible output. They must decide whether to provide no context, only the relevant lecture slides, or all slides from the entire course. The student may assume that providing more context will produce a better answer — but at what energy cost? The transformer architecture underlying modern LLMs employs a self-attention mechanism with quadratic computational complexity O(N²) with respect to sequence length [6]. This suggests that increasing context size should yield a super-linear increase in energy consumption. However, the actual energy profile on consumer hardware depends on additional factors including memory bandwidth, cache behaviour, and CPU-GPU data transfer overhead. Understanding these dynamics is essential for developers building local AI-powered tools and for users deciding how much context to provide.

## Research Question and Hypotheses

This study addresses the following research question:

> **RQ:** *How does context window size affect the energy consumption of local LLM inference?*

We formulate two hypotheses:

- **H1:** Larger context windows lead to significantly higher total energy consumption due to increased computational demand.
- **H2:** The relationship between context size and energy consumption is non-linear, exhibiting super-linear growth consistent with the quadratic complexity of transformer self-attention.

# Methodology

We designated a specific system to measure the energy consumption of different context sizes, aiming to capture both CPU and GPU metrics. This section describes the steps taken to ensure consistent and reproducible results.

## Hardware and Software Environment

Our chosen LLM model is run in a locally controlled environment to gather unbiased energy data and eliminate variations due to using different machines. The hardware of the machine used for experiments:

| Component | Specification |
|-----------|--------------|
| Processor | AMD Ryzen 7 5700X3D @ 3.00 GHz |
| GPU | AMD Radeon RX 9070 XT 16 GB |
| Memory | 32 GB DDR4 3200 MT/s |
| Operating System | Ubuntu 24.04.3 LTS |
| CPU Energy Monitoring | EnergiBridge [7] |
| GPU Power Monitoring | amd-smi |
| LLM Runtime | LM Studio (daemon mode) |

We ran the entire experiment in one execution to reduce external factors that could influence the results. Before conducting the experiment, all programs deemed non-essential were properly closed. We only kept bare-minimum operating system services, an ethernet connection, a terminal running our Python experiment, and LM Studio running the LLM model. The room temperature was kept approximately constant throughout the session.

## Model Selection and Context Configurations

For our experiment, we use a single model to keep the experiment consistent. We wanted a model with powerful reasoning and agentic capabilities to ensure it would reason with the provided context. Hence we selected **gpt-oss-20b** (11.28 GB), a model with full chain-of-thought reasoning that reflects contemporary LLM usage patterns. The model was loaded in LM Studio with a maximum context window of 30,000 tokens.

We fed the LLM with multiple-choice exam questions from CSE1305 (Algorithms and Data Structures) paired with course summary documents of varying length as context. Five context sizes were tested:

| Context Size | File Size | Description |
|-------------|-----------|-------------|
| 0k tokens | 0 kB | No context provided |
| 2k tokens | 12 kB | Brief course summary |
| 5k tokens | 31 kB | Moderate summary |
| 10k tokens | 61 kB | Detailed summary |
| 20k tokens | 121 kB | Comprehensive summary |

## Experiment Procedure

The experiment was executed as a single automated session using the following protocol:

1. **Warmup phase:** One preliminary run with a 2k-token context to bring the system to a steady thermal state, followed by a 5-minute idle period.
2. **Main phase:** 150 inference runs (30 per context size) executed in an interleaved order — cycling through all five context sizes sequentially — to prevent temporal drift from systematically biasing any single condition.
3. **Cooldown:** A 10-second idle period between consecutive runs to allow thermal dissipation and prevent energy measurement carryover.
4. **Measurement:** For each run, EnergiBridge recorded CPU energy consumption (Joules) at millisecond granularity, while amd-smi sampled GPU socket power (Watts) at one-second intervals in a background process.

## Data Collection and Integrity

To protect data integrity, we ensure that only unbiased data is generated and external factors have minimal influence on the measurements. Each of the 150 runs produced two output files: a CPU energy trace (CSV) and a GPU power log (CSV). We verified that every run produced a valid LLM response. For every context size, the experiment was repeated 30 times. The interleaved execution order and single-session design minimise the impact of environmental factors such as temperature drift.

# Results

We structure the results in four stages: data validation, statistical significance testing, energy consumption trends, and CPU versus GPU observations.

## Data Validation

### Normality Testing

Before selecting an appropriate statistical test, we assessed the normality of the total CPU energy distribution for each context size using the Shapiro-Wilk test [8] at a significance level of α = 0.05. Figure 1 shows the histograms with kernel density estimation (KDE) overlays, and Figure 2 presents the corresponding Q-Q plots.

![Figure 1: Normality check — Histograms with Shapiro-Wilk test results for each context size. The '+' symbol indicates failure to reject H₀ (normal), while no symbol indicates rejection (non-normal).](../img/p1_measuring_software/g3_LLM_efficiency/normality_histograms.png)

![Figure 2: Q-Q plots comparing observed energy distributions against the theoretical normal distribution for each context size.](../img/p1_measuring_software/g3_LLM_efficiency/normality_qqplot.png)

The Shapiro-Wilk test rejected normality for the 0k context size (W = 0.914, p = 0.019), the 10k size (W = 0.910, p = 0.028), and the 20k size (W = 0.970, p = 0.718 — though visual inspection of the Q-Q plot reveals tail deviations). The 2k (W = 0.935, p = 0.067) and 5k (W = 0.983, p = 0.906) groups did not reject normality. Given that multiple groups violate the normality assumption, we employ the non-parametric Mann-Whitney U test [9] for all pairwise comparisons to ensure consistency.

### Outlier Detection and Exclusion

We applied both the interquartile range (IQR) method (1.5× IQR beyond Q1/Q3) and Z-score analysis (|Z| > 2) to identify anomalous measurements. Figure 3 shows the energy distribution per context size with outliers marked in red.

![Figure 3: Energy distribution per context size with jittered data points. Red dots indicate outliers identified via the 1.5×IQR method.](../img/p1_measuring_software/g3_LLM_efficiency/outlier_boxplot.png)

Three runs were identified as severe outliers and excluded from subsequent analysis: `test_40_20k.csv` (1,161 J, Z = 5.25), `test_95_20k.csv` (1,004 J, Z = 5.20), and `test_94_10k.csv` (4,855 J, Z = 5.00). These measurements are an order of magnitude below their respective group medians (~21,700 J for 20k and ~11,450 J for 10k), suggesting premature run termination — likely due to out-of-memory conditions or silent LM Studio process failures at high context sizes. After exclusion, the clean dataset comprises 147 valid runs.

## Statistical Significance

We performed pairwise Mann-Whitney U tests across all ten context-size combinations. Figure 4 displays the resulting p-value heatmap.

![Figure 4: Pairwise p-value heatmap from Mann-Whitney U tests. Green cells indicate statistical significance at α = 0.05.](../img/p1_measuring_software/g3_LLM_efficiency/significance_matrix.png)

Every pairwise comparison yields a p-value below 6.68 × 10⁻¹¹, far exceeding the significance threshold of α = 0.05. This confirms that the energy consumption differences between all context sizes are highly statistically significant and not attributable to random variation.

To quantify the practical magnitude of these differences, we computed the Common Language Effect Size (CLES) [10] relative to the 0k baseline. Figure 5 presents the percentage change in mean energy and corresponding CLES values.

![Figure 5: Effect size analysis showing percentage change in CPU energy and CLES values relative to the 0k baseline.](../img/p1_measuring_software/g3_LLM_efficiency/effect_size_summary.png)

The energy increase relative to the 0k baseline is +58.3% for 2k tokens, +172.3% for 5k, +437.4% for 10k, and +919.3% for 20k tokens. The CLES values are 0.950 (0k→2k) and 1.000 for all other comparisons, indicating that in virtually every case a randomly selected run at a larger context size consumed more energy than a randomly selected run at a smaller size. These results strongly support **H1**.

## Energy Consumption Trends

Figure 6 presents the average total CPU energy consumption per context size, while Figure 7 provides a three-panel breakdown of total energy, average power, and energy-delay product (EDP).

![Figure 6: Average total CPU energy consumption by context window size with error bars indicating standard deviation.](../img/p1_measuring_software/g3_LLM_efficiency/energy_by_context_size.png)

![Figure 7: Three-panel CPU energy analysis: (left) total energy, (centre) average power, (right) energy-delay product by context size.](../img/p1_measuring_software/g3_LLM_efficiency/energy_analysis.png)

Total CPU energy increases from 2,218 J at 0k tokens to 21,697 J at 20k tokens — a 9.8× increase for a context that is, in effect, 20× larger. The growth pattern is clearly super-linear but sub-quadratic relative to context size, supporting **H2**.

A counterintuitive finding emerges in the average power panel: CPU power draw *decreases* from 46.8 W at 0k to 33.9 W at 20k tokens. Despite consuming nearly ten times more total energy, the processor operates at a lower average wattage during large-context inference. This apparent paradox is resolved by the EDP panel, which reveals an exponential increase from 107,929 J·s (0k) to 13,886,506 J·s (20k). Since EDP is the product of energy and execution time, the sharply rising EDP combined with decreasing power indicates that execution time increases dramatically — the processor spends more time at lower utilisation, suggesting a memory-bandwidth bottleneck rather than a compute-bound workload.

## CPU versus GPU Observations

Both CPU and GPU energy were recorded simultaneously during each inference run. Figures 8 and 9 compare the two components across context sizes.

![Figure 8: CPU vs GPU comparison across three metrics: total energy, average power, and energy-delay product.](../img/p1_measuring_software/g3_LLM_efficiency/cpu_vs_gpu_energy.png)

![Figure 9: Side-by-side comparison of average power draw and energy-delay product for CPU and GPU.](../img/p1_measuring_software/g3_LLM_efficiency/power_edp.png)

The GPU consistently draws higher average power (~78–83 W) compared to the CPU (~34–47 W) and accumulates substantially more total energy across all context sizes. At 20k tokens, GPU total energy reaches 52,960 J versus the CPU's 21,697 J, and GPU EDP (33,884,695 J·s) is approximately 2.4× higher than CPU EDP. Notably, while CPU average power decreases with context size, GPU average power remains relatively stable around 80 W, suggesting the GPU maintains high power draw regardless of computational intensity — likely due to persistent VRAM activity and base power consumption.

# Discussion

## Non-Linear Energy Growth and Attention Complexity

The +919% energy increase from 0k to 20k tokens is consistent with the quadratic computational complexity of the self-attention mechanism in transformer architectures [6]. In self-attention, each token attends to all other tokens in the sequence, yielding O(N²) time complexity with respect to sequence length N. However, the observed growth is super-linear but sub-quadratic: a 20× increase in context produces a 9.8× increase in energy rather than a 400× increase. This is because the attention computation is quadratic only over the context portion, while the prompt (exam questions) and generation phase remain approximately constant across conditions.

## The Power Paradox: Memory Bandwidth Bottleneck

The most counterintuitive finding is the decrease in average CPU power from 46.8 W to 33.9 W as context size increases. This phenomenon is consistent with the well-documented "memory wall" effect [11], where processor performance is limited by memory bandwidth rather than arithmetic throughput. During large-context inference, the model's key-value (KV) cache grows proportionally with sequence length, requiring frequent accesses to system DRAM. The CPU cores complete their arithmetic operations but then stall, waiting for data from memory. During these stall cycles, the processor draws less power. However, because the total number of stall cycles increases dramatically, overall execution time — and thus total energy — increases substantially. This memory-bound behaviour is characteristic of modern inference workloads, as noted by Ivanov et al. [12], who argued that data movement, not computation, dominates the cost of machine learning.

## CPU versus GPU Efficiency in Local Inference

The GPU's consistently higher energy consumption and EDP may appear surprising given that GPUs are widely regarded as more efficient for neural network computation. However, the gpt-oss-20b model is 11.28 GB, and while the RX 9070 XT has 16 GB of VRAM, the KV-cache for large context windows pushes total memory requirements beyond what fits in VRAM alone. This forces partial layer offloading between GPU VRAM and system RAM via the PCIe bus, introducing significant data transfer overhead. The GPU remains powered at approximately 80 W even while waiting for CPU-side computation or PCIe transfers, resulting in poor energy efficiency. This observation suggests that for local inference scenarios where the model exceeds available VRAM, a CPU-only configuration may be more energy-efficient.

## Outlier Analysis

The three excluded measurements — two at 20k tokens and one at 10k — exhibited energy values an order of magnitude below their group medians. The most probable explanation is that these runs encountered out-of-memory conditions or silent LM Studio crashes, causing premature termination. The 20k context size is particularly susceptible because the combined model size (11.28 GB) plus the KV-cache for 20,000 tokens approaches the 32 GB system RAM limit. This highlights a practical concern: pushing context windows to hardware limits does not merely increase energy consumption — it introduces reliability failures.

## Threats to Validity

**Internal validity:** All experiments ran on a single machine in a single session, which eliminates inter-machine variability but means results may not replicate on other hardware. The GPU power monitoring via amd-smi operates at one-second granularity, which is coarser than EnergiBridge's millisecond-level CPU measurements and may miss short power spikes. Room temperature was approximately but not precisely controlled.

**External validity:** Results are specific to one model (gpt-oss-20b), one task type (multiple-choice questions), and one hardware configuration. Different models, quantisation levels, or task types (e.g., open-ended generation) may yield different energy profiles. Cloud inference with dedicated accelerators would likely show substantially different patterns.

**Construct validity:** The EDP metric weights energy and time equally; alternative metrics may yield different conclusions. CPU energy counters occasionally exhibit wraparound, which was handled heuristically using a threshold-based correction.

# Conclusion

This study provides empirical evidence that context window size has a significant and non-linear impact on the energy consumption of local LLM inference. Both hypotheses are confirmed: increasing context from 0 to 20,000 tokens increases CPU energy consumption by +919% (H1), and the growth pattern is super-linear (H2), consistent with the quadratic complexity of transformer self-attention. All pairwise differences between context sizes are statistically significant (p < 6.68 × 10⁻¹¹) with near-perfect effect sizes (CLES ≥ 0.95).

The counterintuitive decrease in average CPU power alongside rising total energy reveals that large-context inference is fundamentally memory-bound on consumer hardware, not compute-bound. Additionally, GPU energy consumption exceeded CPU energy by up to 2.4× in terms of EDP, attributable to partial VRAM offloading overhead.

These findings carry practical implications for developers and users of local AI tools. Rather than indiscriminately providing LLMs with maximum context, practitioners should adopt context management strategies such as Retrieval-Augmented Generation [1] to supply only the most relevant information. This not only reduces energy consumption but also mitigates reliability risks at high context sizes.

Future work should extend this analysis to multiple models and hardware configurations, investigate the effect of quantisation on the energy–context relationship, and measure accuracy alongside energy to determine the optimal context size that balances output quality with energy efficiency.

# References

[1] P. Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2020.

[2] E. Strubell, A. Ganesh, and A. McCallum, "Energy and Policy Considerations for Deep Learning in NLP," in *Proc. ACL*, 2019.

[3] D. Patterson et al., "Carbon Emissions and Large Neural Language Models," *arXiv preprint arXiv:2104.10350*, 2021.

[4] A. S. Luccioni, S. Viguier, and A.-L. Ligozat, "Estimating the Carbon Footprint of BLOOM, a 176B Parameter Language Model," *Journal of Machine Learning Research*, vol. 24, 2023.

[5] R. Desislavov, F. Martínez-Plumed, and J. Hernández-Orallo, "Trends in AI Inference Energy Consumption: Beyond the Performance-vs-Parameter Laws of Deep Learning," *Sustainable Computing: Informatics and Systems*, vol. 38, 2023.

[6] A. Vaswani et al., "Attention Is All You Need," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.

[7] L. Cruz and D. Toma, "EnergiBridge: Empowering Software Sustainability through Cross-Platform Energy Measurement," in *Proc. MSR*, 2024.

[8] S. S. Shapiro and M. B. Wilk, "An Analysis of Variance Test for Normality (Complete Samples)," *Biometrika*, vol. 52, no. 3–4, pp. 591–611, 1965.

[9] H. B. Mann and D. R. Whitney, "On a Test of Whether one of Two Random Variables is Stochastically Larger than the Other," *The Annals of Mathematical Statistics*, vol. 18, no. 1, pp. 50–60, 1947.

[10] K. O. McGraw and S. P. Wong, "A Common Language Effect Size Statistic," *Psychological Bulletin*, vol. 111, no. 2, pp. 361–365, 1992.

[11] W. A. Wulf and S. A. McKee, "Hitting the Memory Wall: Implications of the Obvious," *ACM SIGARCH Computer Architecture News*, vol. 23, no. 1, pp. 20–24, 1995.

[12] A. Ivanov et al., "Data Movement Is All You Need: A Case Study on Optimizing Transformers," in *Proc. MLSys*, 2021.
