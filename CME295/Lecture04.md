# Lecture 4 — LLM training

- **Date watched:** 2026-07-08
- **Lecture:** https://www.youtube.com/watch?v=VlA_jt_3Qc4
- **Slides:** https://cme295.stanford.edu/slides/fall25-cme295-lecture4.pdf
- **Notes taken by:** Elnaz Shafaei

Notes:

## Pretraining

- **Traditional approach:** train a model from scratch, specifically for the task at hand.
- **Transfer learning:** instead of starting from scratch every time, start from a model pretrained on a general objective
  - then adapt it to your task (finetuning, prompting)
  - pretraining is done **once** (expensive); adaptation is cheap

### Notation

- **FLOPs (FLoating-point OPerations):** a unit of *compute* (total amount of work).
  - Training FLOPs depend on model size, number of training tokens, and architecture.
  - Standard rule of thumb for Transformers, with $N$ = #parameters and $D$ = #training tokens:

    $$C \approx 6\,N\,D$$

    - $\approx 2ND$ from the forward pass, $\approx 4ND$ from the backward pass
  - Orders of magnitude (total training compute):
    - small NN $\sim 10^{7}$
    - large RNN $\sim 10^{14}$
    - frontier LLM $\sim 10^{25}$
- **FLOP/s or FLOPS (FLoating-point OPerations per Second):** a unit of *compute speed*
  - measures how fast the hardware executes those operations
  - Orders of magnitude:
    - smartphone $\sim 10^{12}$
    - computer $\sim 10^{14}$
    - single H100 GPU $\sim 10^{15}$ (FP16/BF16)
    - supercomputer $\sim 10^{18}$
  - *(not in course)* Real training runs reach only a fraction of peak FLOP/s
    - an **MFU** (Model FLOPs Utilization) of 30–50% is considered good

### LLM pretraining
- **Goal:** learn general patterns of language and code.
- **Objective:** next-token prediction (NTP), i.e., minimize cross-entropy over the corpus:

  $$\mathcal{L}(\theta) = -\sum_{t} \log p_\theta\!\left(x_t \mid x_{\lt t}\right)$$

- **Data:**
  - web scrapes (e.g., Common Crawl, Wikipedia), code (e.g., GitHub, Stack Overflow)
  - size: hundreds of billions to trillions of tokens (GPT-3: 300B, Llama 3: 15T)
  - *(not in course)* raw scale is not enough; heavy **curation** matters
    - quality filtering, deduplication, and mixing ratios across sources (e.g., up-weighting code and books)
    - data is typically seen for ~1 epoch; repeating it has fast-diminishing returns

### Scaling laws
- **Kaplan et al., 2020, "Scaling Laws for Neural Language Models"**
  - Test loss falls as a smooth **power law** in each of compute $C$, dataset size $D$, and model size $N$
    - straight lines on log-log plots, with no plateau over many orders of magnitude
    - more compute / more data / bigger model ⇒ predictably lower loss
  - **Sample efficiency:** for a fixed number of tokens processed, a bigger model reaches lower loss than a smaller one
    - i.e., large models learn more per token
- **Chinchilla law (Hoffmann et al., 2022, "Training Compute-Optimal Large Language Models")**
  - For a **fixed compute budget**, training loss as a function of model size is **U-shaped** (IsoFLOP curves)
    - too small ⇒ not enough capacity; too big ⇒ too few tokens seen
    - e.g., at $10^{20}$ FLOPs the optimum is $\approx 1\text{B}$ parameters
  - Tracking the optima across compute budgets (linear in log-log space):
    - optimal $N^{*} \propto C^{0.5}$ and optimal $D^{*} \propto C^{0.5}$: scale model and data **equally**
    - rule of thumb: $D^{*} \approx 20\,N^{*}$, i.e., about **20 training tokens per parameter**
  - Fitted loss decomposition:

    $$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}, \qquad E \approx 1.69,\; \alpha \approx 0.34,\; \beta \approx 0.28$$

    where $E$ is the irreducible entropy of natural text.
  - *(not in course)* This **corrected Kaplan et al.**, who had recommended growing $N$ much faster than $D$ ($N \propto C^{0.73}$)
    - proof point: **Chinchilla** (70B params, 1.4T tokens) beat **Gopher** (280B params, 300B tokens) at the same compute
- **Case study: GPT-3** (175B parameters, 300B tokens)
  - Chinchilla-optimal data for 175B parameters: $175\text{B} \times 20 = 3.5\text{T}$ tokens.
  - Actually trained on 300B tokens ⇒ severely **undertrained** for its size
    - equivalently: oversized for its compute budget
- ***(not in course)* Beyond Chinchilla: inference-aware scaling**
  - Chinchilla optimizes *training* compute only, but a deployed model is run many times
  - So it often pays to **overtrain a smaller model** far past 20 tokens/param
    - slightly worse training efficiency, much cheaper inference
  - E.g., Llama 3 8B was trained on 15T tokens
    - $\approx$ 1,875 tokens per parameter, roughly 100× the Chinchilla ratio

### Challenges
- **Cost**
  - money: millions to hundreds of millions of dollars per frontier run
  - time: weeks to months on thousands of GPUs
    - hardware failures are a routine occurrence at that scale
  - electricity and environmental impact
- **Learned knowledge**
  - knowledge cutoff: the model's knowledge is frozen at training time
  - hard to edit: injecting or correcting facts without regressing elsewhere is an open problem
  - memorization: plagiarism / verbatim regeneration of training data (copyright, privacy concerns)
  - *(not in course)* benchmark contamination: eval sets leak into web-scale training data and inflate reported performance

## Training optimizations 

### Memory 
- What thing do you need to keep in memory
  - parameters (during initialization)
  - activations (during forward pass)
  - gradients (during backward pass)
  - optimizer states (during weights update)
- How much GPU memory do you have?
  - order of 10s of GBs
  - e.g., H100 SXM has 80GB, H100 NVL has 94GB
    
### Data parallelism (DP)
- Idea. 
  - Divide batch of data across devices 
  - Model replicated on each device
- DP with ZeRO (Zero Redundancy Optimization)
  - Redundant information across devices
  - ZeRO-1: share optimizer states
  - ZeRO-2: share optimizer states + gradients
  - ZeRO-3: share optimizer states + gradients + parameters

### Model Parallelism (MP)

- Idea. Split the model computations across several devices
- Variations.
  - Tensor Parallelism (TP)
  - Pipeline Parallelism (PP)
  - Sequence Parallelism (SP)
  - Context Parallelism (CP)
  - Expert Parallelism (EP)

### Flash Attention
- GPU structure
  - HBM
  - CU
  - SRAM
  
### Mixed precision training
Technical specs guide of GPUs gives you FLOPS for each data type

Objective. Speed up training and decrease memory requirements
- Forward pass. Activations in low precision 
  - input: FP32
  - activations: FP16
- Backwards pass. Gradient updates in low precision (FP16)
- Weights update. Keep weights in high precision (FP32)

## Supervised finetuning

## Parameter-efficient finetuning
