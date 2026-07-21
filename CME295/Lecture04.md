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
- What do you need to keep in memory?
  - parameters (from initialization onward)
  - activations (during the forward pass; kept for reuse in the backward pass)
    - grow with batch size and sequence length
  - gradients (during the backward pass)
  - optimizer states (during the weight update)
    - e.g., Adam keeps 2 extra values per parameter (momentum and variance)
- How much GPU memory do you have?
  - on the order of 10s of GBs
  - e.g., H100 SXM has 80GB, H100 NVL has 94GB
- *(not in course)* rule of thumb: mixed-precision training with Adam needs $\approx 16$ bytes per parameter
  - FP16 weights (2) + FP16 gradients (2) + FP32 master weights (4) + momentum (4) + variance (4)
  - plus activations on top
  - e.g., a 7B model needs $\approx 112$ GB of training state; it does not fit on one 80GB H100
- *(not in course)* activation checkpointing: store only a subset of activations
  - recompute the rest on the fly during the backward pass
  - trades extra compute ($\approx 30\%$) for large memory savings

### Data parallelism (DP)
- Idea.
  - divide each batch of data across devices
  - model replicated on each device
- Cost.
  - the full model must still fit on every single device
  - after the forward and backward passes, gradients are aggregated across GPUs (all-reduce)
  - this aggregation incurs communication cost and slows down training
- DP with ZeRO (Zero Redundancy Optimizer, Rajbhandari et al., 2020)
  - goal:
    - decrease the memory load on each device
    - cut redundant copies of the same information across devices
    - makes it easier to fit the model on a device
  - idea: partition the memory buffers across devices; gather pieces only when needed
  - variants (each one partitions one more thing):
    - ZeRO-1: partition optimizer states
    - ZeRO-2: partition optimizer states + gradients
    - ZeRO-3: partition optimizer states + gradients + parameters
  - disadvantage: even more communication cost, hence slower training
  - *(not in course)* PyTorch's FSDP (Fully Sharded Data Parallel) is essentially ZeRO-3

### Model Parallelism (MP)

- Idea.
  - split the model computations across several devices
  - parallelize the computations, even within one batch
- Variations.
  - Tensor Parallelism (TP)
    - split individual weight matrices across devices (e.g., attention heads, FFN columns/rows)
    - needs a very fast interconnect (NVLink), so usually stays within one node
  - Pipeline Parallelism (PP)
    - put different (groups of) layers on different devices
    - batches are cut into micro-batches to keep all stages busy (reduce the "pipeline bubble")
  - Sequence Parallelism (SP)
    - shard activations along the sequence dimension
    - covers the ops TP leaves replicated (LayerNorm, dropout)
  - Context Parallelism (CP)
    - shard very long sequences across devices for attention (e.g., Ring Attention)
  - Expert Parallelism (EP)
    - put different experts (of an MoE) on different devices
- *(not in course)* frontier training runs combine several of these (DP + TP + PP + CP/EP)
  - so-called 3D/4D parallelism

### Flash Attention
- "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", Dao et al., 2022
- GPU structure
  - CU (Compute Units): where the math happens (SMs, streaming multiprocessors, in NVIDIA terms)
  - HBM (High-Bandwidth Memory): large and slow memory
    - the advertised "GPU memory", e.g., 80GB at $\sim 3$ TB/s on an H100
  - SRAM: small and fast on-chip memory, next to the CUs
    - $\sim 10\times$ the bandwidth of HBM, but only 10s of MBs in total
  - takeaway: attention is **memory-bound**; moving data to/from HBM dominates runtime, not the math

- "Standard" attention computation, with every intermediate materialized in HBM:
  - LOAD $Q, K$ from HBM by blocks *(mem op)*
  - compute $S = QK^\top$
  - WRITE $S$ to HBM *(mem op)*
  - READ $S$ from HBM *(mem op)*
  - compute $P = \mathrm{softmax}(S)$
  - WRITE $P$ to HBM *(mem op)*
  - LOAD $P, V$ from HBM by blocks *(mem op)*
  - compute $O = PV$
  - WRITE $O$ to HBM *(mem op)*
  - problem: $S$ and $P$ are $n \times n$
    - reading/writing them scales quadratically with sequence length

- Flash attention ideas:
  - (ONE) minimize reads/writes to HBM with **tiling**: do the computation block by block in SRAM
    - Trick. no need to compute the full $S = QK^\top$ before applying softmax
      - softmax can be computed blockwise and rescaled:

        $$\mathrm{softmax}(S) = \mathrm{softmax}\big([S_1; S_2; \dots; S_B]\big) = \big[\alpha_1\,\mathrm{softmax}(S_1);\ \alpha_2\,\mathrm{softmax}(S_2);\ \dots;\ \alpha_B\,\mathrm{softmax}(S_B)\big]$$

      - the rescaling factors $\alpha_i$ come from running max/sum statistics kept in SRAM ("online softmax")
    - iteratively:
      - READ blocks of $Q, K, V$ from HBM to SRAM *(mem op)*
      - compute a block of $O$, reading from and writing to SRAM only
      - WRITE the result to HBM to accumulate results *(mem op)*
    - the full $n \times n$ matrices $S$ and $P$ are never materialized in HBM
  - (TWO) sometimes it is better to **recompute** instead of storing
    - normally, activations are stored during the forward pass for use in the backward pass
    - instead: free them, and recompute them blockwise during the backward pass
    - we are memory-bound and computation is very fast
    - net effect: more operations, less memory, and *faster*, since there are fewer HBM reads/writes
- Result: **exact** attention (no approximation), 2-4$\times$ faster
  - memory footprint **linear** in sequence length instead of quadratic
- *(not in course)* FlashAttention-2 (2023) and FlashAttention-3 (2024) improve work partitioning
  - and exploit newer GPU features (e.g., H100 asynchrony and FP8)

### Mixed precision training

Representation of a float:
- sign: controls whether the number is positive or negative; 1 bit
- exponent: controls the magnitude of the number; also called range
- mantissa: controls the granularity of the number, i.e., what is after the decimal point
  - also called significand or fraction

Acronyms:
- FP: Floating Point
- BF: Brain Float (bfloat, developed at Google Brain)

|      | sign | exponent | mantissa |
|------|------|----------|----------|
| FP16 |    1 |        5 |       10 |
| FP32 |    1 |        8 |       23 |
| FP64 |    1 |       11 |       52 |
| BF16 |    1 |        8 |        7 |

- *(not in course)* BF16 keeps FP32's exponent (same dynamic range) and gives up mantissa (precision)
  - avoids FP16's overflow/underflow issues; the modern default for LLM training

On a GPU:
- lower precision $\leftrightarrow$ less memory, faster processing
- technical specs guide of GPUs gives you FLOP/s for each data type:

|                      | H100 SXM        | H100 NVL        |
|----------------------|-----------------|-----------------|
| FP64                 | 34 teraFLOPS    | 30 teraFLOPS    |
| FP64 Tensor Core     | 67 teraFLOPS    | 60 teraFLOPS    |
| FP32                 | 67 teraFLOPS    | 60 teraFLOPS    |
| TF32 Tensor Core     | 989 teraFLOPS   | 835 teraFLOPS   |
| BFLOAT16 Tensor Core | 1,979 teraFLOPS | 1,671 teraFLOPS |
| FP16 Tensor Core     | 1,979 teraFLOPS | 1,671 teraFLOPS |
| FP8 Tensor Core      | 3,958 teraFLOPS | 3,341 teraFLOPS |
| INT8 Tensor Core     | 3,958 TOPS      | 3,341 TOPS      |

- *(not in course)* the Tensor Core figures above assume 2:4 structured sparsity; dense throughput is half

Mixed precision training ("Mixed Precision Training", Micikevicius et al., 2018):
- Objective. speed up training and decrease memory requirements without hurting performance too much
  - Forward pass. activations in low precision (FP16)
  - Backward pass. gradient updates in low precision (FP16)
  - Weights update. keep a master copy of the weights in high precision (FP32)
- *(not in course)* loss scaling: with FP16, scale the loss up before backprop, unscale before the update
  - keeps small gradients from underflowing to zero; unnecessary with BF16
- *(not in course)* FP8 training is emerging on H100-class GPUs (roughly 2$\times$ FP16 throughput)

Quantization methods (mapping floats to a lower-precision grid; mainly used for inference):
- zero-point: asymmetric (affine) mapping, with a scale and an offset (the "zero-point")
  - uses the full integer range even when values are not centered at 0
- absmax: symmetric mapping, scale by the maximum absolute value
  - e.g., for INT8: $x \mapsto \mathrm{round}\!\left(\frac{127}{\max\lvert x \rvert}\, x\right)$

## Supervised finetuning (SFT)

### Motivation: pretrained model behavior
- A **pretrained (base) model** only knows how to predict the next token.
  - it is not (yet) a helpful assistant
- Example. prompt: *"Can I put my teddy bear in the washer?"*
  - a base model does not answer; it continues the text with something plausible
    - e.g., *"Teddy bears are often made of materials like polyester and cotton, ..."*
    - or it might just ask another question
  - it mimics likely continuations, not helpful responses
- **Remedy:** a second stage of **finetuning** that adapts the pretrained weights to a specific behavior.

### SFT
- **SFT = Supervised FineTuning**
  - *supervised:* we need labels, i.e., pairs of (input, output)
  - *finetuning:* we refine already-pretrained weights rather than starting from scratch
- **Idea.** Change the model behavior by tuning its weights.
- **Strategy.**
  - collect pairs of input/outputs with the desired behavior (aka SFT data)
  - train using the next-token prediction objective, *given the input*
- **Objective function.** same NTP objective as pretraining, but the loss is computed differently.
  - the **input (prompt) is fixed:** we do not want the model to parrot it
    - no teacher forcing on the input, so no loss over the input tokens
  - the loss starts at the first **output (completion)** token and continues onward:

    ```
     out:                          Sure    ...
                                    ↑        ↑
          ┌──────────────────── LLM ────────────────────┐
     in:  [BOS]    Do      X       .       Sure
    ```

    - i.e., condition on the input, only fit the distribution of the output tokens

### Instruction tuning
- **Special case** of SFT: SFT on instruction-following data.
  - "Finetuned Language Models Are Zero-Shot Learners", Wei et al., 2022 *(FLAN)*
- **Goal.** "graduate" the model from a good language model to a **helpful assistant**.
- **Data mixtures.** can be both human-written and synthetic; categories include:
  - assistant dialogs
  - task examples: story writing, poem creation, list generation, explanation, ...
  - math, reasoning, code
  - **safety alignment**
    - teach the model to reject harmful queries (rather than hard-coded regex, which does not scale)
    - **hedging:** nuance answers instead of making blanket statements
  - each example is (instruction, ground-truth answer); loss is fit on the answer
- **Data generation.**
  - originally almost all **human-written** (expert linguists following answer-writing guidelines)
  - increasingly **synthetic:** a strong LLM generates candidate answers, then a human or another LLM reviews quality
    - speeds up dataset curation
- **Generalization.** the model generalizes beyond the exact examples seen at SFT time.
  - it leans on knowledge acquired during pretraining
    - e.g., "write a story about poetry" teaches the *concept* of story writing; pretraining supplies the specifics
- **Size.** thousands to millions of examples.

  | Model   | SFT size (# examples) |
  |---------|-----------------------|
  | GPT-3   | 13 thousand           |
  | Llama 3 | 10 million            |

  - reported in **# examples**, not tokens (unlike pretraining)
  - at ~1k tokens/example, this is still several **orders of magnitude smaller** than pretraining
  - mental model: pretraining = lots of data for general language; SFT = little, very high-quality data to align behavior to the target task
- **Behavior after instruction tuning.** same teddy-bear prompt now gets a helpful answer.
  - e.g., *"No, it might get damaged. Try hand washing instead."*

### Challenges
- **High-quality data needed.**
  - often requires humans in the loop; expensive in time and resources
  - upside: SFT datasets are reusable across model versions (curate once, extend over time)
- **Sensitive to prompt distribution.**
  - performance depends on how close the SFT prompt distribution is to the inference distribution
  - out-of-distribution prompts (e.g., a story style not seen in SFT) may not generalize well
    - fix: add more data covering that region of the distribution
- **Generalization.** data-mixture coverage matters more than repetition.
  - points spread across the distribution give the model the *gist* to generalize
  - repeating the same example again and again does not help
- **Difficult to evaluate** (see below).
- **Computationally expensive** (motivates parameter-efficient finetuning, next section).

### Evaluation
- **Benchmarks.** decompose "quality" into measurable dimensions:
  - general knowledge: **MMLU** *(Massive Multitask Language Understanding, ~57 tasks)*
  - basic reasoning: **ARC-Challenge**
  - math reasoning: **GSM8K** *(Grade School Math, ~8.5k problems)*
  - code generation: **HumanEval**
  - *(not in course)* benchmarks proliferate because models optimize for existing ones, so new ones fill the gaps
- **Validity: "training on the test task".**
  - "Training on the Test Task Confounds Evaluation and Emergence", Dominguez-Olmedo et al., 2024
    - the **test task**, not the test *set* (this is not data leakage)
  - to fairly compare models, ensure **parity** on whether each was trained on the target task
    - e.g., both trained on math-style data, or neither
    - benchmarks often ship *auxiliary training sets* for this purpose
  - sudden benchmark spikes across model versions often trace back to this, not to intrinsic ability
- **"Real-life" feeling: Chatbot Arena / LMArena.**
  - website where users compare two anonymous model responses (A/B test) and vote for the better one
  - pairwise votes are aggregated into a ranking *(not in course)* via an Elo / Bradley-Terry rating
  - **benefit:** puts a number on "vibes" (subjective helpfulness)
  - **outstanding challenges:**
    - **cold start:** a new model's first few matchups are noisy but strongly influence its ranking
    - **easy to rig:** models are identifiable (e.g., "who are you?" → "I'm ChatGPT"), so an adversary can game votes
      - "Exploring and Mitigating Adversarial Manipulation of Voting-Based Leaderboards", Huang et al., 2025
    - **factuality:** users can't reliably judge correctness (a fluent, actionable answer can still be wrong)
    - **personal preference bias:** voters are not representative of the wider user population
      - e.g., emojis: popular broadly, disliked by many domain experts
    - **safety penalization:** users dislike refusals, biasing votes against intended safety behavior
  - takeaway: **evaluation is a hard problem**; no single number captures it, tailor to your use case

### Alignment
- **Alignment** = the post-pretraining stages that make the model do what you want.
  - **finetuning + preference tuning** (preference tuning covered in Lecture 5)
- *(not in course)* **Mid-training:** an emerging stage between pretraining and finetuning.
  - same NTP objective as pretraining, but on data/tasks closer to what you care about

## Parameter-efficient finetuning

### LoRA
- **Context.** SFT is resource intensive and not everyone has big GPUs.
- **Idea.** Low-Rank Adaptation (LoRA): approximate the weight update with a product of two **low-rank** matrices.
  - "LoRA: Low-Rank Adaptation of Large Language Models", Hu et al., 2021
  - decompose the finetuned weights as:

    $$W = W_0 + BA$$

    - $W_0 \in \mathbb{R}^{d \times k}$: pretrained weights, **frozen**
    - $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$: the only trainable matrices
    - $r$ is the **rank**, taken very small (e.g., up to ~10), while $d, k$ are hundreds to thousands
  - since $r \ll \min(d, k)$, there are far fewer parameters to train, at similar performance
- **Forward pass.** run both terms and add them:

  $$h = W_0 x + B(A x)$$

  - $W_0$ never changes; only $A, B$ receive gradients
- **Benefit: swap matrices = swap tasks.**
  - start from one base model $W_0$, learn task-specific $(A, B)$ per task
    - e.g., $(A_\text{spam}, B_\text{spam})$ for spam detection, $(A_\text{sentiment}, B_\text{sentiment})$ for sentiment, etc.
  - swap the small adapters at will instead of storing a full finetuned model per task
- **Where to apply LoRA.**
  - originally (Hu et al., 2021): on the **attention** weight matrices
  - updated guidance: the **feed-forward blocks** are the most impactful location
    - "LoRA Without Regret", Schulman et al., 2025
  - today both usually carry LoRA matrices, but the bulk of the gain is in the feed-forward block
- **Training dynamics** *(empirical, not fully explained)*.
  - LoRA needs a **higher learning rate** than full finetuning (~10x)
    - intuition: the small rank restricts the space, so larger steps are needed
  - LoRA does **poorly on large batch sizes** compared to full finetuning
    - intuition: the training dynamics of a product of matrices differ from a full matrix
- ***(not in course)*** the rank $r$ is a design choice.
  - grid-search it, or just pick a common value (e.g., 4); the parameter reduction is already so large that shrinking further matters little
- **Other methods.** prefix tuning and adapters (in the class textbook; less commonly used).

### QLoRA
- **Idea.** Quantize all frozen weights to relieve the memory bottleneck.
  - "QLoRA: Efficient Finetuning of Quantized LLMs", Dettmers et al., 2023
  - $W_0$ is **stored quantized**; the trainable $A, B$ are kept in **full precision** (BF16)
  - **computations are done in full precision** (dequantize $W_0$ on the fly)
- **Efficient quantization: NF4 (4-bit NormalFloat).**
  - assumes weights are **normally distributed**, so it splits the range by **normal quantiles**
    - rather than uniform (fixed-width) cutoffs like INT8
  - result: roughly equal number of values per bucket, so the 4 bits are used efficiently
- **Double quantization.** quantize the quantization constants too.
  - converting weights in/out of quantized form generates **quantization constants**
  - QLoRA quantizes those constants as well (FP32 $\to$ FP8), for extra savings
- **Benefits.**
  - VRAM savings enable finetuning on smaller GPUs, and faster
  - better memory/quality trade-off
- **Orders of magnitude** (reported on Llama 65B):
  - ~**16x** VRAM savings during finetuning
  - the double-quantization trick saves an extra ~6%
