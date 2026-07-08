# Lecture 3 — Large Language Models

- **Date watched:** 2026-07-05
- **Lecture:** https://www.youtube.com/watch?v=Q5baLehv5So
- **Slides:** https://cme295.stanford.edu/slides/fall25-cme295-lecture3.pdf
- **Notes taken by:** Elnaz Shafaei

Notes:

## LLM overview

### Terminology
- LLM: Large Langauge Model
  - Language Model := a statistical or machine learning model that assigns probabilities to sequences of tokens.
  - Large
    - Model size: billions of parameters or more
    - Training data: 100s of billions of tokens or more 
    - Compute: a lot of GPUs

## MoE-based LLMs

### Motivation
- In a standard dense Transformer, **every** parameter participates in the forward pass for **every** token. 
  - Cost scales with the total parameter count.
- Key observation: not all weights are useful for a given input — only a subset is. 
- A **Mixture of Experts (MoE)** exploits this by activating only part of the network per token, so we can grow the *total* parameter count (more capacity/knowledge) while keeping the *active* (per-token) compute roughly fixed.

### Setup and notation
Let $x \in \mathbb{R}^{d}$ be the hidden representation of a single token (the input to the MoE layer).

- **Experts:** $E_1, \dots, E_N$, where each $E_i : \mathbb{R}^{d} \to \mathbb{R}^{d}$ is its own sub-network (in an LLM, typically an independent FFN). $N$ = number of experts.
- **Gate (router):** a small learned network $G : \mathbb{R}^{d} \to \mathbb{R}^{N}$ that produces routing weights over the experts. Usually a linear layer followed by a softmax:

$$
G(x) = \mathrm{softmax}(W_g\, x), \qquad W_g \in \mathbb{R}^{N \times d},
$$

so $G(x)_i \ge 0$ and $\sum_{i=1}^{N} G(x)_i = 1$. $G(x)_i$ is the weight (routing probability) assigned to expert $i$ for token $x$.

### Two flavors

**Dense MoE** — every expert runs; the output is the gate-weighted average of all expert outputs:

$$
\hat{y} = \sum_{i=1}^{N} G(x)_i \, E_i(x).
$$

This uses all experts, so there is no compute saving — mainly of theoretical/conceptual interest.

**Sparse MoE** — only the top-$k$ experts (by gate weight) are evaluated. Let

$$
\mathcal{I}_k(x) = \operatorname{arg\,top\text{-}k}_{i} \; G(x)_i
$$

be the set of the $k$ highest-scoring experts for token $x$ ($k \ll N$ is a hyperparameter, e.g. $k=2$). Then

$$
\hat{y} = \sum_{i \in \mathcal{I}_k(x)} \tilde{G}(x)_i \, E_i(x),
$$

where $\tilde{G}(x)_i$ are the gate weights **renormalized over the selected experts** (i.e. softmax restricted to $\mathcal{I}_k(x)$) so they again sum to 1. Because only $k$ of $N$ experts run per token, active compute depends on $k$, not $N$ — this is what makes MoE cheap at scale.

### Where in the Transformer?
- The MoE replaces the **compute-heavy FFN sub-layer** of each Transformer block (the FFN dominates the parameter count in the block).
- Each expert $E_i$ is itself an FFN, so per token:

$$
\hat{y} = \sum_{i \in \mathcal{I}_k(x)} \tilde{G}(x)_i \, \mathrm{FFN}_i(x).
$$

- Routing is **per token**, not per sequence: different tokens in the same sentence can be sent to different experts.

### Training challenge: routing collapse
- **Problem:** the router tends to keep picking the same few experts. Those experts get more gradient updates → get better → get picked even more, a self-reinforcing loop. The remaining experts are starved of tokens, never train, and the effective model capacity collapses to a handful of experts. Load is also badly imbalanced, which hurts hardware utilization.

- **Remedy — auxiliary load-balancing loss:** add a term to the training objective that pushes the router toward spreading tokens *evenly* across all $N$ experts. For a batch $\mathcal{B}$ of $T$ tokens, define, for each expert $i$:

  - $f_i$ = **fraction of tokens routed to expert $i$** (a hard count — it uses the top-$k$ selection):
    $$
    f_i = \frac{1}{T} \sum_{x \in \mathcal{B}} \mathbb{1}\!\left[\, i \in \mathcal{I}_k(x) \,\right].
    $$
  - $P_i$ = **average routing probability assigned to expert $i$** (a soft quantity — it uses the gate weights, so it is differentiable):
    $$
    P_i = \frac{1}{T} \sum_{x \in \mathcal{B}} G(x)_i .
    $$

  The auxiliary loss (Switch-Transformer / GShard style) is:

  $$
  \mathcal{L}_{\text{aux}} = \alpha \, N \sum_{i=1}^{N} f_i \, P_i ,
  $$

  and the total loss is $\mathcal{L} = \mathcal{L}_{\text{LM}} + \mathcal{L}_{\text{aux}}$, where $\mathcal{L}_{\text{LM}}$ is the usual language-modeling loss and $\alpha$ is a small weight (e.g. $\sim 10^{-2}$).

  **Why this works, intuitively:**
  - The product $f_i P_i$ is minimized (subject to $\sum_i f_i = k$ and $\sum_i P_i = 1$) exactly when both the token counts $f_i$ and the probabilities $P_i$ are **uniform** across experts — i.e. balanced routing. Any expert that is over-used has *both* a high $f_i$ and a high $P_i$, so it contributes a large penalty.
  - $f_i$ is not differentiable (it comes from the discrete top-$k$ choice), while $P_i$ is. Multiplying them lets the gradient flow through $P_i$: to lower the penalty on an over-used expert, the router must **decrease the probability mass $P_i$** it sends there, which in turn pushes tokens toward under-used experts.
  - The factor $N$ makes the loss scale-invariant: at perfect balance ($f_i = k/N$, $P_i = 1/N$) the sum is $N \cdot N \cdot \frac{k}{N}\cdot\frac{1}{N} = k$, independent of $N$, so $\alpha$ has a consistent meaning regardless of the number of experts.

- **Related/complementary tricks (worth knowing):** adding noise to the gate logits ("noisy top-$k$") to encourage exploration; per-expert **capacity factors** that cap how many tokens an expert accepts and drop the overflow; and newer *auxiliary-loss-free* balancing (e.g. DeepSeek-V3's learned per-expert bias) that adjusts routing without adding a loss term.

### Reference
- **"Mixtral of Experts"**, Jiang et al., 2024 (Mistral AI). Mixtral 8×7B: $N=8$ experts, top-$k=2$ per token. Their analysis visualized **which expert each token is routed to**, finding routing correlates more with *syntax/token position* than with high-level topic.

## Response generation

### The decoding problem
An autoregressive LLM generates text one token at a time. At step $t{+}1$, given the context (all tokens so far) $C = (w_1, \dots, w_t)$, the model's final layer produces a probability distribution over the vocabulary $V$ via softmax:

$$
P(w_{t+1} = v \mid C) = \frac{\exp\!\big(z_v\big)}{\sum_{u \in V} \exp\!\big(z_u\big)}, \qquad v \in V,
$$

where $z_v$ is the **logit** (pre-softmax score) for vocabulary token $v$. **Decoding** (a.k.a. the search/sampling strategy) is the rule that turns this distribution into an actual chosen token $\hat{w}_{t+1}$. The same distribution can yield very different text depending on this choice. The methods below trade off **quality/likelihood** against **diversity/creativity**.

### Deterministic search

- **Greedy decoding** — pick the single most probable token at each step:

  $$
  \hat{w}_{t+1} = \arg\max_{v \in V} P(w_{t+1} = v \mid C).
  $$

  Fast, but locally optimal $\neq$ globally optimal: maximizing token-by-token does **not** maximize the probability of the whole sequence $P(w_1, \dots, w_T)$. Tends to produce bland, repetitive, unnatural text.

- **Beam search** — keep the $B$ most probable *partial sequences* ("beams") at each step instead of just one ($B$ = beam width). At each step, expand every beam by all possible next tokens, score the resulting sequences by their cumulative log-probability

  $$
  \log P(w_1, \dots, w_{t+1}) = \sum_{s=1}^{t+1} \log P(w_s \mid w_{`\lt s}),
  $$

  and keep the top $B$. Approximates the highest-probability sequence better than greedy, but costs $\sim B\times$ the compute and still favors high-likelihood text, so it **lacks diversity/creativity** (and can still be repetitive). Common for tasks with a "correct" answer (translation, summarization); less so for open-ended generation.

### Stochastic sampling
Instead of always taking the most likely token, **sample** the next token from the distribution:

$$
\hat{w}_{t+1} \sim P(w_{t+1} \mid C).
$$

This injects randomness → more natural and diverse output, at the risk of occasionally picking a low-probability (bad) token. The variants below reshape or truncate the distribution to control that risk:

- **Top-$k$ sampling** — restrict sampling to the $k$ most probable tokens (e.g. $k = 4$), renormalize their probabilities, and sample from that. Cuts off the long low-probability tail. Weakness: $k$ is fixed, so it ignores how peaked or flat the distribution is.

- **Top-$p$ (nucleus) sampling** — restrict to the *smallest* set of tokens whose cumulative probability is $\ge p$ (e.g. $p = 0.9$), then renormalize and sample. The candidate set size adapts to the distribution: small when the model is confident, large when it is uncertain.

- **Temperature $T$** — before the softmax, divide the logits by a temperature $T > 0$:

  $$
  P_T(w_{t+1} = v \mid C) = \frac{\exp(z_v / T)}{\sum_{u \in V} \exp(z_u / T)}.
  $$

  - $T \to 0$: distribution sharpens toward a one-hot → approaches greedy (more deterministic, "safer").
  - $T = 1$: the model's original distribution.
  - $T > 1$: distribution flattens → more diverse/creative, but more errors.

  Temperature is **orthogonal** to top-$k$/top-$p$ and is usually combined with them.

### Guided (constrained) decoding
Force the output to satisfy a structural constraint by masking out any next token that would violate it (set those logits to $-\infty$ before softmax), so only "valid" tokens can be sampled. Used to guarantee outputs match a required format — e.g. valid **JSON**, a regex, or a grammar — which is essential for tool calling and downstream parsing.

## Prompting strategies
**Scope:** the model's weights are fixed — these are inference-time techniques that shape the *input* (the prompt) to get better responses, without any retraining or fine-tuning. Let $C$ denote the prompt/context fed to the model and $y$ the generated answer; every strategy below is a way of constructing $C$.

### Context
- **Terminology:** *context length* = *context size* = *(context) window size* — the maximum number of tokens the model can take in a single forward pass (prompt **plus** generated output must fit inside it).
- For modern LLMs this is on the order of tens/hundreds of thousands, or even millions of tokens (e.g. Gemini, Claude).
- **Bigger context is not automatically better — beware "context rot":**
  - *"Context Rot: How Increasing Input Tokens Impacts LLM Performance"*, Hong et al., 2025.
  - In a **needle-in-a-haystack** test (retrieve one target fact planted in a long input), retrieval accuracy **degrades as the context length grows**, even when the relevant information is fully present.
  - The study controlled for confounds such as **distractors** (plausible-but-wrong passages) to isolate the length effect.
  - **Takeaway:** relevance and density matter more than raw volume — put the important material where it counts and avoid padding the window with low-signal text.
- **Prompt structure** — a useful skeleton for organizing $C$:
  - **Context** (background the model needs) / **Instruction** (the task) / **Input** (the specific data to act on) / **Constraints** (format, length, tone, what to avoid).

### In-context learning (ICL)
The model learns the task *from the prompt itself* at inference time — no weight updates. Distinguished by how many worked examples (input→output pairs, "shots") are included.

- **Zero-shot** — the task/question is posed with **no examples**:
  $$
  C = (\text{instruction},\ \text{input}).
  $$
  Relies entirely on the base model's pretrained ability; performance is highly sensitive to how capable that model already is.
- **Few-shot** — the prompt includes $n$ demonstration pairs before the actual query:
  $$
  C = \big(\underbrace{(x_1, y_1), \dots, (x_n, y_n)}_{n\ \text{demonstrations}},\ x_{\text{query}}\big).
  $$
  ($n = 1$ is "one-shot".) Typically improves performance by showing the model the desired format and behavior.
- **Trade-offs:** including examples generally helps, but:
  - requires effort to write good demonstrations;
  - increases token count → higher compute **cost**;
  - increases **latency** (longer prompt to process).

### Chain of Thought (CoT)
- **Idea:** prompt the model to produce intermediate reasoning steps *before* the final answer (e.g. "let's think step by step") instead of jumping straight to it. Making the reasoning explicit improves performance on multi-step problems (arithmetic, logic, commonsense).
- **Trade-offs:**
  - **+** intermediate steps give interpretability / an explanation of *how* the answer was reached;
  - **−** generating more tokens raises cost and latency.

### Self-consistency
- **Idea:** a single CoT chain can go wrong; instead **sample $M$ independent reasoning paths** (with a nonzero temperature so they differ) and **aggregate by majority vote** over their final answers:
  $$
  \hat{y} = \arg\max_{y} \sum_{m=1}^{M} \mathbb{1}\!\left[\, a_m = y \,\right],
  $$
  where $a_m$ is the final answer of the $m$-th sampled chain. The intuition: different valid reasoning routes tend to converge on the same correct answer, while mistakes are scattered — so the mode is more reliable than any single path.
- **Trade-off:** better accuracy at the cost of running the model $M$ times (roughly $M\times$ the compute/latency).

## Inference optimizations
**Scope:** everything here targets **inference** (serving a trained model), not training. Autoregressive generation has two phases: **prefill** (process the whole prompt in one parallel pass) and **decode** (generate tokens one at a time, each pass depending on all previous tokens). Decoding is the bottleneck — it is sequential and **memory-bandwidth bound** (little compute per step, but the full KV state must be read every step), so most tricks below attack memory traffic or the number of sequential steps.

**Notation used throughout:** sequence length $T$; number of layers $L$; number of attention heads $h$; per-head dimension $d_h$ (so model dim $d = h\,d_h$); bytes per stored value $b$ (e.g. $b=2$ for fp16).

Two broad families:

**Category 1 — exact** (same output distribution, just computed more cheaply):
- avoid redundant recomputation → **KV cache**
- better memory management → **paged attention**
- reformulate the sampling math → **speculative decoding**

**Category 2 — approximate** (change the model/output slightly for speed):
- architectural change → **grouped-query attention (GQA)**
- denser (compressed) representations → **latent attention (MLA)**
- predict several tokens at once → **multi-token prediction (MTP)**

### KV caching  *(exact)*
- **Problem:** in self-attention, generating token $t{+}1$ requires its query to attend over the **keys and values of all previous tokens** $1,\dots,t$. Naively, each new step recomputes $K,V$ for the entire prefix → the same vectors are recomputed again and again, giving $\mathcal{O}(T^2)$ redundant work over a full generation.
- **Idea:** the $K,V$ of past tokens never change, so **cache** them. At each step compute only the new token's query, key, value, append its $k_t,v_t$ to the cache, and attend against the stored $K,V$. This turns per-step attention from recompute-everything into $\mathcal{O}(t)$ read-from-cache.
- **Cost paid instead — memory:** the cache stores $K$ and $V$ for every layer and head:

  $$
  \text{KV-cache size} = 2 \cdot L \cdot h \cdot d_h \cdot T \cdot b \quad \text{bytes per sequence,}
  $$

  (the leading $2$ is for $K$ and $V$), multiplied by the batch size. This grows linearly with sequence length and quickly dominates memory at long context — which motivates every optimization below.

### Paged attention  *(exact — memory management)*
- **Observation:** the KV cache wastes a lot of memory. Because the final output length is unknown in advance, naive serving pre-allocates a **contiguous** max-length buffer per request → large **internal fragmentation** (reserved-but-unused slots), and cached blocks cannot be shared across requests.
- **Idea (vLLM's PagedAttention):** borrow **virtual-memory paging** from operating systems. Split each sequence's KV cache into fixed-size **blocks** and store them in **non-contiguous** physical memory, with a per-sequence **block table** mapping logical block → physical block.
- **Payoff:** near-eliminates fragmentation (waste drops to a few %), lets memory be allocated on demand as generation proceeds, and enables **sharing** identical blocks across sequences (e.g. a common prompt prefix, or parallel samples/beams) via copy-on-write. Higher effective batch size → much better throughput.

### Speculative decoding  *(exact — reformulate the math)*
- **Idea:** decoding is slow because it is sequential and each big-model step is memory-bound. Use a small, cheap **draft** model to *propose* several tokens, then have the large **target** model **verify** them all in a **single parallel** forward pass. Good proposals let us advance several tokens per target pass; the accept/reject rule guarantees the output is **distributed exactly as if sampled from the target model** — so there is *no* quality loss.
- **Notation:** let $q(\cdot)$ be the draft model's next-token distribution and $p(\cdot)$ the target model's. *(The lecture writes $Q$ for the target and $P$ for the draft; below we follow the more common convention $q=$ draft, $p=$ target.)* For each token $x$ the draft proposes ($x \sim q$), verify it as follows:

  $$
  \text{accept } x \text{ with probability } \min\!\left(1,\ \frac{p(x)}{q(x)}\right).
  $$

  Concretely:
  - **If $p(x) \ge q(x)$** — the target likes $x$ at least as much as the draft did → **accept** (probability $1$).
  - **Otherwise** — accept with probability $p(x)/q(x)$, and **reject** with probability $1 - p(x)/q(x)$.
  - **On the first rejection:** discard the rest of the drafted tokens and **resample** one token from the normalized residual distribution
    $$
    p_{\text{res}}(x) = \frac{[\,p(x) - q(x)\,]_+}{\sum_{x'} [\,p(x') - q(x')\,]_+}, \qquad [z]_+ \equiv \max(0, z),
    $$
    then continue from there.
- **Why it's exact:** the accept step keeps tokens where the draft over- or correctly-estimated $p$; the residual $[p-q]_+$ exactly "tops up" the probability mass the draft *under*-sampled. Together they reconstruct $p$ token-for-token (Leviathan et al., 2023; Chen et al., 2023).
- **Speedup:** roughly proportional to how many tokens the draft gets accepted per target pass — best when the draft closely mimics the target on easy/predictable spans.

### Grouped-query attention (GQA)  *(approximate — architecture)*
- **Recall the bottleneck:** the KV cache size scales with the number of **key/value** heads. Query heads don't sit in the cache; K/V heads do.
- **Vanilla multi-head attention (MHA):** $\#\text{query} = \#\text{key} = \#\text{value} = h$ heads.
- **Grouped-query attention:** keep $h$ **query** heads but let them **share** a smaller number of key/value heads — partition the $h$ query heads into $G$ groups ($1 \le G \le h$), one shared $K/V$ head per group, so $\#\text{key} = \#\text{value} = G$.

  $$
  \text{KV-cache size} = 2 \cdot L \cdot G \cdot d_h \cdot T \cdot b,
  \qquad \text{a factor } \tfrac{h}{G}\ \text{smaller than MHA.}
  $$

  - $G = h$ recovers full MHA; $G = 1$ is **multi-query attention (MQA)**, the most aggressive (single shared K/V head).
  - Trade-off: fewer distinct K/V heads is a mild approximation (small quality cost) in exchange for a large cache reduction and faster decoding.

### Latent (compressed) attention — MLA  *(approximate — denser representation)*
- **Goal:** shrink what the KV cache stores per token.
- **Idea (DeepSeek's Multi-head Latent Attention):** instead of caching full-dimensional $K$ and $V$, project them **down** into a small shared **latent vector** $c_t \in \mathbb{R}^{d_c}$ with $d_c \ll h\,d_h$, and cache only $c_t$. During attention, $K$ and $V$ are reconstructed on the fly by **up-projecting** $c_t$. Only the compressed $c_t$ lives in memory, so the cache shrinks dramatically while retaining more information than GQA's head-sharing (reported to match or beat MHA quality).

### Multi-token prediction (MTP)  *(approximate — faster token prediction)*
- **Standard training** predicts only the **next** token. **MTP** adds $k$ prediction heads on top of the shared trunk, trained to predict the next $k$ tokens $(w_{t+1}, \dots, w_{t+k})$ at once.
- **Two benefits:**
  1. **Denser training signal** — each position supervises $k$ future tokens, which tends to improve the base model.
  2. **Faster inference** — the extra heads can propose the next few tokens, giving **self-speculative decoding** where the **draft and target are the same model** (no separate draft model needed); the main model then verifies them in parallel (as in speculative decoding above). Used in DeepSeek-V3.