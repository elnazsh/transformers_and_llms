# Lecture 2 — Transformer-Based Models & Tricks

- **Date watched:** 2026-05-22
- **Lecture:** https://www.youtube.com/watch?v=yT84Y5zCnaA
- **Slides:** https://cme295.stanford.edu/slides/fall25-cme295-lecture2.pdf
- **Notes taken by:** Elnaz Shafaei

## Notes

### 1. Position embeddings
- **Motivation:**
  - self-attention by itself is permutation-invariant
  - meaning it doesn’t know whether a token came first or last.
  - positional embeddings inject sequence order information
- different approaches:
  - learned positional embeddings _vs_ hardcoded/static positional embeddings
  - absolute positional information _vs_ relative positional information

#### 1.1 Absolute positional embeddings

- We have a unique positional embedding for each position in the sequence.
- The positional embedding is added to the input.

##### 1.1.1 learned (absolute) positional embeddings
- how it works:
  - model has a trainable vector for each position $p_i$
  - and input becomes $x_i + p_i$ where $p_i$ is optimized during training
- pros:
  - flexible: the model can discover whatever positional structure helps the task; no inductive bias is forced.
  - well-suited to domains where positional behavior differs from natural language, e.g. code, proteins, music.
  - very expressive (in training sequence regimes) as every position gets its own degrees of freedom.
- cons:
  - no length generalization: positions beyond the max seen in training have no embedding at all. Extending the context requires retraining (or hacks like interpolation).
  - parameter cost scales linearly with max sequence length ($\text{max_seq_len} \times d$ extra params), which becomes painful for long contexts.
  - long-tail positions are seen rarely during training, so their embeddings stay under-trained and noisy.
  - prone to overfitting: nothing prevents the model from latching onto idiosyncratic per-position patterns from the training distribution.
  - no built-in geometric structure: there is no guarantee that
    - adjacent positions get similar embeddings,
    - or that the embedding of position $m+k$ relates to position $m$ in any consistent way.

##### 1.1.2. hardcoded (absolute) positional embeddings: sinusoidal embeddings
Main idea: tokens that are closer in position should have positional embeddings that are closer in vector space (i.e. higher dot product), and farther positions should have less similar embeddings.
- sinusoidal embeddings were proposed in the original attention paper, kinda obsolete now
- encode positions as vectors, add these vectors to input token vectors
- therefore, $d$ = positional embeddings dimension = input tokens dimension = model dimension 

For position $m$ in the sequence, and vector dimension index $i$:
$$e_{m,2i} = \sin(w_i \cdot m)$$
$$e_{m,2i+1} = \cos(w_i \cdot m)$$
where $w_i = 10000^{-2i/d}$

---
**A deep dive** (small deviation from the slides):

**Lemma.** Closer sinusoidal positional embeddings are more similar (in dot product), locally around $m=n$.

**Reminder.** $\cos(a-b) = \cos(a)\cos(b) + \sin(a)\sin(b)$

**Proof.** Let the embedding dimension be $d$ (even) with frequencies $w_i = 10000^{-2i/d}$ for $i = 0, \dots, d/2 - 1$. Then

$$
e_m \cdot e_n = \sum_i \big(e_{m,2i}\,e_{n,2i} + e_{m,2i+1}\,e_{n,2i+1}\big)
= \sum_i \big(\sin(w_i m)\sin(w_i n) + \cos(w_i m)\cos(w_i n)\big)
= \sum_i \cos\!\big(w_i m - w_i n\big)
= \sum_i \cos\!\big(w_i (m - n)\big).
$$

Define $f(k) := \sum_i \cos(w_i k)$, so that $e_m \cdot e_n = f(m - n)$. Two observations:

1. **Evenness.** Since $\cos$ is even, $f(-k) = f(k)$, so the similarity depends only on $|m-n|$.

2. **Global maximum at $k = 0$.** For any $k$,
$$
f(k) = \sum_i \cos(w_i k) \le \sum_i 1 = \tfrac{d}{2} = f(0),
$$
with equality iff every $w_i k \in 2\pi\mathbb{Z}$. Because the $w_i$ form a geometric sequence with irrational ratio relative to $2\pi$, this only holds at $k = 0$. 
Hence, $e_m \cdot e_n$ is uniquely maximized when $m = n$.

3. **Local monotonic decrease.** Taylor expanding around $k=0$,
$$
f(k) = \sum_i \Big(1 - \tfrac{(w_i k)^2}{2} + O((w_i k)^4)\Big)
= \tfrac{d}{2} - \tfrac{k^2}{2}\sum_i w_i^2 + O(k^4).
$$
The leading correction is $-\tfrac{k^2}{2} S$ with $S = \sum_i w_i^2 > 0$, so $f$ is strictly decreasing in $|k|$ on a neighborhood of $0$. Equivalently, $\tfrac{d}{dk}f(k) = -\sum_i w_i \sin(w_i k)$, which is negative for small $k > 0$ since every term $w_i \sin(w_i k) > 0$ when $0 < w_i k < \pi$ — and this holds simultaneously for all $i$ whenever $0 < k < \pi / w_0 = \pi$ (the largest $w_i = w_0 = 1$).

**Combining (2) and (3):** the dot product is maximized at $m=n$, and for $|m-n|$ in the local regime where all $w_i |m-n| < \pi$, the dot product is a strictly decreasing function of $|m-n|$. $\blacksquare$

**Remark.** Outside this local regime, $f$ oscillates and the "closer ⇒ more similar" statement fails pointwise — different frequency components dephase, and one can find $k_1 < k_2$ with $f(k_1) < f(k_2)$. The lemma is therefore best read as a statement about the immediate neighborhood of each position, which is the regime relevant to attention's local-bias behavior.

See this figure for the dot product of embeddings for all position pairs https://kazemnejad.com/img/transformer_architecture_positional_encoding/time-steps_dot_product.png

---

Sinusoidal embeddings (contd.)
- pros:
  - works at any sequence length; the formula is defined for arbitrary $m$, so no retraining is needed to handle longer inputs (in principle).
  - zero learnable parameters for the positional encoding; saves memory and removes a source of overfitting.
  - deterministic and reproducible; the same position always produces the same vector.
  - has a useful algebraic property: $e_{m+k}$ can be written as a fixed linear transform of $e_m$. This means relative offsets are *representable* by the model, and it directly motivates RoPE later.
  - "closer ⇒ more similar" is provable locally (see the lemma above), giving a sensible inductive bias.
- cons:
  - "closer ⇒ more similar" only holds in the local neighborhood of each position. Outside that regime $f(k)$ oscillates and the property fails pointwise.
  - **(for absolute embeddings in general)** added to the input, so
    - the position signal has to survive being mixed with the token embedding
    - as it gets propagated through every layer, it can get washed out in deeper networks.
  - in practice, length extrapolation is weaker than the formula suggests: attention weights *trained* on short sequences don't behave well at unseen positions, even though the encoding itself is defined there. Specifically:
    - **the encoding is defined everywhere, but the model isn't.** The formula gives a valid vector $e_m$ for any $m$, but the downstream weights ($W_Q, W_K, W_V$, LayerNorm, FFN) were only optimized on inputs of the form $x_i + e_{m_i}$ for $m_i$ in the training range. At larger $m$, the model is being asked to read a distribution of inputs it has never seen.
    - **softmax dilution.** Even ignoring the encoding, a softmax over many more keys behaves differently from a softmax over few — the denominator is larger and attention spreads thinner. Weights were calibrated for the short-sequence regime, so effective focus shifts as context grows.
    - **no training signal at those positions.** Gradient never flowed through "what should I do at position 4999," so any behavior there is accidental rather than learned.
    - **empirical track record.** The original Transformer paper claimed extrapolation as a benefit of sinusoidal embeddings, but Press et al. (ALiBi, 2021) showed clearly that vanilla sinusoidal Transformers degrade quickly past their training length. This motivated ALiBi and, later, RoPE + length-extension techniques (position interpolation, NTK-aware scaling, YaRN), all of which try to make the *downstream weights* behave sensibly at unseen positions — not just define an encoding there.
    - **slogan:** an encoding that is defined at long positions is *necessary* for length generalization, but nowhere near *sufficient*. The trained weights have to play along too.

#### 1.2 Relative positional embeddings
- What matters in language is usually relative position (e.g., positions 2 tokens apart or **distance**), not absolute position (e.g., position 10 and position 12): 
  - "the cat sat" should mean the same thing whether it starts at token 0 or token 1000. 
  - with sinusoidal embeddings the model has to learn this invariance from data: wasteful and imperfect.
  - encoding relative position directly bakes the right inductive bias into the architecture.
  - so we cannot simply add the position embeddings to input vector, and instead we need to modify the attention mechanism.
- Length extrapolation didn't actually work. 
  - the original Transformer paper claimed sinusoidal would extrapolate to longer sequences.
  - empirically it doesn't: performance collapses past the training length. Relative schemes give you a real chance at length generalization. 
- Adding to the input is structurally bad. 
  - the position signal is added once at the bottom of the stack, then has to survive being mixed with token content through every layer (attention, FFN, residual).
  - it tends to get washed out in deeper models. 
  - better to inject position where it's used, i.e., inside the attention score, where the model is actually deciding _who attends to whom_.
- proposed methods:
  - relative positional embeddings (RPE): used in T5
  - ALiBi (Attention with Linear Biases):
  - rotary positional embeddings (RoPE): dominant in frontier LLMs nowadays

##### 1.2.1. relative positional information via attention bias
- we can do this by adding a _bias_ to the dot product of the query and key vectors,
- this bias value should be large for tokens closer to the query, and small for tokens further away from the query.

So we can change the formula from 
$$\text{softmax}\left({q_i k_j \over \sqrt{d_k}}\right)$$ to
$$\text{softmax}\left({q_i k_j \over \sqrt{d_k}} + \text{bias}(i,j)\right) $$

Different implementations of the bias term:

**Relative positional bias (T5 paper):**

bias is learned per head, as a function of the relative distance $i-j$.
$$
\text{bias}(i,j) = \beta_{\text{bucket}}(i-j)
$$
- Instead of learning one scalar per *exact* distance (which would blow up for long sequences), T5 discretizes the integer distance $i-j$ into a fixed, small set of bucket indices.
- Each bucket has one learned scalar per head. So the entire positional-bias parameter set is a scalar table of shape $[\text{num_heads}, \text{num_buckets}]$, where in T5 defaults $\text{num_buckets} = 32$.
  - The bias table is *shared across all attention layers* within the encoder (and a separate one within the decoder). 
  - T5 also has *no* input-side positional embedding; this relative bias is the *only* source of position information in the model.
  - So the entire model adds only $2 \times \text{num_heads} \times \text{num_buckets}$ positional parameters in total (extremely cheap).
  - Independent of sequence length.
$$B = \begin{pmatrix}
b_{0}  & b_{1}  & b_{2}  & b_{3}  & b_{4} \\
b_{-1} & b_{0}  & b_{1}  & b_{2}  & b_{3} \\
b_{-2} & b_{-1} & b_{0}  & b_{1}  & b_{2} \\
b_{-3} & b_{-2} & b_{-1} & b_{0}  & b_{1} \\
b_{-4} & b_{-3} & b_{-2} & b_{-1} & b_{0}
\end{pmatrix}$$
- The bucketing scheme:
  1. Exact buckets for small distances. For $|i-j| < \text{max_exact}$ (T5 default: 16), each distance gets its own bucket. This gives fine-grained control where it matters most (nearby tokens).
  2. Log-spaced buckets for large distances. For $|i-j| \ge \text{max_exact}$, distances are grouped logarithmically up to some $\text{max_distance}$ (T5 default: 128). The bucket index grows like $\log(|i-j|)$, so the further out you go, the coarser the bucketing. Everything beyond $\text{max_distance}$ is clamped into the last bucket.
  - Intuition: the difference between "1 token apart" and "3 tokens apart" really matters; the difference between "60 apart" and "62 apart" matters very little. 
- Direction matters. In bidirectional attention (T5 encoder), the sign of $i-j$ is meaningful: attending forward should be allowed to have different biases than attending backward. T5 splits the buckets in half: one half for $i-j > 0$, the other for $i-j < 0$. In causal attention (decoder), only one direction is possible, so all buckets serve that direction.
- implementation detail: the bias matrix is added to the $QK^T$ matrix

pros:
  - learned: the model can choose how strong the bias should be at each distance, per head; different heads can specialize (e.g. local vs long-range).
  - translation-invariant: the bias depends on $i-j$, not on absolute position, which matches what we usually want from position info.
  - far cheaper in parameters than learned absolute embeddings
  - bucketing lets a finite set of parameters cover arbitrarily long distances.

cons:
  - still learned, so the model can only really trust the bias at distances well-represented in the training data; bucket values for rare long distances stay under-trained.
  - bucketing is a heuristic and log-spaced buckets discard fine-grained distance information at long ranges.
  - empirically too slow to train
    - extra matrix addition in self-attention
    - changes in every step --> difficult for KV cache

**ALiBi (Attention with Linear Biases):**
- bias is linear in the distance, deterministic, not bounded. Each head $h$ has a fixed (not learned) slope $\mu_h$, typically chosen as a geometric sequence over heads.
$$
\text{bias}(i,j) = \mu \cdot (j-i)
$$
For causal attention $j \le i$, so $(j-i) \le 0$, and $\mu > 0$, giving a negative bias that grows with distance — a recency prior.
- pros:
  - zero learnable parameters in the positional component.
  - explicitly designed for length extrapolation: a model trained at context length $L$ generalizes well to lengths $> L$ at inference, because the bias is defined for any $i-j$ and decays smoothly.
  - dirt simple to implement: just a precomputed bias matrix added to attention logits.
  - per-head slopes give a natural mix of short-range and long-range heads without any learning.
- cons:
  - hard-codes a strong recency bias. Tasks where the relevant token is far away (e.g. retrieval over a long context) can suffer.
  - slopes are predetermined, not learned: the model can't reweight long vs short range based on the task.
  - linear (unbounded) penalty: very far tokens get arbitrarily large negative bias, effectively masked.

Con for both 'attention bias' approaches: 
- the bias is added to the *score*, after the dot product is computed and not built into $q$ and $k$
  - it is a scalar that depends only on the distance.
  - it doesn't interact with the content of the vectors $q$ and $k$
  - so it can't express direction-dependent interactions between content and position (cf. RoPE).

##### 1.2.2. relative positional information via rotary embeddings

**RoPE (Rotary Positional Embeddings):**
- key insight: rotate $q$ and $k$ by angles proportional to their absolute positions
  - $v_{\text{dog at position 1}} = v_{\text{dog at position 0}} \times  \text{rotation matrix with angle } \theta$
  - $v_{\text{dog at position 4}} = v_{\text{dog at position 0}} \times  \text{rotation matrix with angle } 4\theta$
- this makes their inner product depend only on the relative offset.
  - i.e., angle between two target tokens is preserved as long as they have the same distance, e.g.,
    - angle between "pig" and "dog" in "The pig chased the dog" is equal to
    - angle between "pig" and "dog" in "Once upon a time, the pig chased the dog".

Implementation:
- rotates the query and key vectors in 2D subspaces by an angle proportional to their absolute position.
- query at position $m$: $q_m = x_m \cdot W_q \cdot R^T_{\theta , m}$
- key at position $n$: $k_n = x_n \cdot W_k \cdot R^T_{\theta , n}$
- $R_{\theta, m}$ is a $d \times d$ block-diagonal rotation matrix made of $d/2$ blocks of size $2 \times 2$. 
- For $1 \le i \le \frac{d}{2}$, the $i$-th block rotates the pair of dimensions $(2i-1, 2i)$ by angle $m \theta_i$:
$$
\text{block}_i(m) = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \phantom{-}\cos(m\theta_i) \end{pmatrix}
$$
- frequencies are geometrically spaced (same convention as sinusoidal):
$$
\theta_i = 10000^{-2(i-1)/d}
$$
- full matrix:
$$
R_{\theta, m} = \begin{pmatrix}
\cos(m\theta_1) & -\sin(m\theta_1) & 0 & 0 & \cdots & 0 & 0 \\
\sin(m\theta_1) & \phantom{-}\cos(m\theta_1) & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos(m\theta_2) & -\sin(m\theta_2) & \cdots & 0 & 0 \\
0 & 0 & \sin(m\theta_2) & \phantom{-}\cos(m\theta_2) & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos(m\theta_{d/2}) & -\sin(m\theta_{d/2}) \\
0 & 0 & 0 & 0 & \cdots & \sin(m\theta_{d/2}) & \phantom{-}\cos(m\theta_{d/2})
\end{pmatrix}
$$
- Reminder: Rotations in the same plane compose by adding angles: $R_a^T R_b = R_{b-a}$
- Plugging $q_m$ and $k_n$ into the attention score:
$$
q_m k_n^T = (x_m W_q)\, R_{\theta, m}^T R_{\theta, n} \,(x_n W_k)^T = (x_m W_q)\, R_{\theta,\, n-m} \,(x_n W_k)^T.
$$
- The score depends only on the relative offset $n - m$, even though each token only ever saw a rotation by its own *absolute* position.
- visually, an n-dimensional corkscrew that is rotating in space

In practice, you never materialize the full matrix $R_{\theta, m}$. It's $d \times d$ but mostly zeros, so it would be wasteful.
- the same computation can be expressed in a much simpler way
- with two elementwise vector multiplications (denoted $\odot$), and one vector addition
- using precomputed sine and cosine vectors:
$$
R_{\theta, m}\, x \;=\;
\underbrace{\begin{pmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \\ \vdots \\ x_{d-1} \\ x_d \end{pmatrix}}_{x}
\odot
\underbrace{\begin{pmatrix} \cos(m\theta_1) \\ \cos(m\theta_1) \\ \cos(m\theta_2) \\ \cos(m\theta_2) \\ \vdots \\ \cos(m\theta_{d/2}) \\ \cos(m\theta_{d/2}) \end{pmatrix}}_{\cos_m}
\;+\;
\underbrace{\begin{pmatrix} -x_2 \\ \phantom{-}x_1 \\ -x_4 \\ \phantom{-}x_3 \\ \vdots \\ -x_d \\ \phantom{-}x_{d-1} \end{pmatrix}}_{\tilde{x}}
\odot
\underbrace{\begin{pmatrix} \sin(m\theta_1) \\ \sin(m\theta_1) \\ \sin(m\theta_2) \\ \sin(m\theta_2) \\ \vdots \\ \sin(m\theta_{d/2}) \\ \sin(m\theta_{d/2}) \end{pmatrix}}_{\sin_m}
$$
- compactly: $R_{\theta, m}\, x = x \odot \cos_m \,+\, \tilde{x} \odot \sin_m$, where
  - $\cos_m, \sin_m \in \mathbb{R}^d$ are built by repeating each $\cos(m\theta_i)$ (resp. $\sin(m\theta_i)$) twice.
  - $\tilde{x}$ is the "pair-swap with sign flip" of $x$. 
- $O(d)$ per token instead of $O(d^2)$, no sparse kernels needed, and trivially vectorizable on GPU.

```python
# RoPE implementation
import torch

def build_rope_cache(seq_len, dim, base=10000.0):
    """Precompute the cos_m and sin_m tensors once, reused across all layers & heads.
    Returns: two tensors of shape (seq_len, dim).
    """
    assert dim % 2 == 0
    half = dim // 2
    theta = base ** (-torch.arange(half).float() * 2 / dim)   # (d/2,)
    pos = torch.arange(seq_len).float()                       # (T,)
    angles = torch.outer(pos, theta)                          # (T, d/2)
    # duplicate each value: [θ1, θ1, θ2, θ2, ...] to get shape (T, d)
    cos = angles.cos().repeat_interleave(2, dim=-1)
    sin = angles.sin().repeat_interleave(2, dim=-1)
    return cos, sin


def pair_swap_neg(x):
    """Compute x̃. Input and output are torch.Tensor of the same shape"""
    x_even = x[..., 0::2]                       # x_1, x_3, x_5, ...
    x_odd  = x[..., 1::2]                       # x_2, x_4, x_6, ...
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def apply_rope(x, cos, sin):
    """Apply RoPE rotation to x.
    x:        torch.Tensor (..., T, d) e.g. (batch, heads, tokens, head_dim)
    cos, sin: torch.Tensor      (T, d) precomputed by build_rope_cache (truncated to current T)
    """
    T = x.shape[-2]
    return x * cos[:T] + pair_swap_neg(x) * sin[:T]

def rope_attention(q, k, v, cos, sin):
    """Scaled dot-product attention with RoPE applied to q and k (not v)."""
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)
    d = q.shape[-1]
    scores = q @ k.transpose(-1, -2) / d ** 0.5
    return scores.softmax(dim=-1) @ v

class RoPEAttention(torch.nn.Module):
    def __init__(self, max_seq_len, head_dim):
        super().__init__()
        cos, sin = build_rope_cache(max_seq_len, head_dim)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, q, k, v):
        return rope_attention(q, k, v, self.cos, self.sin)

# Note: production code (LLaMA, HuggingFace) uses a mathematically equivalent
# "halved" layout where dims (i, i+d/2) form a pair instead of (2i-1, 2i).
# It's the same operation up to a fixed permutation, and more cache-friendly.
```

**Pros:**
- best of both worlds:
  - the per-token application of **absolute** encoding: you can compute the positional transform from just a single token's own position, cheap, no need to construct a full pairwise bias matrix.
  - the **relative**-encoding behavior: the score depends only on the relative offset $n - m$
- Position acts on $q$ and $k$ themselves, so content and position interact *multiplicatively*
  - heads can learn content-dependent positional behavior (T5 and ALiBi can't).
- Zero learnable positional parameters; 
  - one cos/sin table is shared across all layers and heads.
- $O(d)$ per token via the elementwise formula
  - no sparse kernels, no pairwise bias matrix.
- KV-cache friendly: 
  - previously cached $k_n$ stays valid; 
  - only the new $q_m, k_m$ need rotation.
- Plays well with length-extension techniques (NTK-aware scaling, YaRN, position interpolation).
- Dominant in frontier LLMs (LLaMA, Mistral, Qwen, DeepSeek, Gemma, etc.). strong empirical track record.

**Cons:**
- Native extrapolation past training context is limited
  - high-frequency components dephase quickly, so quality drops without scaling tricks.
- Rotation frequencies are fixed at design time (like sinusoidal); the model can't learn the spectrum.
- Rotations are unitary (rotate without stretching or shrinking), so $\langle q, k \rangle$ doesn't decay with distance the way ALiBi imposes; any recency bias has to be learned.

### 2. Layer normalization
### 3. Attention approximation
### 4. Transformer-based models
### 5. BERT deep dive

# References

- Kazemnejad's post on sinusoidal embeddings: https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
- Google T5 paper (Raffel et al., 2020): https://arxiv.org/pdf/1910.10683
- ALiBi: Train short, Test long paper (Press et al., 2021): https://arxiv.org/pdf/2108.12409
- RoPE: RoFormer paper (Su et al., 2021): https://arxiv.org/pdf/2104.09864
