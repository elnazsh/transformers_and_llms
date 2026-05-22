# Lecture 2 — Transformer-Based Models & Tricks

- **Date watched:** 2026-05-22
- **Lecture:** https://www.youtube.com/watch?v=yT84Y5zCnaA
- **Slides:** https://cme295.stanford.edu/slides/fall25-cme295-lecture1.pdf
- **Notes taken by:** Elnaz Shafaei
- 
## Notes

### 1. Position embeddings
- self-attention by itself is permutation-invariant; meaning it doesn’t know whether a token came first or last.
- positional embeddings inject sequence order information
- two broad approaches: learned positional embeddings vs hardcoded/static positional embeddings

#### 1.1. learned positional embeddings
- how it works:
  - model has a trainable vector for each position $p_i$
  - and input becomes $x_i + p_i$ where $p_i$ is optimized during training
- pros:
  - the model can discover whatever positional structure helps the task
  - different domains have different positional behavior, e.g. code vs protein vs music vs text
- cons:
  - limited to the max position present in the training data
  - need to retrain for longer sequences
  - more parameters
  - the model could learn arbitrary patterns solely based on position 

#### 1.2. hardcoded positional embeddings
- Main idea: that closer (in position) tokens are more similar (in semantics), and farther tokens are more dissimilar.
- different methods:
  - using _absolute_ positional information:
    - sinusoidal embeddings: in the original attention paper, kinda obsolete now
  - using _relative_ positional information:
    - relative positional embeddings (RPE): used in T5
    - ALiBi
    - rotary positional embeddings (RoPE): dominant in frontier LLMs nowadays
- pros:
  - extendable to any sequence length

##### 1.2.1. absolute positional information: sinusoidal embeddings
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

##### 1.2.2. relative positional information
- at the end, we want the closer tokens to be more similar to each other _in the self-attention computation_
- so we can just modify the self-attention formula to directly reflect this idea, rather than adding a positional embedding to the input.
- we can do this by adding a _bias_ to the dot product of the query and key vectors,
- this bias value should be large for tokens closer to the query, and small for tokens further away from the query.


### 2. Layer normalization
### 3. Attention approximation
### 4. Transformer-based models
### 5. BERT deep dive

# References

- Kazemnejad's post on sinusoidal embeddings: https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
- Google T5 paper (2023): https://arxiv.org/pdf/1910.10683