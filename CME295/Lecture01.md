# Lecture 1 — Transformers

- **Date watched:** 2026-05-21
- **Lecture:** https://www.youtube.com/watch?v=Ub3GoFaUcds
- **Slides:** https://cme295.stanford.edu/slides/fall25-cme295-lecture1.pdf
- **Notes taken by:** Elnaz Shafaei 

## Notes

### 1. NLP tasks overview
- single-label classification
  - e.g. sentiment extraction
    - datasets: IMDB critiques, amazon reviews, twitter
    - metrics: accuracy, precision, recall, f1-score
- multi-label classification
  - e.g. named entity recognition
    - datasets: annotated Reuters news (CoNLL)
    - metrics: accuracy, precision, recall, f1-score at a token level, per entity type
- Generation
  - e.g., machine translation
    - datasets: WMT
    - metrics: BLEU, ROUGE, perplexity

---

### 2. Tokenization
- at word level
  - pros: simple, fast, interpretable 
  - cons: not suitable for all langs, great risk of OOV, does not leverage morphology
- at subword level, e.g. WordPiece, BPE
  - pros: leverages common affixes, learned from the data, suitable for all langs
  - cons: still some chance of OOV
- at character level
  - pros: very small chance of OOV, robust to casing and misspellings
  - cons: more tokens i.e. slower computation, uninterpretable embeddings

---

### 3. Word representation 
- embeddings
  - word2vec: proxy tasks (CBOW, skip-gram)
    - pros: simple yet powerful, intuitive embeddings (proportional analogies)
    - cons: word order does not matter, are not context aware

---

### 4. RNNs 
- sequential nature
  - pros: 
    - sequence-to-sequence tasks / word order does matter
    - potentially input can be of any length
    - model size does not increase with input size
    - takes historical information into account
    - weights are shared across time
  - cons:
    -  serial computation, slow 
    - long-term dependencies: 
      - difficulty of accessing information from a long time ago
      - vanishing/exploding gradient problem 
    - unable to look into the future

- vanilla RNNs:
  - output at t is a function of input at t and activation at t-1
- GRUs/LSTMs:
  - output at t is a function of input at t, activation at t-1, and hidden state at t-1
- See Cheat Sheet at https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks/

---

### 5. Self-attention mechanism 
- history: cross-attention with RNNs (2014)
- "Attention is all you need": cross-attention and self-attention, no need for RNNs (2017)

#### 5.1. a single attention head:

For a given Head, we have:
- $d$ — model dimensionality = input dimensionality = output dimensionality
- $W^Q, W^K \in \mathbb{R}^{d \times d_k}$, $W^V \in \mathbb{R}^{d \times d_v}$ — learned projection matrices


##### 5.1.1. one attention head, one token:

```
   v_a      v_cute   v_teddy_bear   v_is    v_reading   v_.
 ┌─────┐  ┌──────┐  ┌───────────┐  ┌────┐  ┌─────────┐ ┌───┐
 │  a  │  │ cute │  │ teddy bear│  │ is │  │ reading │ │ . │
 └─────┘  └──────┘  └───────────┘  └────┘  └─────────┘ └───┘
   k_aᵀ   k_cuteᵀ  k_teddy_bearᵀ   k_isᵀ   k_readingᵀ  k_.ᵀ
      \      \         |         /         /         /
       \      \        |        /         /         /
        \______\_______▼_______/_________/_________/
                       │
                 q_teddy_bear
              ┌───────────────┐
  (a) (cute)  │  teddy  bear  │ (is) (reading) (.)
              └───────────────┘
               ↑ current token
```
- Intuitively, for one position in the input sequence, we
  - compute a similarity score to all positions in the input sequence (dot product of $q$ and $k^T$s)
  - use these scores as weights to compute a weighted average of the values 
- So in one head, for target token $i$ and for all tokens $1 \le j \le n$, we compute:
$$q_i = x_iW^Q; \quad k_j = x_jW^K; \quad v_j = x_jW^V$$
$$\textbf{score}_{ij} = \text{softmax} \left({q_i k_j \over \sqrt{d_k}}\right)$$
$$\textbf{head}_i = \sum_j \textbf{score}_{ij} v_j$$

where:
- $x_i, x_j \in \mathbb{R}^{1 \times d}$ — input embeddings
- $\sqrt{d_k}$ — scaling factor that keeps the dot products from growing too large (positive or negative). 
Exponentiating large values (in softmax) can lead to numerical issues and loss of gradients during
training. To avoid this, we scale the dot product by a factor related to the size of the
embeddings, via dividing by the square root of the dimensionality of the query and
key vectors.

Therefore: 
- $q_i, k_j \in \mathbb{R}^{1 \times d_k}$, $v_j \in \mathbb{R}^{1 \times d_v}$
- $\textbf{score}_{ij} \in \mathbb{R}^{1\times 1}$ scalar
- $\textbf{head}_i \in \mathbb{R}^{1 \times d_v}$

---

##### 5.1.2. one attention head, all tokens: 
- We parallelize the computation of all of the tokens in the input sequence
- So in the efficient computations with matrices, for $1 \le i \le n$ we have:
$$Q = XW^Q; \quad K = XW^K; \quad V = XW^V$$
$$\textbf{head} = \text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

where:
- $X \in \mathbb{R}^{n \times d}$ — input sequence of $n$ tokens, embedding dim $d$

Therefore: 
- $Q, K \in \mathbb{R}^{n \times d_k}$, $V \in \mathbb{R}^{n \times d_v}$
- $\textbf{head} \in \mathbb{R}^{n \times d_v}$

---

#### 5.2. Multi-head attention:

- A multi-head attention block can have several attention heads: $\textbf{head}^h$  where $1 \le h \le H$
- Each head has its own set of parameters
  - i.e., its own set of query, key, and value matrices: $W^{Qh}, W^{Kh}, W^{Vh}$
- This allows the head to model different aspects of the relationships among inputs
- The output of each head is **concatenated** and then fed into a single linear transformation with $W^O \in \mathbb{R}^{d_v \times d}$

##### 5.2.1. multiple attention heads, one token:

- We computed $\textbf{head}_i$ for target token $i$ in one head in 5.1.1.
- Now, the output for this target token over all heads is:

$$\textbf{output}_i = \text{Concat}\left(\textbf{head}^h_i\right) W^O$$

where:
- $\textbf{head}^h_i \in \mathbb{R}^{1 \times d_v}$
- $\text{Concat}\left(\textbf{head}^h_i\right) \in \mathbb{R}^{1 \times Hd_v}$
- $W^O \in \mathbb{R}^{Hd_v \times d}$

Therefore, $\textbf{output}_i \in \mathbb{R}^{1 \times d}$

---

##### 5.2.2. multiple attention heads, all tokens:

- We computed $\textbf{head}$ for all tokens in one head in 5.1.2.
- Now, the all tokens over all heads is:

$$\text{MulitHeadAttention}(X) = \text{Concat}\!\left(\textbf{head}\right) W^O$$

where:
- $\textbf{head} \in \mathbb{R}^{n \times d_v}$
- $\text{Concat}\left(\textbf{head}\right) \in \mathbb{R}^{n \times Hd_v}$
- $W^O \in \mathbb{R}^{Hd_v \times d}$

**Note:** Usually $H=d/d_v$, and so $W^O \in \mathbb{R}^{d \times d}$

---

### 6. Original Transformer paper

- positional embeddings: sinusoidal
- layer norm: post-norm
- no attention approximation: MQA (multi-head attention)
- training task:
  - trained autoregressively using next-token prediction on the target sentence $P(y_t | y_{\lt t}, x)$
  - where
    - $x$ is the source sentence, 
    - $y_t$ is the next target token, and
    - $y_{\lt t}$ is the previously generated target tokens

```
                                               Output Probabilities
                                                       ▲
                                               ┌───────────────┐
                                               │    Softmax    │
                                               └───────┬───────┘
                                                       ▲
                                               ┌───────────────┐
                                               │    Linear     │
                                               └───────┬───────┘
                                                       ▲
  ENCODER (×N) ┌────────────────────┐    DECODER  (×N) │
   ┌───────────┼──────────┐         │      ┌───────────┼──────────┐
   │           │          │         │      │           │          │
   │  ┌────────┴───────┐  │         │      │  ┌────────┴───────┐  │
   │  │   Add & Norm   │◄─┼──┐      │      │  │   Add & Norm   │◄─┼──┐
   │  └────────┬───────┘  │  │      │      │  └────────┬───────┘  │  │
   │           ▲          │  │      │      │           ▲          │  │
   │  ┌────────────────┐  │  │      │      │  ┌────────────────┐  │  │
   │  │  Feed Forward  │  │  │      │      │  │  Feed Forward  │  │  │
   │  └────────┬───────┘  │  │      │      │  └────────┬───────┘  │  │
   │           ▲ ─────────┼──┘      │      │           ▲ ─────────┼──┘
   │  ┌────────────────┐  │         │      │  ┌────────────────┐  │
   │  │   Add & Norm   │◄─┼──┐      │      │  │   Add & Norm   │◄─┼──┐
   │  └────────┬───────┘  │  │      │      │  └────────┬───────┘  │  │
   │           ▲          │  │      │      │           ▲          │  │
   │  ┌────────────────┐  │  │      │      │  ┌────────────────┐  │  │
   │  │   Multi-Head   │  │  │      │      │  │   Multi-Head   │  │  │
   │  │   Attention    │  │  │      │      │  │   Attention    │  │  │
   │  └─┬────┬──────┬──┘  │  │      │      │  └─┬─────┬─────┬──┘  │  │
   │    ▲    ▲      ▲     │  │      │      │    ▲     ▲     ▲     │  │
   │    Q    K      V     │  │      │      │    K     V     Q     │  │
   │    └────┴──────┘     │  │      │      │    ▲     ▲     │     │  │
   │           ▲          │  │      │      │    │     │     │     │  │
   │           │          │  │      └──────┼────┴─────┘     │     │  │
   │           │          │  │             │                │─────┼──┘
   │           │──────────┼──┘             │  ┌────────────────┐  │
   │           │          │                │  │   Add & Norm   │◄─┼──┐
   │           │          │                │  └────────┬───────┘  │  │
   │           │          │                │           ▲          │  │
   │           │          │                │  ┌────────────────┐  │  │
   │           │          │                │  │ Masked Multi-  │  │  │
   │           │          │                │  │ Head Attention │  │  │
   │           │          │                │  └─┬────┬──────┬──┘  │  │
   │           │          │                │    ▲    ▲      ▲     │  │
   │           │          │                │    Q    K      V     │  │
   │           │          │                │    └────┴──────┘     │  │
   │           │          │                │           ▲          │  │
   │           │          │                │           │──────────┼──┘
   └───────────┼──────────┘                └───────────┼──────────┘
               ▲                                       ▲
          ┌────┴────┐                             ┌────┴────┐
          │    ⊕    │ ◄── Positional Encoding ──► │    ⊕    │
          └────┬────┘                             └────┬────┘
               ▲                                       ▲
       ┌───────────────┐                       ┌───────────────┐
       │     Input     │                       │    Output     │
       │   Embedding   │                       │   Embedding   │
       └───────┬───────┘                       └───────┬───────┘
               ▲                                       ▲
            Inputs                                  Outputs
                                                (shifted right)
```

## Things to revisit later

- BLUE
- ROGUE
- perplexity
- BPE

## References

- Attention is all you need: https://arxiv.org/abs/1706.03762
- MT evaluation metrics: Stat MT (Philipp Koehn) https://www2.statmt.org/book/slides/08-evaluation.pdf
- More details on the self-attention mechanism: https://web.stanford.edu/~jurafsky/slp3/8.pdf
