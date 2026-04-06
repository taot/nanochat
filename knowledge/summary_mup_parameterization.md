# Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer

**Paper:** arXiv:2203.03466 (Greg Yang, Edward J. Hu, et al., Microsoft/OpenAI)

## Core Idea

muP (Maximal Update Parameterization) ensures that each layer's activations are updated on the same order during training *regardless of width*. This enables **zero-shot HP transfer**: tune HPs on a small (proxy) model, then copy them directly to a large (target) model.

## The Three muP Tables

The paper provides three equivalent formulations. All differ from SP (Standard Parameterization) and from each other only by constant rescalings. Below we reproduce the primary table (Table 1, `tab:MUP`) and the alternative formulation (Table 3, `tab:MUPalt`).

### Table 1: Primary muP (tab:MUP)

| | Input weights & biases | Output weights | Hidden weights |
|---|---|---|---|
| **Init Var** | 1/fan_in | **1/fan_in^2** (SP: 1/fan_in) | 1/fan_in |
| **SGD LR** | **fan_out** (SP: 1) | **1/fan_in** (SP: 1) | 1 |
| **Adam LR** | 1 | **1/fan_in** (SP: 1) | **1/fan_in** (SP: 1) |

### Table 3: Alternative muP for Easier Implementation (tab:MUPalt)

| | Input weights & biases | Output weights | Hidden weights |
|---|---|---|---|
| **Init Var** | 1/fan_in | **1** (SP: 1/fan_in) | 1/fan_in |
| **Multiplier** | 1 | **1/fan_in** (SP: 1) | 1 |
| **SGD LR** | **fan_out** (SP: 1) | **fan_in** (SP: 1) | 1 |
| **Adam LR** | 1 | 1 | **1/fan_in** (SP: 1) |

Key difference: In `tab:MUPalt`, all vector-like parameters (input/output/biases) share the same Adam LR scaling rule (constant), while hidden (matrix-like) parameters get LR scaled by 1/fan_in.

### Table 4: Original TP4-style muP (tab:MUPorig)

| | Input weights & biases | Output weights | Hidden weights |
|---|---|---|---|
| **Init Var** | **1/fan_out** (SP: 1/fan_in) | 1/fan_in | 1/fan_in |
| **Multiplier** | **sqrt(fan_out)** (SP: 1) | **1/sqrt(fan_in)** (SP: 1) | 1 |
| **SGD LR** | 1 | 1 | 1 |
| **Adam LR** | **1/sqrt(fan_out)** (SP: 1) | **1/sqrt(fan_in)** (SP: 1) | **1/fan_in** (SP: 1) |

## Parameter Classification

The paper classifies parameters by how many "infinite" (width-scaling) dimensions they have:

- **Matrix-like** (2 infinite dims): Q, K, V, O projections, MLP W1/W2 -- these are "hidden weights"
- **Vector-like** (1 infinite dim): word embeddings (input), unembeddings (output), layernorm weights, biases
- **Scalar-like** (0 infinite dims): learnable scalar multipliers, positional bias -- constant init, constant LR

## Transformer-Specific Rules

### Attention Scaling
muP requires **1/d attention** instead of 1/sqrt(d):
```
AttnLogit = alpha_attn * sqrt(d_head_0) / d_head * q^T k
```
This is because during training, q and k become correlated, so q^T k scales like d (LLN) rather than sqrt(d) (CLT).

### Walkthrough for Transformer with Adam (from tab:MUP/MUPalt)

Using base width d_model_0, define d_tilde = d_model / d_model_0:

- **Word embeddings**: init var = sigma^2 (constant), Adam LR = eta (constant)
- **Q, K, V, O matrices**: init var = sigma^2/fan_in, Adam LR = eta/d_tilde
- **MLP W1, W2**: init var = sigma^2/fan_in, Adam LR = eta/d_tilde
- **Unembeddings**: init var = sigma^2/(d_model * d_tilde), Adam LR = eta/d_tilde
- **Layernorm**: standard init (weight=1, bias=0), Adam LR = eta (constant)
- **Scalars**: init to constant, Adam LR = constant

### What Transfers and What Doesn't

| Transferable | Not Transferable | Transferred Across |
|---|---|---|
| LR, momentum, Adam beta, LR schedule, init, parameter multipliers | Regularization (dropout, weight decay) | Width, depth*, batch size*, training time*, seq length* |

(* = empirically validated only, no theoretical guarantee)

### AdamW and Weight Decay
The paper explicitly recommends AdamW (not Adam + L2) for muP because AdamW automatically scales weight decay correctly.

## Relation to nanochat

### What nanochat does that aligns with muP

1. **Width-scaled initialization** (`gpt.py:221-230`):
   - Hidden weights (Q, K, V, c_fc): init std = 1/sqrt(n_embd) -- matches muP's init var = 1/fan_in
   - Output projections (c_proj): zero-initialized -- compatible with muP (zero init variance is allowed)
   - Embedding (wte): init std = 0.8 (constant, width-independent) -- matches muP input weight rule
   - Unembedding (lm_head): init std = 0.001 (constant, but muP says it should scale as 1/fan_in or 1/fan_in^2 depending on formulation)

2. **Width-scaled learning rates** (`gpt.py:383-392`):
   - AdamW LR for embeddings/unembeddings scaled by `(d_model/768)^(-0.5)` = `1/sqrt(d_tilde)`
   - This is a *partial* muP: muP prescribes 1/d_tilde for output weights with Adam, nanochat uses 1/sqrt(d_tilde)

3. **Parameter grouping** (`gpt.py:369-409`):
   - Matrix-like params -> Muon optimizer (separate from AdamW)
   - Vector-like params (embeddings, unembeddings) -> AdamW with width-scaled LR
   - Scalar-like params (resid_lambdas, x0_lambdas) -> AdamW with constant LR
   - This classification matches muP's matrix/vector/scalar taxonomy perfectly

4. **Reference model for HP transfer** (`base_train.py:270-273`):
   - d12 (768-dim) as reference model, HPs tuned there and transferred to larger depths
   - This is exactly the muP workflow: tune on proxy, transfer to target

### Where nanochat diverges from muP

1. **Attention scaling**: nanochat uses standard `1/sqrt(d)` via `F.scaled_dot_product_attention`, NOT `1/d` as muP prescribes. This is a notable departure.

2. **LR scaling exponent**: muP says Adam LR for hidden weights should scale as `1/fan_in` (i.e., `1/d_tilde`). Nanochat scales AdamW parameters by `1/sqrt(d_tilde)` instead. The matrix params use Muon (not Adam), so muP theory doesn't directly apply there.

3. **Muon optimizer**: The paper only covers SGD and Adam. Nanochat uses Muon for all matrix-like parameters. Muon does Newton-Schulz orthogonalization which implicitly normalizes updates, somewhat analogous to the Frobenius normalization optimizers discussed in the paper (LARS, LAMB, etc.). The paper notes these optimizers need LR scaling of `1/sqrt(fan_in)` for hidden weights (less aggressive than Adam's `1/fan_in`), which may explain why nanochat doesn't apply additional width-based LR scaling to Muon.

4. **Depth transfer**: muP primarily guarantees transfer across *width*. Nanochat transfers across *depth* (from d12 to d24, d32, etc.), which the paper marks as "empirically validated only" with caveats (e.g., init std doesn't transfer well across depth).

5. **Batch size / weight decay scaling**: nanochat applies `eta ~ sqrt(B/B_ref)` and `lambda ~ sqrt(B/B_ref) * (D_ref/D)`. These are not from muP but from other scaling law papers (Power Lines, T_epoch framework). The muP paper itself says batch size transfer works empirically but doesn't prescribe specific formulas.

6. **Unembedding init**: muP says output weight init var should be 1/fan_in^2 (much smaller than 1/fan_in). Nanochat uses std=0.001 for lm_head, which is a fixed small constant -- roughly in the spirit of muP's "make output weights small" but doesn't scale with width.

### Potential improvements inspired by the paper

1. **1/d attention**: Switching from 1/sqrt(d) to 1/d attention could improve HP transfer across widths. With a base width, one could use `alpha_attn * sqrt(d_head_0) / d_head` for smooth transition.

2. **Stricter LR scaling**: For AdamW parameters, consider 1/d_tilde instead of 1/sqrt(d_tilde) for closer alignment with muP theory.

3. **Unembedding init scaling**: Scale lm_head init as 1/n_embd (or 1/n_embd^2 for init variance) rather than fixed 0.001.

4. **"Wider is better" debugging**: The paper suggests checking that wider models strictly outperform narrower ones as a cheap debug tool for muP correctness.
