# Step 2: Parallel Scan Algorithm

## Overview

In the previous step, we implemented SSM as a **sequential recurrence** - computing one token at a time. This is O(N) sequentially but can't be parallelized.

In this step, we'll learn the **parallel scan algorithm** - the key insight that makes Mamba efficient!

## The Problem

Sequential computation:
```python
h = 0
for t in range(seq_len):
    h = A_bar * h + B_bar * x[t]
    y[t] = C * h
```

This is **inherently sequential** - we must compute $h_t$ before $h_{t+1}$.

But can we compute all states in parallel?

## The Key Insight: Associativity

The SSM recurrence has a special structure:

$$h_k = \bar{A}h_{k-1} + \bar{B}x_k$$

Let's unroll this:

$$h_0 = \bar{B}x_0$$
$$h_1 = \bar{A}\bar{B}x_0 + \bar{B}x_1$$
$$h_2 = \bar{A}^2\bar{B}x_0 + \bar{A}\bar{B}x_1 + \bar{B}x_2$$

We can see: **each $h_k$ is a sum of weighted inputs**.

## Reformulating as a Scan

Define the **scan pair** $(a_k, b_k)$:

- $a_k$ = accumulated state transition
- $b_k$ = accumulated input contribution

The recurrence can be written as:
$$(a_k, b_k) = (a_{k-1}, b_{k-1}) \oplus (\bar{A}, \bar{B}x_k)$$

Where the binary operator $\oplus$ is:
$$\boxed{(a_L, b_L) \oplus (a_R, b_R) = (a_L \cdot a_R, a_L \cdot b_R + b_L)}$$

## Proving Associativity

We need to verify: $(a \oplus b) \oplus c = a \oplus (b \oplus c)$

Let:
- $a \oplus b = (a_1a_2, a_1b_2 + b_1)$
- $(a \oplus b) \oplus c = (a_1a_2c, a_1a_2c_0 + a_1b_2 + b_1)$

Similarly:
- $b \oplus c = (a_2c_0, a_2c_1 + b_2)$
- $a \oplus (b \oplus c) = (a_1a_2c_0, a_1(a_2c_1 + b_2) + b_1)$

These are equal! ✅

## Visualizing Parallel Scan

```
Sequential (O(N)):
    h0 → h1 → h2 → h3 → h4 → h5 → h6 → h7
    
Parallel (O(log N)):
         h0 ──┬── h1     h2 ──┬── h3     h4 ──┬── h5     h6 ──┬── h7
             │      │          │      │          │      │          │
             ▼      ▼          ▼      ▼          ▼      ▼          ▼
         combine  combine  combine  combine
             │              │              │
             ▼              ▼              ▼
         h0-1          h2-3          h4-5
             │              │              │
             ▼              ▼              ▼
         combine          combine          combine
             │                              │
             ▼                              ▼
         h0-3                              h4-7
             │                              │
             ▼                              ▼
         combine (final result in O(log N) steps!)
             │
             ▼
         h0-7
```

## The Parallel Scan Algorithm

```python
def parallel_scan(els, op):
    """
    Compute parallel scan using binary tree.
    
    els: (batch, seq_len, ...) - elements to scan
    op: binary operator (associative)
    """
    batch, seq_len = els.shape[:2]
    
    # Pad to power of 2
    n = 1
    while n < seq_len:
        n *= 2
    
    # Pad with identity
    padded = pad(els, n)
    
    # Bottom-up passes
    for stride in 1, 2, 4, 8, ...:
        # Combine elements stride apart
        even = padded[:, ::stride*2]
        odd = padded[:, stride::stride*2]
        
        combined = op(even, odd)
        
        # Write to odd positions
        padded[:, stride::stride*2] = combined
    
    return padded[:, :seq_len]
```

## Implementing SSM Scan

For SSM, the operator is:
```python
def ssm_operator(left, right):
    A_left, B_left = left
    A_right, B_right = right
    
    A_combined = A_left @ A_right
    B_combined = A_left @ B_right + B_left
    
    return (A_combined, B_combined)
```

But we can simplify for **diagonal A**:

```python
def ssm_scan_diagonal(A_bar, B_bar):
    """
    A_bar: (batch, seq_len, d_state) - diagonal elements
    B_bar: (batch, seq_len, d_state) - input contributions
    """
    # For diagonal A:
    # h_k = A_k * h_{k-1} + B_k
    
    # Binary operator:
    # (A_prev, B_prev) ⊕ (A_curr, B_curr) = 
    # (A_prev * A_curr, A_prev * B_curr + B_prev)
    
    for stride in [1, 2, 4, 8, ...]:
        # Even indices: keep as-is
        # Odd indices: combine with stride-neighbor
        A_even = A_bar[:, ::stride*2]
        B_even = B_bar[:, ::stride*2]
        A_odd = A_bar[:, stride::stride*2]
        B_odd = B_bar[:, stride::stride*2]
        
        A_bar[:, stride::stride*2] = A_even * A_odd
        B_bar[:, stride::stride*2] = A_even * B_odd + B_even
    
    return B_bar
```

## Complexity Analysis

| Operation | Sequential | Parallel Scan |
|-----------|-----------|---------------|
| Time | O(N) | O(log N) |
| Memory | O(N) | O(N) |
| Parallelizable | ❌ | ✅ |

The parallel scan maintains O(N) memory but dramatically reduces computation time!

## Full Implementation

```python
class SSMWithScan(nn.Module):
    def __init__(self, d_model, d_state):
        self.A_log = nn.Parameter(torch.randn(d_state))
        self.B = nn.Linear(d_model, d_state)
        self.C = nn.Linear(d_state, d_model)
        self.delta = nn.Parameter(torch.ones(d_state) * 0.1)
    
    def forward(self, x):
        # Discretize
        A_bar = torch.exp(self.A_log * self.delta)
        B_proj = self.B(x)  # (batch, seq, d_state)
        
        # Discretize B
        B_bar = B_proj * (A_bar - 1) / self.A_log * self.delta
        
        # Parallel scan
        h = parallel_scan(A_bar, B_bar)
        
        # Output
        return self.C(h)
```

## Why This Works

The magic is in **reformulating** the sequential recurrence as an **associative scan**.

1. **Sequential**: $h_k = Ah_{k-1} + Bx_k$
2. **Unrolled**: $h_k = A^kx_0 + A^{k-1}x_1 + \ldots + x_k$
3. **Associative**: The sum can be parallelized!

## Benchmark: Sequential vs Parallel

```python
# Benchmark code
seq_lens = [32, 64, 128, 256, 512, 1024]

for seq_len in seq_lens:
    # Sequential
    t_seq = time.SequentialSSM(seq_len)
    # Parallel
    t_par = time.ParallelSSM(seq_len)
    print(f"{seq_len}: {t_seq/t_par:.1f}x speedup")
```

Expected results:
- 32 tokens: ~2x speedup
- 128 tokens: ~10x speedup  
- 512 tokens: ~50x speedup
- 1024 tokens: ~100x+ speedup

## Summary

In this step, we learned:

1. **Associative property**: SSM recurrence is associative
2. **Binary operator**: $(A_L, B_L) \oplus (A_R, B_R) = (A_LA_R, A_LB_R + B_L)$
3. **Parallel scan**: Compute all states in O(log N)
4. **Efficiency**: Dramatic speedup over sequential

## Key Takeaway

The parallel scan is the **secret sauce** that makes Mamba fast. Without it, SSMs would be too slow for practical use.

## Next Step

In [Step 3](./step_03.md), we'll add **selection** - making parameters input-dependent to enable content-based reasoning!
