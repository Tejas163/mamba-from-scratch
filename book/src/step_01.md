# Step 1: State Space Model Foundations

## Overview

In this step, we'll learn the mathematical foundations of State Space Models (SSMs). These are the building blocks that make Mamba work.

## What is a State Space Model?

A **State Space Model** represents a system through:

1. A **hidden state** $h(t)$ that captures all relevant information
2. An **input** $x(t)$ that drives the system
3. An **output** $y(t)$ that we want to predict

The key idea: instead of storing all past information explicitly (like attention), we compress it into a finite-dimensional state vector.

## Continuous-Time Formulation

The continuous-time SSM is defined by a system of linear differential equations:

$$\boxed{h'(t) = Ah(t) + Bx(t)}$$

$$\boxed{y(t) = Ch(t) + Dx(t)}$$

Where:
- $h(t) \in \mathbb{R}^N$ - hidden state vector (dimension $N$)
- $x(t) \in \mathbb{R}$ - input (scalar for simplicity)
- $y(t) \in \mathbb{R}$ - output
- $A \in \mathbb{R}^{N \times N}$ - **state transition matrix**
- $B \in \mathbb{R}^{N \times 1}$ - **input projection**
- $C \in \mathbb{R}^{1 \times N}$ - **output projection**
- $D \in \mathbb{R}$ - **feed-through** (often 0)

## Intuition: What Each Matrix Does

| Matrix | Role | Analogy |
|--------|------|---------|
| **A** | How state evolves over time | Memory dynamics |
| **B** | How input affects state | Input encoding |
| **C** | How to read output from state | State readout |

### The State Transition Matrix A

The matrix $A$ determines how information flows through time:

- **Eigenvalues $\lambda$ with $|\lambda| < 1$**: Information **decays** → short-term memory
- **Eigenvalues $\lambda$ with $|\lambda| = 1$**: Information **persists** → long-term memory  
- **Eigenvalues $\lambda$ with $|\lambda| > 1$**: Information **grows** → unstable!

For stable learning, we want eigenvalues close to the unit circle but not outside.

## Solving the Continuous-Time SSM

The ODE $h'(t) = Ah(t) + Bx(t)$ has a closed-form solution:

$$h(t) = e^{At}h(0) + \int_0^t e^{A(t-\tau)}Bx(\tau)d\tau$$

This is the **variation of constants** formula.

### Without Input (Homogeneous Solution)

When $x(t) = 0$:
$$h(t) = e^{At}h(0)$$

The matrix exponential $e^{At}$ encodes the time evolution.

### With Input (Particular Solution)

When there's input, we get an integral term. For constant input $x(t) = x$:
$$h(t) = e^{At}h(0) + \int_0^t e^{A\tau}d\tau \cdot Bx$$

The integral evaluates to:
$$\int_0^t e^{A\tau}d\tau = A^{-1}(e^{At} - I)$$

(When $A$ is invertible)

## Discretization: From Continuous to Discrete

Neural networks process **discrete** tokens. We need to convert the continuous ODE to a discrete recurrence.

### The Problem

We have:
- Input sequence: $x_0, x_1, x_2, \ldots$
- Want: $h_0, h_1, h_2, \ldots$

### Zero-Order Hold (ZOH) Discretization

Assume input is **constant** between samples:
$$x(t) = x_k \quad \text{for} \quad t \in [k\Delta, (k+1)\Delta)$$

Where $\Delta$ is the **step size** (sampling interval).

### Derivation

At step $k+1$:
$$h_{k+1} = e^{A\Delta}h_k + \int_0^{\Delta} e^{A\tau}d\tau \cdot Bx_k$$

Evaluating the integral:
$$h_{k+1} = e^{A\Delta}h_k + (e^{A\Delta} - I)A^{-1}Bx_k$$

### Define Discretized Matrices

$$\boxed{\bar{A} = e^{A\Delta}}$$

$$\boxed{\bar{B} = (e^{A\Delta} - I)A^{-1}B\Delta}$$

### The Discrete SSM

$$\boxed{h_k = \bar{A}h_{k-1} + \bar{B}x_k}$$

$$\boxed{y_k = Ch_k}$$

This is a **linear recurrent neural network**!

## Special Case: Diagonal A

For efficiency, we often use **diagonal** matrices. Let $A = \text{diag}(\lambda_1, \ldots, \lambda_N)$:

$$\bar{A}_{ii} = e^{\lambda_i \Delta}$$

$$\bar{B}_i = \frac{e^{\lambda_i\Delta} - 1}{\lambda_i} \cdot B_i$$

This is much cheaper to compute - no matrix exponentials!

## Parameterization: Learning the Log-Eigenvalues

Instead of learning $A$ directly (which might become unstable), we parameterize:

$$A = \text{diag}(\exp(\theta))$$

where $\theta$ is learnable. This ensures:
- Eigenvalues are always positive
- Log-eigenvalues can be any real number
- Stability is guaranteed

## The Complete SSM Forward Pass

```python
class SSM(nn.Module):
    def __init__(self, d_model, d_state):
        # A: diagonal, parameterized by log-eigenvalues
        self.A_log = nn.Parameter(torch.randn(d_state))
        
        # B and C: projections
        self.B = nn.Linear(d_model, d_state)
        self.C = nn.Linear(d_state, d_model)
        
        # Delta: step size
        self.delta = nn.Parameter(torch.ones(d_state) * 0.1)
    
    def forward(self, x):
        # Discretize
        A_bar = torch.exp(self.A_log * self.delta)
        B_bar = (A_bar - 1) / self.A_log * self.delta
        
        # Sequential scan
        h = torch.zeros(batch, d_state)
        outputs = []
        for t in range(seq_len):
            h = A_bar * h + self.B(x[:, t])
            outputs.append(self.C(h))
        
        return torch.stack(outputs, dim=1)
```

## Why This Matters

This basic SSM has:

✅ **Linear complexity** - O(N) per step
✅ **Fixed state size** - Memory efficient
✅ **Differentiable** - Can backprop through time
❌ **Sequential** - Can't parallelize across time
❌ **Not selective** - Same dynamics for all inputs

The key innovations in Mamba address these limitations!

## Summary

In this step, we learned:

1. **Continuous-time SSM**: $h' = Ax + Bx$, solved by matrix exponential
2. **Discretization**: ZOH gives $\bar{A} = e^{A\Delta}$, $\bar{B} = (e^{A\Delta}-I)A^{-1}B\Delta$
3. **Diagonal form**: Simplifies computation significantly
4. **Sequential computation**: Simple but slow

## Next Step

In [Step 2](./step_02.md), we'll implement the **parallel scan algorithm** - the key insight that makes SSMs fast!
