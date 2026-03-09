# Step 3: Selective State Spaces

## Overview

In the previous step, we implemented the parallel scan for efficient SSM computation. Now we'll add the **selective mechanism** - Mamba's key innovation that enables content-based reasoning.

## The Problem with Basic SSMs

In our basic SSM:
- $A$, $B$, $C$ are **fixed** after training
- Same state dynamics for all inputs
- Cannot adapt to **content**

This is fundamentally **positional** - the model can only use position information, not content.

## Mamba's Solution: Selective Parameters

Make parameters **input-dependent**:

$$\boxed{B_k = \text{Linear}_B(x_k)}$$
$$\boxed{C_k = \text{Linear}_C(x_k)}$$
$$\boxed{\Delta_k = \text{softplus}(\text{Linear}_\Delta(x_k))}$$

Now each token can have different:
- Input influence ($B_k$)
- Output extraction ($C_k$)
- Time step size ($\Delta_k$)

## Why Selection Matters

Without selection:
- All tokens treated equally
- Cannot ignore noise
- Cannot focus on relevant context

With selection:
- Learn to ignore irrelevant tokens
- Focus on key information
- Content-based reasoning!

## Implementation

```python
class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state, dt_rank=16):
        # A: state dynamics (fixed, not selective)
        self.A_log = nn.Parameter(torch.randn(d_state))
        
        # Projections to compute selective parameters
        self.x_proj = nn.Linear(d_model, dt_rank + d_state * 2)
        
    def forward(self, x):
        # Compute selective parameters
        params = self.x_proj(x)  # (batch, seq, dt_rank + 2*d_state)
        
        # Split into Δ, B, C
        delta = params[:, :, :self.dt_rank]
        B = params[:, :, self.dt_rank:self.dt_rank + self.d_state]
        C = params[:, :, self.dt_rank + self.d_state:]
        
        # Softplus for positive Δ
        delta = F.softplus(delta)
        
        # Discretize with selective parameters
        A_bar = torch.exp(self.A_log * delta)
        
        # Parallel scan
        h = parallel_scan(A_bar, B_bar)
        
        return C @ h
```

## The Complete Mamba Block

Mamba adds several modern architecture tricks:

1. **Pre-norm**: LayerNorm before computation
2. **Expansion**: $d_{model} \rightarrow 2 \times d_{model}$
3. **SiLU Gate**: Gating activation
4. **Residual**: Skip connection

```python
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=128, expand=2):
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, expand * d_model * 2)
        self.ssm = SelectiveSSM(expand * d_model, d_state)
        self.out_proj = nn.Linear(expand * d_model, d_model)
    
    def forward(self, x):
        # Pre-norm
        x = self.norm(x)
        
        # Split and gate
        x, gate = self.in_proj(x).chunk(2, dim=-1)
        gate = F.silu(gate)
        
        # SSM with gating
        y = self.ssm(gate) * gate
        
        # Output and residual
        return self.out_proj(y) + x
```

## Comparison

| Feature | Basic SSM | Selective SSM (Mamba) |
|---------|-----------|----------------------|
| A matrix | Fixed | Fixed |
| B matrix | Fixed | **Input-dependent** |
| C matrix | Fixed | **Input-dependent** |
| Δ | Fixed | **Input-dependent** |
| Selection | ❌ | ✅ |

## Summary

In this step, we learned:

1. **Selection problem**: Basic SSMs can't adapt to content
2. **Solution**: Make B, C, Δ input-dependent
3. **Softplus**: Ensures positive step sizes
4. **Full block**: Adding gating, norm, residuals

## Next Step

In the next step, we'll build a complete **language model** and train it on text data!
