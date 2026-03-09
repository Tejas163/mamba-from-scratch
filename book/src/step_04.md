# Step 4: Complete Language Model

## Overview

Now we'll build a complete character-level language model using Mamba and train it on text data.

## Architecture

```
Input: "hello"
  ↓
Embedding (vocab_size → d_model)
  ↓
Mamba Layers (stacked blocks)
  ↓
LayerNorm
  ↓
LM Head (d_model → vocab_size)
  ↓
Output: next token logits
```

## Implementation

```python
class MambaLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=4, d_state=128):
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Weight tying
        self.lm_head.weight = self.embedding.weight
    
    def forward(self, x, targets=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                targets.view(-1)
            )
        return logits, loss
    
    @torch.no_grad()
    def generate(self, x, max_tokens):
        for _ in range(max_tokens):
            x_cond = x[:, -512:]
            logits, _ = self(x_cond)
            next_token = logits[:, -1].argmax(dim=-1)
            x = torch.cat([x, next_token.unsqueeze(0)], dim=1)
        return x
```

## Training

Standard language modeling training:
- Cross-entropy loss
- Autoregressive prediction
- Next token classification

## Summary

We've built a complete Mamba language model!

1. ✅ Token embeddings
2. ✅ Stacked Mamba blocks
3. ✅ Language model head
4. ✅ Text generation

## Next Steps

- Experiment with larger models
- Try different datasets
- Compare with Transformer LM
- Explore hybrid architectures
