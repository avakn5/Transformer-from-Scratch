# A Transformer Implementation from Scratch

This repository contains my implementation of a Transformer from scratch in Pytorch.

As part of a self-study, I followed CS336 — Spring 2025 Assignment 1 instructions (*cs336_spring2025_assignment1_basics.pdf*). Look at section 3 for Transformer instructions.

Every module—linear layers, attention, normalization, and feed-forward—is hand-built using PyTorch nn.Module, without relying on high-level shortcuts.


__The codebase includes extensive inline comments to aid understanding of the Transformer implementation.__

![Transformer Architecture](cs336_basics/READMEfigure.jpg)


### Quick Start

Setup 

```
git clone https://github.com/avakn5/cs336_basics && cd cs336_basics
pip install -e .
```

Run the transformer
```
import torch
from cs336_basics.transformer.Transformer import TransformerLM

model = TransformerLM(
    vocab_size=10_000,
    context_length=16,
    num_layers=3,
    theta=10_000.0,
    d_model=64,
    d_ff=128,
    num_heads=4,
)
x = torch.randint(0, 10_000, (2, 12))   # [batch=2, seq=12]
logits = model(x)                       # [2, 12, vocab_size]
```

### STEPS to the Transformer implementation from scratch (sequential implementation):

* 1- Implement a linear module 
* 2- Implement an embedding module 
* 3- Implement a RMSNorm module
* 4- Implement a SWIGLU
* 5- Implement RoPE(Relative Positional Embeddings)
* 6- Implement softmax
* 7- Implement Self-Attention
* 8- Implement Multi-Head Attention
* 9- Implement Transformer block
* 10- Implement Transformer Language Model


### Implemented Features:

* Linear Module
* Embedding Module
* RMSNorm Module
* SWIGLU
* RoPE
* Softmax
* Scaled Dot Product Attention
* Causal MultiHead Self Attention

* Passing all 39/39 tests.

### Next Steps: 

* Add functionnality for config, reading hyper-parameters from a config.
* Load Llama 3.1 8B parameters checkpoints.
* Generate tokens from llama 3.1 8B.
