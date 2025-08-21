import torch
import torch.nn as nn
from cs336_basics.transformer.RMSNorm import RMSNorm
from cs336_basics.transformer.Attention import MultiHeadSelfAttention
from cs336_basics.transformer.SWIGLU import SWIGLU
from cs336_basics.transformer.RoPE import RoPE
from cs336_basics.transformer.Embedding import Embedding
from cs336_basics.transformer.Linear import LinearModule
from cs336_basics.transformer.Softmax import softmax

class TransformerBlock(nn.Module):
 
    def __init__(self, d_model: int, num_heads: int, d_ff: int, use_rope: bool = True, max_seq_len: int = None, theta: int= None): 
        '''
        Args: 
            - d_model: int Dimensionality of the Transformer block inputs.
            - num_heads: int Number of heads to use in multi-head self-attention.
            - d_ff: int Dimensionality of the position-wise feed-forward inner layer.
            - max_seq_len: int Maximum sequence length that will be inputted.
            - theta: int  Θ value for the RoPE.
        '''
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.ln1 = RMSNorm(self.d_model)
        self.attn = MultiHeadSelfAttention(self.d_model, num_heads, use_rope =use_rope, max_seq_len= max_seq_len, theta = theta)
        self.ln2 = RMSNorm(self.d_model)
        self.ffn = SWIGLU(self.d_model, self.d_ff)  
        
    def forward(self, x: torch.Tensor)-> torch.Tensor:
        '''
        Input tensor shape is [batch_size, seq_len, d_model] e.g. [4,12,64]
        Out tensor shape is [batch_size, seq_len, d_model].
        '''
        # # STEP 1: first sub-layer
        out = x +  self.attn(self.ln1(x))
        
        # STEP 2: second sub-layer
        out = out + self.ffn(self.ln2(out)) 
        return out
       
class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, theta: int, d_model: int, d_ff: int, num_heads: int): 
        '''
        Args: 
            - vocab_size: int The size of the vocabulary, necessary for determining the dimensionality of the token embedding matrix.
            - context_length: int The maximum context length, necessary for determining the dimensionality of the position embedding matrix.
            - num_layers: int The number of Transformer blocks to use.
            - theta: int  Θ value for the RoPE.
            - d_model: int Dimensionality of the Transformer block inputs.
            - num_heads: int Number of heads to use in multi-head self-attention.
            - d_ff: int Dimensionality of the position-wise feed-forward inner layer.
        '''
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.theta = theta
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        
        self.token_embeddings = Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model=self.d_model,
                                        num_heads= num_heads, 
                                        d_ff = d_ff, 
                                        use_rope= True,
                                        max_seq_len= self.context_length, 
                                        theta = theta) 
                       for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = LinearModule(self.d_model, self.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_embed = self.token_embeddings(x)
        
        for block in self.layers: 
            token_embed = block(token_embed)
            
        out = self.ln_final(token_embed)
        out = self.lm_head(out)
        return out # output probabilities
