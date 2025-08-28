import torch, math
import torch.nn as nn
from model.transformer.Softmax import softmax
from model.transformer.RoPE import RoPE
from model.transformer.Linear import LinearModule

class ScaledDotProductAttention(nn.Module): 
    '''
    
    Attention(Q, K, V ) = softmax (QK.T/ √dk)V
   
    Q K.T =   |  ------ q_1 ----  |      |  |   |           |        |  
              |  ------ q_2 ----  |      |  |   |           |        |      
                         .               |  k_1  k_2 . . . k_seqlen  |     
                         .               |  |   |           |        |
              |  -- q_seq_len --  |      |  |   |           |        |
                                               
    
     
     Q Kᵀ =     |   q₁·k₁    q₁·k₂   ...   q₁·kₙ |        
                |   q₂·k₁    q₂·k₂   ...   q₂·kₙ |       
                |    ...       ...   ...    ...  |
                |   qₙ·k₁    qₙ·k₂   ...   qₙ·kₙ  |    
                
    Q Kᵀ shape will be [batch_size, seq_len, seq_len]
                
    softmax((QK.T/ √dk) will be a matrix of attention weights of shape [b_size, seq_len, seq_len]
    The subsequent mask's shape is [seq_len, seq_len]. 
    '''
    def __init__(self): 
        super().__init__()
     
    def forward(self, queries : torch.Tensor, keys : torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor: 
        # queries: [batch_size, ..., seq_len, dk]
        # keys:  [batch_size, ..., seq_len, dk]
        # values: [batch_size, ..., dv]
        dk = keys.shape[-1] # by using -1, we apply attention to the last dimension without needing to know if your input tensor is 3D or 4D.
        scaled_QK =  queries @ torch.transpose(keys,-1, -2)/ math.sqrt(dk) # [batch_size, seq_len, seq_len]
       
        filled_mask = torch.zeros_like(mask, dtype=torch.float)  
        filled_mask = filled_mask.masked_fill(~mask, float('-inf'))
        
        scaled_QK = scaled_QK + filled_mask      
        scores = softmax(scaled_QK, -1) # shape [batch_size, seq_len, seq_len]
        # now we need to multiply with values [batch_size, ..., seq_len, dv]
        # the output of the scaled_dot_product_attention will be [batch_size, ..., seq_len, dv]
        return torch.matmul(scores, values)
 
class MultiHeadSelfAttention(nn.Module): 
    '''
    MultiHead(Q, K, V ) = Concat(head1, . . . , headh) with headi = Attention(Qi, Ki, Vi) 
    MultiHeadSelfAttention(x) = WO MultiHead(WQ x, WK x, WV x) with Wo, WQ, QK, and WV being learnable parameters
    
    How does it work ? 
    1- what does the hid_dim/ d_model correspond to ? 
        The d_model corresponds to the embedding dimension. In Llama 3.1 8B, the hidden size is d_model = 4096.
        This means a token like "the" is embedded (via a lookup table) as a tensor of shape [1, 4096].

    2- what is multihead self attention ? 
     It is the process by which attention is computed in multiple smaller batches called heads.
     The per-head dimension is head_dim = d_model / num_heads.

 
     Q1 =     |  ***--- q_1 ----  |       
              |  ***--- q_2 ----  |     
                 ***     .               
                 ***    .              
              |  ***q_seq_len --  |    
                (first bloc)   
                  (head1)           -- same for k1, v1 --               
             
     Q, K, and V are split across num_heads blocks to compute each head.
        For example, with d_model = 4096 and num_heads = 8, each head has head_dim = 4096 / 8 = 512. 
        For head 1 (conceptually using the first block of 512 features):
            Q1 shape: [batch_size, seq_len, head_dim]        # = [B, T, 512]
            K1 shape: [batch_size, seq_len, head_dim]        # = [B, T, 512]
            Scores:  Q1 @ K1^T / sqrt(head_dim) → [B, T, T]
            Softmax over the last dim → [B, T, T]
            Head1 output: softmax(scores) @ V1 → [B, T, head_dim]  # = [B, T, 512]

    Concatenate all heads along the last dimension to get [B, T, d_model],
    then apply W_O to produce the final output.

    '''
    def __init__(self, d_model: int, num_heads: int, use_rope: bool = False, max_seq_len: int | None = None, theta: float | None = None, token_positions: torch.Tensor | None = None):
        '''
         Args: 
            - d_model: int Dimensionality of the Transformer block inputs.
            - num_heads: int Number of heads to use in multi-head self-attention.
            - d_k, and d_v: int Dimensions of keys and values. 
            - token_positions : specify the token position along the sequence dimension.
            - use_rope: bool Boolean, if set to True, the mha should appply Rope to keys and values.
        '''
        super().__init__()
                
        self.d_model = d_model
        self.num_heads = num_heads
        assert self.d_model % self.num_heads == 0
        self.d_k = self.d_model // self.num_heads
        self.d_v = self.d_k
        self.token_positions = token_positions
        self.q_proj = LinearModule(self.d_model, self.num_heads * self.d_k) # Wq
        self.k_proj = LinearModule(self.d_model, self.num_heads * self.d_k) # Wk
        self.v_proj = LinearModule(self.d_model, self.num_heads * self.d_v) # Wv
        self.output_proj = LinearModule(self.d_model, self.d_model)         # WO
        
        self.use_rope = use_rope
        self.self_attn = ScaledDotProductAttention()
        self.rope = RoPE(theta = theta, d_k = self.d_model // self.num_heads, max_seq_len= max_seq_len) if (use_rope) else None
             
    def forward(self, x: torch.Tensor)-> torch.Tensor: 
        '''
        Input x : [b_size, seq_len, d_model]
        '''
        batch_size, seq_len, d_model = x.shape
       
        # STEP 1: for each head, compute Q,K,V, and split into heads: 
        # -> (batch_size, num_heads, seq_len, d_model)
        Q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(-3, -2) 
        K = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(-3, -2) 
        V = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(-3, -2)  

        if self.use_rope:
            if self.token_positions is None: 
                token_positions = torch.arange(seq_len)
            Q = self.rope(Q, self.token_positions)
            K = self.rope(K, self.token_positions)

        # STEP 2: compute the causal mask
        mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=0).T.bool() # [seq_len, seq_len]

        # STEP 3 : use self mask attention to compute th 
        out = self.self_attn(Q, K, V, mask)
        out = out.transpose(-2, -3).reshape(batch_size, seq_len, self.num_heads * self.d_k)
    
        return self.output_proj(out)
