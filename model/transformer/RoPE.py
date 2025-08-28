import torch 
import torch.nn as nn 
import math 

class RoPE(nn.Module): 
    '''
    Transformers are inherently positional invariant. Therefore, we need to inject positions.
    We are doing it in this implementation thanks to Rope (Relative Positional Embeddings). 
    Interestingly, this layer has no learnable parameter. Therefore, as the parameters for this module are non-learnable,
    and we don’t want it to receive gradients or participate in optimization, we will use register_buffer. 
    Sine and Cosine values are fixed and not nn.Parameter. 
    I find this concept more challenging to understand: read "https://arxiv.org/pdf/2104.09864".
    '''
    
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None): 
        '''
        Args: 
            - theta: float Θ value for the RoPE
            - d_k: int dimension of query and key vectors
            - max_seq_len: int Maximum sequence length that will be inputted 
            - device: torch.device | None = None Device to store the buffer on
        '''
        super().__init__()
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        k = torch.arange(d_k // 2)          
        inv_freq = theta ** (-2 * k / d_k)                                
        i = torch.arange(max_seq_len).unsqueeze(1)  
        angles = i * inv_freq.unsqueeze(0)              
                     
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.device = device
        
       
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor: 
        '''
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
        '''   

        seq_len = x.shape[-2] 
     
        if token_positions==None: 
            token_positions = torch.arange(seq_len)
       
        cos_pos = self.cos[token_positions]
        sin_pos = self.sin[token_positions]
        
        x_even = x[..., 0::2]                                                         
        x_odd  = x[..., 1::2]          

        # Apply 2x2 rotation per pair
        y_even = x_even * cos_pos - x_odd * sin_pos
        y_odd  = x_even * sin_pos + x_odd * cos_pos

        rotated_tensor = torch.empty_like(x)
        rotated_tensor[..., 0::2] = y_even
        rotated_tensor[..., 1::2] = y_odd
        
        return rotated_tensor
