import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    '''Implement Root Mean Square Layer Normalization'''
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        '''
        Args: 
            - d_model: int Hidden dimension of the model
            - eps: float = 1e-5 Epsilon value for numerical stability
            - device: torch.device | None = None Device to store the parameters on 
            - dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype)) #learnable parameter gi, there are d_model such parameters.
        # no need to use torch.nn.init.trunc_normal_ which is applicable in the transformer concept only for embedding matrices, and linear weight.
        self.device = device
        self.dtype = dtype
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        '''
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape
        ''' 
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        scaling = torch.sqrt(torch.mean(x**2, dim=2, keepdim=True) + self.eps)
        result =  (1/scaling)* x * self.weight
        # Return the result in the original dtype
        return result.to(in_dtype)