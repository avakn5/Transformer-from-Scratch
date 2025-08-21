import torch
import math 
import torch.nn as nn 
from einops import einsum 

class LinearModule(nn.Module): # we subclass nn.Module
    ''' implementing the Linear Module.'''
    
    def __init__(self, in_features: int, out_features:int, device=None, dtype=None): 
        '''
        Args: 
            - in_features: int final dimension of the input
            - out_features: int final dimension of the output
            - device: torch.device | None = None Device to store the parameters on 
            - dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__() # superclass constructor
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.sigma = math.sqrt(2/(in_features + out_features)) # the sigma value was suggested by CS336 instructions.
        self.weight = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight, mean= 0 , std = self.sigma , a= -3*self.sigma, b= 3*self.sigma)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        '''
        x.shape : [1, in_features]
        W.shape : [out_features, in_features]
        
        Returns: 
            - x @ W.T.shape : [1, out_features]
        '''        
        # equivalent to computing x @ W.T 
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out") # ... enables to use any batch_size or seq length.
        # d_in is x.shape, the first element of the einsum. d_out d_in is the dimension of W.
        