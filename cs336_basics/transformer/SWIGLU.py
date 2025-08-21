import torch
import torch.nn as nn
import math 
from cs336_basics.transformer.Linear import LinearModule

class SWIGLU(nn.Module): 
    '''The Swiglu feed-forward network is composed of a SILU activation function and a GLU'''
    def __init__(self, d_model, dff): 
        super().__init__()
        self.d_model = d_model
        self.dff = dff
        self.w1 = LinearModule(d_model, dff)    
        self.w3 = LinearModule(d_model, dff)   
        self.w2 = LinearModule(dff, d_model)   
        
    def SILU(self, x): 
        return x * torch.sigmoid(x)
    
    def forward(self, x): 
        return self.w2(self.SILU(self.w1(x)) * self.w3(x))      # [b_size, seq_len, d_model]