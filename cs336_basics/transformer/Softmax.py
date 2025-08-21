import torch
import torch.nn as nn


def softmax(input: torch.Tensor, dim_i : int) -> torch.Tensor:
   
    '''This function applies the softmax to the i-th dimension of the input tensor.
    The output tensor should have the same shape as the input tensor.'''
    
    numerator = torch.exp(input- torch.max(input, dim_i, keepdim=True).values)
    denominator = torch.sum(torch.exp(input - torch.max(input, dim_i, keepdim=True).values), dim=dim_i, keepdim=True)
    return numerator/ denominator