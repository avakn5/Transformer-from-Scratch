import torch.nn as nn
import torch

class Embedding(nn.Module): 
    '''
    The embedding module performs an embedding lookup. 
    maps integer token IDs into a vector space which dimension is : d_model (=d_embedding). 
    
    '''
    
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None): 
        '''
        Args: 
            - num_embeddings: int Size of the vocabulary. 
            - embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
            - device: torch.device | None = None Device to store the parameters on 
            - dtype: torch.dtype | None = None Data type of the parameters
        '''
        super().__init__() 
        self.num_embeddings = num_embeddings
        # Create learnable embedding matrix called weight: shape (vocab_size, d_model): 
        self.weight = nn.Parameter(torch.zeros(self.num_embeddings, embedding_dim, device=device, dtype=dtype)) 
        torch.nn.init.trunc_normal_(self.weight, mean = 0, std = 1, a= -3, b= 3)
        self.device = device 
        self.dtype = dtype
     
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:       
        ''' 
        The forward methods selects the embedding vector for each token ID by indexing into an embedding matrix of shape (vocab_size, d_model).
        To do so, it uses a torch.LongTensor of token IDs with shape (batch_size, sequence_length).
        '''
        return self.weight[token_ids]
