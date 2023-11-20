"""This script implements a low-rank linear layer."""
import torch 
import torch.nn as nn 

from .hypercomplex.inits import glorot_uniform, glorot_normal

class LowRankLinear(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int, rank: int = 1,
        bias: bool = True, w_init: str = "glorot-uniform"):
        super(LowRankLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim 
        self.rank = rank
        self.bias = bias
        self.w_init = w_init
        self.W_left = nn.Parameter(torch.Tensor(size=(input_dim, rank)), requires_grad=True)
        self.W_right = nn.Parameter(torch.Tensor(size=(rank, output_dim)), requires_grad=True)
        if bias:
            self.b = nn.Parameter(torch.Tensor(output_dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.bias:
            self.b.data = torch.zeros_like(self.b.data)
        if self.w_init == "glorot-uniform": 
            self.W_left.data = glorot_uniform(self.W_left.data) 
            self.W_right.data = glorot_uniform(self.W_right.data)          
        elif self.w_init == "glorot-normal":
            self.W_left.data = glorot_normal(self.W_left.data)
            self.W_right.data = glorot_normal(self.W_right.data)
        else:
            raise ValueError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.W_left.matmul(self.W_right)
        output = torch.matmul(input=x, other=W)
        if self.bias:
            output += self.b
        return output
