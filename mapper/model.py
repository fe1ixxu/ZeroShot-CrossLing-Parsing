import torch
import torch.nn as nn

class Aligner(nn.Module):
    def __init__(self, size):
        super(Aligner, self).__init__()
        self.W = nn.Parameter(torch.eye(size))
        self.sig = nn.Linear(3,1)
    def forward(self, x):
        return (self.W @ x.t()).t()