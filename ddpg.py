import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3,3)

    def forward(self,x):
        return torch.tanh(self.fc(x))
