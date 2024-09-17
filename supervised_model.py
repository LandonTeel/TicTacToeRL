import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Model):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
    
    def forward(self, x) -> torch.Tensor:
        out = self.layers(x)
        return out

def train():
    model = Model().to(device)



if __name__ == "__main__":
    main()

