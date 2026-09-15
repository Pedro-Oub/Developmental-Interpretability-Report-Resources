"""
Phase 1: Model definition.

Small enough model, so that weights can even be manually checked.

"""

import torch
import torch.nn as nn

N_CLASSES = 11 # N of different identifiable colors
HIDDEN_SIZE = 16

# Main model definition:
# 3 -> 16(ReLU) -> 11, 251 parameters

class ColorNet(nn.Module):
    def __init__(self, hidden_size: int = HIDDEN_SIZE, n_classes: int = N_CLASSES):
        super().__init__()
        self.fc1 = nn.Linear(3, hidden_size)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x, return_hidden: bool = False):
        h = self.act1(self.fc1(x))
        logits = self.fc2(h)
        if return_hidden:
            return logits, h
        return logits


if __name__ == "__main__":
    torch.manual_seed(0)
    model = ColorNet()
    dummy = torch.rand(5, 3)
    logits, hidden = model(dummy, return_hidden=True)
    print("Input shape:", dummy.shape)
    print("Hidden shape:", hidden.shape)
    print("Logits shape:", logits.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total params:", total_params)

# Program by Pedro Oubiña S. 2026