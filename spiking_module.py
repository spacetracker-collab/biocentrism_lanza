
import torch
import torch.nn as nn

class LIFNeuron(nn.Module):
    def __init__(self, threshold=1.0, decay=0.9):
        super().__init__()
        self.threshold = threshold
        self.decay = decay

    def forward(self, input_current, membrane):
        membrane = self.decay * membrane + input_current
        spike = (membrane > self.threshold).float()
        membrane = membrane * (1 - spike)
        return spike, membrane
