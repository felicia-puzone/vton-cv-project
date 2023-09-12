import torch
import torch.nn as nn


class SimilarityNet(nn.Module):
    def __init__(self, in_features: int, hidden_units: int):
        super().__init__()

        self.sequential = nn.Sequential(
                nn.BatchNorm1d(num_features=in_features),
                nn.Linear(in_features=in_features, out_features=hidden_units, bias=True),
                nn.Dropout(p=0.5),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=hidden_units),
                nn.Linear(in_features=hidden_units, out_features=hidden_units, bias=True),
                nn.Dropout(p=0.5),
                nn.ReLU(),
                nn.Linear(in_features=hidden_units, out_features=2, bias=True)
        )

    def forward(self, x):
        return self.sequential(x)
