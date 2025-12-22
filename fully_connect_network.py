import numpy as np
import torch
from torch import nn
import pandas as pd

class FullConnect(nn.Module):
    def __init__(self):
        super(FullConnect, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.linear1 = nn.Linear(5, 32)  # 增加隐藏层神经元数量
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, 32)
        self.linear4 = nn.Linear(32, 2)
        # 不需要softmax，CrossEntropyLoss会自动处理

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.linear4(x)
        return x  # 直接返回logits，不需要softmax

