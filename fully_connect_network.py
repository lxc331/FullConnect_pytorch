import numpy as np
import torch
from torch import nn
import pandas as pd

class FullConnect(nn.Module):
    def __init__(self):
        super(FullConnect, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.linear1 = nn.Linear(5, 25)  # 增加隐藏层神经元数量
        self.linear2 = nn.Linear(25, 50)
        self.linear3 = nn.Linear(50, 25)
        self.linear4 = nn.Linear(25, 2)
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

