from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 50)  # 入力層から隠れ層へ
        self.l2 = nn.Linear(50, 10)  # 隠れ層から出力層へ

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # テンソルのリサイズ: (N, 1, 28, 28) --> (N, 784)
        x = self.l1(x)
        x = self.l2(x)
        return x
