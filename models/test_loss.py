import torch.nn as nn
import torch


x = torch.tensor([[0.4702,0.890],
        [0.5272,0.890],
        [0.4809,0.890],
        [0.4951,0.890],
        [0.5225,0.890],
        [0.4777,0.890],
        [0.5266,0.890],
        [0.5035,0.890]])
y = torch.tensor([0, 0, 0, 0, 0, 0, 1, 0])
print(x.shape[1])
crossentropyloss = nn.CrossEntropyLoss()
loss = crossentropyloss(x,y)
loss = torch.mean(loss)
print(loss)