import torch
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
times = 100
res = []
for i in range(times):
    a = torch.randn((1000, 1000))
    b = torch.randn((1000, 1000))
    c = torch.matmul(a, b)
    res.append(c)
