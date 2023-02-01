# -*- coding: utf-8 -*-
import torch 
from torch.nn import Linear

import numpy as np

girdi=torch.rand(1)
print(girdi)

Lineer11 =Linear(in_features=1,out_features=1)
print("Ağırlık w:",Lineer11.weight)
print("Y ekseni kesit b:",Lineer11.bias)


print("Torch ile lineer:")
print(Lineer11(girdi))

print("python ile hesapladık")

print("m*x+b, m+girdi+b  w+girdi+b")

print(Lineer11.weight*girdi+Lineer11.bias) 

Lin1=Linear(in_features=1,out_features=5,bias=True)
print("Lin1",Lin1.weight)
Lin2=Linear(in_features=5,out_features=1)
print("Lİn2",Lin2.weight)
print(Lin2(Lin1(girdi)))