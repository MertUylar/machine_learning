
import torch

rand1=torch.rand(2,2)
rand2=torch.rand(2,2)

print(rand1,rand2)
print(rand1+rand2)
print(torch.add(rand1,rand2))
print(rand1*rand2)
print(rand1.mul(rand2))
print(rand1.mul(5))

rand1=rand1.cuda()  # gpu is faster than cpu
rand2=rand2.cuda()