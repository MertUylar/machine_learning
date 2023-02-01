import  torch
from sklearn.datasets import load_boston
x=torch.rand(10)
print(x)
print(x.size)
print(x.size())

temp=torch.FloatTensor([23,24,25,26,27,112,225])
print(temp)
print(temp.size())

bostonhouses=load_boston()
print(bostonhouses)

bostonhouses =torch.from_numpy(bostonhouses.data) #alınan data yı torch array'e çevirir
print(bostonhouses)

print("---")

print(bostonhouses.size())

print("---")

print(bostonhouses[0:2])