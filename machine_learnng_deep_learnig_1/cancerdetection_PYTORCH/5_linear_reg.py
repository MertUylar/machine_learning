# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 15:36:59 2022

@author: ozkan
"""

from builtins import property

import torch
import numpy

# x= torch.tensor(3.)
# w = torch.tensor(4., requires_grad=True)
# b = torch.tensor(5., requires_grad=True)
#
#
# print(x)
# print(w)
# print(b)
#
# y=w*x + b
# y.backward()
#
# print(y)
#
# print('dy/dw', w.grad)
# print('dy/dx', b.grad)

inputs = numpy.array([[73, 67, 43],
                      [91, 88, 64],
                      [87, 134, 58],
                      [102, 43, 37],
                      [69, 96, 70]], dtype='float32')

targets = numpy.array([[56,70],
                       [81, 101],
                       [119, 133],
                       [22, 37],
                       [103, 119]], dtype='float32')

inputs_torch= torch.from_numpy(inputs)
targets_torch= torch.from_numpy(targets)

# print(inputs_torch , targets_torch)

w = torch.rand(2, 3, requires_grad = True)
b = torch.rand(2, requires_grad = True)

# print(w, b)

def model(x):
    return x @ w.t() + b

def mse(real , pred):
    diff = real - pred
    return torch.sum(diff*diff)/diff.numel()



pred =model(inputs_torch)
loss = mse(targets_torch, pred)
print("First Loss  :", loss)

loss.backward()
# print(b)
# print(b.grad)

# print(w)
# print(w.grad)
for i in range(1000):
   pred =model(inputs_torch)
   loss = mse(targets_torch, pred) 
   loss.backward()
   with torch.no_grad():
     w -= w - w.grad*1e-5
     b -= b - b.grad*1e-5
     w.grad.zero_()
     b.grad.zero_()
  


pred = model(inputs_torch)
loss = mse(targets_torch, pred)

print("Second Loss :", loss)

print(targets)
print(pred)