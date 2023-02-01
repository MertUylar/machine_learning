import torch
import torch.nn as nn
from sklearn.datasets import  load_breast_cancer #datayı alabilmek için

device=torch.device("cpu")

#hyper parameter

imput_size =30
hidden_size =500
num_classes=2
num_epoch=100

learning_rate=1e-8

girdi,cikti= load_breast_cancer(return_X_y=True)

print(girdi)
print(girdi.shape)
print(cikti)
print(cikti.shape)

#Numpy'dan pytorch array 

train_input =torch.from_numpy(girdi).float()
train_output = torch.from_numpy(cikti).long()

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.lrelu = nn.LeakyReLU(negative_slope=0.02)
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    def forward(self, inputt):
        outfc1 = self.fc1(inputt)
        outfc1lrelu = self.lrelu(outfc1)
        out = self.fc2(outfc1lrelu)
        return out
         
model= NeuralNet(imput_size, hidden_size, num_classes)


# Loss and optimizer
lossf = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
         
        
for epoch in range(num_epoch):
    outputs=model(train_input)
    loss=lossf(outputs,train_output)     
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print ('Epoch [{}/{}],Loss: {:.4f}' .format(epoch+1, num_epoch, loss.item()))