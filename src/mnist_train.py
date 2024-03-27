import datetime
import time

import numpy as np
from dateutil.parser import parse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from scipy.interpolate import make_interp_spline

from MLP_DNN import NeuralNet
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

input_size = 784
hidden_size = 500
output_size = 10
num_epochs = 20
batch_size = 100
learning_rate = 0.001


train_dataset = torchvision.datasets.MNIST(root='./datam',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='./datam',
                                          train=False,
                                          transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

model = NeuralNet(input_size, hidden_size, output_size).to(device)  
print(model)
params = list(model.parameters())
print(len(params))
print(params[0].size())

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)  


acc = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.reshape(-1, 28 * 28).to(device)  
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))


    with torch.no_grad():
        correct = 0
        total = 0
        test_imgset_one = []
        for images, labels in test_loader:
            test_imgset_one = images
            images = images.reshape(-1, 28 * 28)
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)  
            correct += (predicted == labels).sum().item()  ##
        acc_this = 100 * correct / total
        print('Accuracy of the network on the 10000 test images: {} %，total test case number:{}'.format(acc_this, total))
        acc.append(acc_this)

epoch_num = np.arange(1, num_epochs+1, 1)#np.linspace(1, epoch, 1)

plt.figure()
plt.plot(epoch_num,acc, 'b', label='acc')
plt.title("Test acc in different EPOCH (EPOCH-Acc plot)")
plt.ylabel('acc %')
plt.xlabel('epoch_num')
plt.show()


x_smooth = np.linspace(epoch_num.min(), epoch_num.max(), 300)
y_smooth = make_interp_spline(epoch_num, acc)(x_smooth)
plt.plot(x_smooth, y_smooth)
plt.scatter(epoch_num, acc, marker='o')
plt.show()



torch.save(model.state_dict(), 'model.ckpt')