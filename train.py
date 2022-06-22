#!/usr/bin/env python
# author = 'ZZH'
# time = 2022/5/21
# project = train
import torch.utils.data
import torchvision.datasets
from torch import nn
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from resnet import ResNet, BasicBlock
from Dataloader.cat_dog_dataloader import Cat_Dog_Dataset

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
batch_size = 4
# trainset = torchvision.datasets.CIFAR10(root='../dataset', train=True, download=True, transform=transform)
trainset = Cat_Dog_Dataset(transformer=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
# testset = torchvision.datasets.CIFAR10(root='../dataset', train=False, download=True, transform=transform)
testset = Cat_Dog_Dataset(transformer=transform, train=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import torch.optim as optim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = ResNet(BasicBlock, [5, 5, 5], num_classes=2)
net.to(device)
from torchsummary import summary

summary(net, (3, 224, 224))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
pbar = tqdm(range(3))
for epoch in pbar:
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # dataloader 返回tensor类型
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 0:
            print('{} loss : {}'.format(epoch + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished training')

PATH = './cat_dog.pth'
torch.save(net.state_dict(), PATH)

# net.load_state_dict(torch.load(PATH))
# dataiter = iter(testloader)
# images, labels = dataiter.next()
# outputs = net(images)
# print(outputs)
