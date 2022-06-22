#!/usr/bin/env python
# author = 'ZZH'
# time = 2022/5/21
# project = test
import numpy as np
import torch.utils.data
import torchvision.datasets
from matplotlib import pyplot as plt
from torchvision import transforms
from resnet import ResNet, BasicBlock
from Dataloader.cat_dog_dataloader import Cat_Dog_Dataset


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
batch_size = 4
# testset = torchvision.datasets.CIFAR10(root='../dataset', train=False, download=True, transform=transform)
testset = Cat_Dog_Dataset(transformer=transform, train=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

net = ResNet(BasicBlock, [5, 5, 5], num_classes=2)
# PATH = './cifar_net.pth'
PATH = './cat_dog.pth'

net.load_state_dict(torch.load(PATH))
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
classes = ('cat', 'dog')

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
