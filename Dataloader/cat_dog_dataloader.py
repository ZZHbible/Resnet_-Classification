#!/usr/bin/env python
# author = 'ZZH'
# time = 2022/5/21
# project = cat_dog_dataloader
import os

import cv2
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# datasets https://www.kaggle.com/competitions/dogs-vs-cats
class Cat_Dog_Dataset(Dataset):
    def __init__(self, transformer=None, train=True):
        super(Cat_Dog_Dataset, self).__init__()
        if train:
            self.img_path = 'D:\Py_learn\dog_cat\data\\train1'  # 先都设成小的数据集，方便调试
        else:
            self.img_path = 'D:\Py_learn\dog_cat\data\\train1'
        self.images = []
        self.labels = []
        for file in os.listdir(self.img_path):
            img = cv2.imread(os.path.join(self.img_path, file))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            # img_tensor = transforms.ToTensor()(img)
            label = file.split('.')[0]
            if transformer:
                self.images.append(transformer(img))
            else:
                self.images.append(img)
            # self.labels.append(torch.tensor(0) if label == 'cat' else torch.tensor(1))
            self.labels.append(0 if label == 'cat' else 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

# train_dataset = Cat_Dog_Dataset()
# batch_size = 5
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#
# for i, data in enumerate(train_dataloader):
#     img, label = data
#     for j in range(len(img)):
#         print(type(img[j]))
#     # print('epoch {} image {} label {}'.format(i, img, label))
