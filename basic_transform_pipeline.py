# import torch
# import torchvision


# dataset = torchvision.datasets.MNIST(root='./data', transform=torchvision.transforms.ToTensor())


import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

#We implement a wraper that we can pass data to 
class ExampleDataset(Dataset):
    def __init__(self, transform=None):
        data=np.loadtxt('wine.csv',delimiter=',', dtype=np.float32,skiprows=1)
        self.X = data[:,1:]
        self.y = data[:,[0]]
        self.n_samples = data.shape[0]
        self.transform = transform

    def __getitem__(self,index):
        sample = self.X[index],self.y[index]
        if self.transform:
            sample=self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples

class ToTensor():
    def __call__(self, sample):
        inputs, target = sample
        return torch.from_numpy(inputs), torch.from_numpy(target)

class MulTransform():
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, sample):
        inputs, target = sample
        inputs*= self.factor
        return inputs, target

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])

dataset = ExampleDataset(transform=composed)
first = dataset[0]
features, labels = first
print(type(features), type(labels))


