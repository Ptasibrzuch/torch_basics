import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

#We implement a wraper that we can pass data to 
class ExampleDataset(Dataset):
    def __init__(self):
        data=np.loadtxt('wine.csv',delimiter=',', dtype=np.float32,skiprows=1)
        self.X = torch.from_numpy(data[:,1:])
        self.y = torch.from_numpy(data[:,[0]])
        self.n_samples = data.shape[0]

    def __getitem__(self,index):
        return self.X[index],self.y[index]

    def __len__(self):
        return self.n_samples

dataset = ExampleDataset()
dataloader = DataLoader(dataset,batch_size=4,shuffle=True)#, num_workers=2)
num_epoch = 2
total_samples = len(dataset)
num_iter = math.ceil(total_samples/4)
for epoch in range(num_epoch):
    for i,(inputs,labels) in enumerate(dataloader):
        if (i+1) % 5 == 0 :
            print(f'Epoch : {epoch +1}/{num_epoch}, step: {i+1}/{num_iter}. Inputs: {inputs.shape}.')