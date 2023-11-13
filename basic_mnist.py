import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
device = 'cpu'

#hyper parameters
input_size = 784 #28x28 pictures, stored as tensors
hidden_size = 100 #arbitrary
num_classes = 10 #10 digits
num_epochs = 3
batch_size = 100
learning_rate = 0.001
#dataset
train_data = torchvision.datasets.MNIST(root='./data', train = True, transform= transforms.ToTensor(), download=True)
test_data = torchvision.datasets.MNIST(root='./data', train = False, transform= transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_data,batch_size=batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,batch_size=batch_size, shuffle = False)

# examples = iter(train_loader)
# samples, labels = next(examples)
# print(samples.shape, labels.shape)
# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.imshow(samples[i][0],cmap='gray')
#     plt.show()

class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size,num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes)

#loss & optimizer
criterion = nn.CrossEntropyLoss() # it has softmax!
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

#training loop
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i,(images, labels) in enumerate(train_loader):
        #reshape tensors
        images = images.reshape(-1, 784).to(device)
        labels = labels.to(device)

        #forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        #backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0 :
            print(f'Epoch {epoch + 1}/{num_epochs}, step {i+1}/{n_total_steps}. Loss = {loss.item():.4f}.')

#Evaluation
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 784).to(device)
        labels = labels.to(device)
        outputs = model(images)
        #true labels
        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions==labels).sum().item()        
    accuracy = 100.0 * n_correct/n_samples
    print(f'Accuracy: {accuracy:.4f}%')