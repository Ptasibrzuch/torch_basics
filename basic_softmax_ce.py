import torch
import torch.nn as nn

#Calculating Softmax layer output
x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)
print(outputs)

loss = nn.CrossEntropyLoss()
 #instead of one sample as below, we can input tensors
Y = torch.tensor([0])
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
Y_pred_bad = torch.tensor([[0.1, 2.0, 0.3]])
#calculating Cross Entropy Loss
#CEL applies logsoftmax and NLL loss,
#  we cannot add softmax here, also we can one-hot encode Y
#y_pred musi byc logitami(P), nie wartosciami klas
l1 = loss(Y_pred_good,Y)
l2 = loss(Y_pred_bad, Y)
print(l1.item())
print(l2.item())
#predicted classes
_, prediction1 = torch.max(Y_pred_good, 1) # we unpack only what we need
_, prediction2 = torch.max(Y_pred_bad, 1)

print(prediction1)
print(prediction2)
