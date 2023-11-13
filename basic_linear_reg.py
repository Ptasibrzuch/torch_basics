import torch
import torch.nn as nn


print('------------------------- \n ----------Linear regression pyTorch----------\n------------------------- ')
#Linear regression pytorch
X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)
n_samples, n_features = X.shape
learning_rate = 0.01
iterations = 100
loss_func = nn.MSELoss()
model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(f'Pred before training f(5)= {model(torch.tensor([5],dtype=torch.float32)).item()}.')
for epoch in range(iterations):
    y_pred = model(X)
    loss = loss_func(Y, y_pred)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch%10==0:
        [w,b] = model.parameters()
        print(f'epoch: {epoch+1}: w= {w[0][0].item()}, loss= {loss}.')

print(f'Pred after training f(5)= {model(torch.tensor([5],dtype=torch.float32)).item()}.')