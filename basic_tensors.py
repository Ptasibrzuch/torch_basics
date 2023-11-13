import torch
import numpy as np

# x = torch.rand(3,3, requires_grad=True)
# print(x)
# y = x*x+2
# print(y)
# z = y*x + x+ 3
# print(z)
# q = z.mean()
# print(q)
# # t = torch.tensor([1,2,3], dtype=torch.float16)
# q.backward()
# print(x.grad)

# #problem statement
# x = torch.tensor(1.0)
# y = torch.tensor(2.0)
# w = torch.tensor(1.0, requires_grad=True)
# # forward pass 
# y_hat = x*w
# loss = (y - y_hat)**2
# print(loss)
# loss.backward()
# grad = w.grad
# print(f"Gradient: {grad}.")

#Linear regression manual
print('------------------------- \n ----------Linear regression MANUAL----------\n------------------------- ')
X = np.array([1,2,3,4], dtype=np.float16)
Y = np.array([2,4,6,8], dtype=np.float16)
w =0.0

def forward_pass(x):
    return x*w

def loss_func(y,y_pred):
    return ((y_pred-y)**2).mean()


def gradient_func(x,y,y_predicted):
    return np.dot(2*x, y_predicted-y).mean()

print(f'Pred before training f(5)= {forward_pass(5)}.')

learning_rate = 0.01
iterations = 10
for epoch in range(iterations):
    y_pred = forward_pass(X)
    loss = loss_func(Y, y_pred)
    gradient = gradient_func(X, Y, y_pred)
    w -= learning_rate*gradient
    if epoch%1==0:
        print(f'epoch: {epoch+1}: w= {w}, loss= {loss}.')

print(f'Pred after training f(5)= {forward_pass(5)}.')
print('------------------------- \n ----------Linear regression pyTorch----------\n------------------------- ')
#Linear regression pytorch
X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forward_pass(x):
    return x*w

def loss_func(y,y_pred):
    return ((y_pred-y)**2).mean()


def gradient_func(x,y,y_predicted):
    return np.dot(2*x, y_predicted-y).mean()

print(f'Pred before training f(5)= {forward_pass(5)}.')

learning_rate = 0.01
iterations = 100
for epoch in range(iterations):
    y_pred = forward_pass(X)
    loss = loss_func(Y, y_pred)
    loss.backward()
    with torch.no_grad():
        w -= learning_rate*w.grad
    w.grad.zero_()
    if epoch%10==0:
        print(f'epoch: {epoch+1}: w= {w}, loss= {loss}.')

print(f'Pred after training f(5)= {forward_pass(5)}.')