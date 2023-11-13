import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


#Data prep
bc = datasets.load_breast_cancer()
X,y = bc.data,bc.target
n_samples,n_features = X.shape
print(n_samples,n_features)
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)
#Scaling
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test=ss.transform(X_test)
X_train = torch.from_numpy(X_train.astype(np.float32)) #to tensors
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32)) 
y_test = torch.from_numpy(y_test.astype(np.float32))
y_train = y_train.view(y_train.shape[0],1) #transpose
y_test = y_test.view(y_test.shape[0],1)
#Model
class LogReg(nn.Module):

    def __init__(self, n_input_features):
        super(LogReg, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
    def forward(self,x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
model = LogReg(n_features)
#Loss and optimizer
criterion=nn.BCELoss()
optimizer=torch.optim.SGD(model.parameters(), lr=0.01)
#Training loop
num_iter = 100
for iter in range(num_iter):
    y_predicted = model(X_train)
    loss = criterion(y_predicted,y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if (iter+1) %10 == 0:
        print(f'Epoch: {iter+1}, loss = {loss.item():.4f}')
#Evaluation
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_class = y_pred.round()
    acc = y_pred_class.eq(y_test).sum()/y_test.shape[0]
    print(f'Model accuracy: {acc:.4f}.')