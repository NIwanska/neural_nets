import torch
from torch import nn, optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import csv


device = torch.device("cuda")

data = pd.read_csv('train_data.csv')
X = data.drop(columns=['SalePrice']).values
Y = data['SalePrice'].values

test_data = pd.read_csv('test_data.csv')
test_data = test_data.astype('float32')
test_data = torch.from_numpy(test_data.values[:,:])


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)