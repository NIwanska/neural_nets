import torch
from torch import nn, optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import csv


device = torch.device("cuda")
data = pd.read_csv('train_data.csv')
data_dropped = data.drop(columns=['HallwayType','HeatingType','AptManageType','SubwayStation', 'TimeToBusStop', 'TimeToSubway'])
pd_HallwayTypepd = pd.get_dummies(data['HallwayType'],  dtype=float)
pd_HeatingType = pd.get_dummies(data['HeatingType'],  dtype=float)
pd_AptManageType = pd.get_dummies(data['AptManageType'],  dtype=float)
pd_SubwayStation = pd.get_dummies(data['SubwayStation'],  dtype=float)
pd_TimeToSubway = pd.get_dummies(data['TimeToSubway'],  dtype=float)
pd_TimeToBusStop = pd.get_dummies(data['TimeToBusStop'],  dtype=float)
data_encoded = pd.concat([pd_HallwayTypepd,pd_HeatingType ,pd_AptManageType,pd_SubwayStation,pd_TimeToSubway,pd_TimeToBusStop , data_dropped], axis=1, join='outer')
print(pd.get_dummies(data['HeatingType']))
print(data_encoded)
X = data_encoded.drop(columns=['SalePrice']).values
Y = data_encoded['SalePrice'].values


test_data = pd.read_csv('test_data.csv')
test_data_dropped = test_data.drop(columns=['HallwayType','HeatingType','AptManageType','SubwayStation', 'TimeToBusStop', 'TimeToSubway'])
test_data_encoded = pd.concat([pd.get_dummies(test_data['HallwayType']),pd.get_dummies(test_data['HeatingType']),pd.get_dummies(test_data['AptManageType']), pd.get_dummies(test_data['SubwayStation']), pd.get_dummies(test_data['TimeToSubway']), pd.get_dummies(test_data['TimeToBusStop']), test_data_dropped], axis=1, join='outer', dtype=float)

test_data = test_data_encoded.astype('float32')
test_data = torch.from_numpy(test_data.values[:,:])

print(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)