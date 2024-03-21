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
X = data_encoded.drop(columns=['SalePrice']).values
Y = data_encoded['SalePrice'].values


test_data = pd.read_csv('test_data.csv')
test_data_dropped = test_data.drop(columns=['HallwayType','HeatingType','AptManageType','SubwayStation', 'TimeToBusStop', 'TimeToSubway'])
pd_HallwayTypepd = pd.get_dummies(test_data['HallwayType'],  dtype=float)
pd_HeatingType = pd.get_dummies(test_data['HeatingType'],  dtype=float)
pd_AptManageType = pd.get_dummies(test_data['AptManageType'],  dtype=float)
pd_SubwayStation = pd.get_dummies(test_data['SubwayStation'],  dtype=float)
pd_TimeToSubway = pd.get_dummies(test_data['TimeToSubway'],  dtype=float)
pd_TimeToBusStop = pd.get_dummies(test_data['TimeToBusStop'],  dtype=float)
test_data_encoded = pd.concat([pd_HallwayTypepd,pd_HeatingType ,pd_AptManageType,pd_SubwayStation,pd_TimeToSubway,pd_TimeToBusStop , test_data_dropped], axis=1, join='outer')

test_data = test_data_encoded.astype('float32')
test_data = torch.from_numpy(test_data.values[:,:])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

class LinearRegressionModel(nn.Module):

    def __init__(self, num_inputs):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, 64)
        self.act_fn = nn.ReLU()
        self.linear2 = nn.Linear(64, 64)
        self.linear3 = nn.Linear(64, 128)
        self.linear4 = nn.Linear(128, 256)
        self.linear5 = nn.Linear(256, 128)
        self.linear6 = nn.Linear(128, 64)
        self.linear7 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act_fn(x)
        x = self.linear2(x)
        x = self.act_fn(x)
        x = self.linear3(x)
        x = self.act_fn(x)
        x = self.linear4(x)
        x = self.act_fn(x)
        x = self.linear5(x)
        x = self.act_fn(x)
        x = self.linear6(x)
        x = self.act_fn(x)
        x = self.linear7(x)
        return x

input_size = 33

model = LinearRegressionModel(input_size)
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)



# Trenowanie modelu
num_epochs = 100
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels.view(-1, 1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')



with torch.no_grad():
    model.eval()
    X_test, y_test = X_test.to(device), y_test.to(device)
    y_pred = model(X_test)
    test_loss = criterion(y_pred, y_test.view(-1, 1))
    print(f'Final Test Loss: {test_loss.item():.4f}')

model.eval()
true_preds, num_preds = 0., 0.
total_loss = 0
with open('wyniki.csv', 'w') as f:
  write = csv.writer(f)
  with torch.no_grad():
      for data_inputs in test_data:
          data_inputs = data_inputs.to(device)
          preds = model(data_inputs)
          write.writerow([preds.item()])