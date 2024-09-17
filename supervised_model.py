import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from dataset import return_data

# TODO: fix the model and add torch load/save for weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 1e-3
batch_size = 32
epochs = 10
train_split_ratio = 0.9

class SupervisedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(9, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn1(x) 
        x = nn.functional.leaky_relu(x, negative_slope=0.01)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x) 
        x = nn.functional.leaky_relu(x, negative_slope=0.01)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

"""
class SupervisedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x) -> torch.Tensor:
        return self.layers(x)

"""

def train(model: SupervisedModel, dataloader: DataLoader) -> None:
    model.train()
    loss_function = nn.MSELoss()
    optim = torch.optim.SGD(model.parameters(), lr=lr)
    i = 0
    for epoch in range(epochs):
        print(f"\nepoch: {epoch}\n")
        for batch_inputs, batch_labels in dataloader:
            # for idx in range(len(batch_inputs)):
                # inp, label = batch_inputs[idx].to(device).float(), batch_labels[idx].to(device).float()
                batch_inputs, batch_labels = batch_inputs.to(device).float(), batch_labels.to(device).float()
                optim.zero_grad()
                # prediction = model(inp)
                prediction = model(batch_inputs)
                loss = loss_function(prediction, batch_labels)
                loss.backward()
                optim.step()

                if i % 100 == 0:
                    print(f"loss: {loss.item()}")

                i += 1

def test(model: SupervisedModel, dataloader: DataLoader) -> None:
    model.eval()
    loss_function = nn.MSELoss()
    count, correct, total_loss = 0, 0, 0
    with torch.no_grad():
        for batch_inputs, batch_labels in dataloader:
                batch_inputs, batch_labels = batch_inputs.to(device).float(), batch_labels.to(device).float()

            # for idx in range(len(batch_inputs)):
                # inp, label = batch_inputs[idx].to(device).float(), batch_labels[idx].to(device).float()
                prediction = model(batch_inputs)
                total_loss += loss_function(prediction, batch_labels).item()
                for i in range(len(prediction)):
                    row, col = prediction[i]
                    row, col = row.item(), col.item()
                    if row < 0:
                        row = 0
                    if row > 2:
                        row = 2
                    if col < 0:
                        col = 0
                    if col > 2:
                        col = 2
                    row, col = round(row), round(col)
                    if row == batch_labels[i][0] and col == batch_labels[i][1]:
                        correct += 1
                    count += 1

    average_loss = total_loss / len(dataloader.dataset)
    print(f"Average test loss: {average_loss}")
    print(f"Accuracy: {correct*100/count}%")

def main():
    dataset = return_data()
    total_size = len(dataset)
    train_size = int(total_size * train_split_ratio)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = SupervisedModel().to(device)
    train(model, train_dataloader)
    test(model, test_dataloader)

if __name__ == "__main__":
    main()

