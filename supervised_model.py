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
        self.layers = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x) -> torch.Tensor:
        out = self.layers(x)
        return out

def train(model: SupervisedModel, dataloader: DataLoader) -> None:
    model.train()
    loss_function = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), lr=lr)
    i = 0
    for epoch in range(epochs):
        print(f"\nepoch: {epoch}\n")
        for batch_inputs, batch_labels in dataloader:
            for idx in range(len(batch_inputs)):
                inp, label = batch_inputs[idx].to(device).float(), batch_labels[idx].to(device).float()
                optim.zero_grad()
                prediction = model(inp)
                loss = loss_function(prediction, label)
                loss.backward()
                optim.step()

                if i % 100 == 0:
                    print(f"loss: {loss.item()}")

                i += 1

def test(model: SupervisedModel, dataloader: DataLoader) -> None:
    model.eval()
    loss_function = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    count = 0
    with torch.no_grad():
        for batch_inputs, batch_labels in dataloader:
            for idx in range(len(batch_inputs)):
                inp, label = batch_inputs[idx].to(device).float(), batch_labels[idx].to(device).float()
                prediction = model(inp)
                total_loss += loss_function(prediction, label).item()
                row, col = prediction
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
                if row == label[0] and col == label[1]:
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

