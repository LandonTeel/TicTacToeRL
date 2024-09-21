import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from dataset import return_data
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 10
TRAIN_SPLIT_RATIO = 0.9

class SupervisedModel(nn.Module):
    def __init__(self) -> None:
        super(SupervisedModel, self).__init__()
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
        x = self.fc3(x)  # crossentropyloss applies softmax internally
        return x

def train(model: SupervisedModel, dataloader: DataLoader) -> None:
    model.train()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\nepoch {epoch}/{EPOCHS}")
        epoch_loss = 0.0
        for batch_inputs, batch_labels in dataloader:
            batch_inputs = batch_inputs.to(DEVICE).float()
            batch_labels = torch.argmax(batch_labels, dim=1).to(DEVICE).long()

            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = loss_function(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if global_step % 100 == 0:
                print(f"step {global_step} loss: {loss.item():.4f}")
            global_step += 1

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"avg loss for epoch {epoch}: {avg_epoch_loss:.4f}")

def test(model: SupervisedModel, dataloader: DataLoader) -> None:
    model.eval()
    loss_function = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_inputs, batch_labels in dataloader:
            batch_inputs = batch_inputs.to(DEVICE).float()
            batch_labels = torch.argmax(batch_labels, dim=1).to(DEVICE).long()

            outputs = model(batch_inputs)
            loss = loss_function(outputs, batch_labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)

    average_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    print(f"\naverage test loss: {average_loss:.4f}")
    print(f"accuracy: {accuracy:.2f}%")

def main():
    dataset = return_data()
    total_size = len(dataset)
    train_size = int(total_size * TRAIN_SPLIT_RATIO)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SupervisedModel().to(DEVICE)

    train(model, train_dataloader)
    test(model, test_dataloader)

if __name__ == "__main__":
    main()

