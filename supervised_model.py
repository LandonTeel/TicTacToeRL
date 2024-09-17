import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset import return_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr = 1e-3
batch_size = 32
epochs = 10

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

def train() -> None:
    model = SupervisedModel().to(device)

    dataset = return_data()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_function = nn.MSELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

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


def main():
    train()

if __name__ == "__main__":
    main()

