from minimax import minimax_algorithm, get_empty_positions, P1, P2
from torch.utils.data import Dataset
from pathlib import Path
import random
import torch

# TODO: current dataset has too many duplicates

PATH = Path("./data/dataset.txt")

class TicTacToeDataset(Dataset):
    def __init__(self, data) -> None:
        self.inputs = [torch.tensor(d[0], dtype=torch.float32).flatten() for d in data]
        self.labels = [torch.tensor(d[1], dtype=torch.long) for d in data]

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.labels[idx]


def generate_dataset(sample_size=100, early_game_bias=False) -> list[tuple[list[list[int]], tuple[int, int]]]:
    data = []
    i = 0
    while i < sample_size:
        print(i)
        state = [[-1 for _ in range(3)] for _ in range(3)]
        depth = random.randint(0, 4) if early_game_bias else random.randint(0, 8)
        for j in range(depth):
            if j % 2 == 0:
                _, move = minimax_algorithm(state)
                if move is not None:
                    row, col = move
                    data.append(tuple([state, move]))
                    state[row][col] = P2
                    i += 1
            else:
                options = get_empty_positions(state)
                move = random.choice(options)
                row, col = move
                state[row][col] = P1
        
        _, move = minimax_algorithm(state)
        if move is not None:
            data.append(tuple([state, move]))
            i += 1

    return data
    
def write_dataset(data: list[tuple[list[list[int]], tuple[int, int]]]) -> None:
    with open(PATH, "w") as f:
        for p in data:
            state, move = p
            r, c = move
            str_state = " ".join([str(v) for row in state for v in row])
            f.write(f"{str_state}:{r}{c}\n")


def read_dataset() -> list[tuple[list[int], tuple[int, int]]]:
    data = []
    with open(PATH, "r") as f:
        for line in f:
            line = line.rstrip()
            split_line = line.split(":")
            state, move = [int(i) for i in split_line[0].split(" ")], split_line[1]
            # state = [state[i:i + 3] for i in range(0, len(state), 3)]
            move = tuple([int(i) for i in move])
            data.append(tuple([state, move]))

    return data


def return_data() -> TicTacToeDataset:
    try:
        data = read_dataset()
    except Exception:
        inp = input("WARNING: Generating a dataset is slow. Do you really want to (y/N)? ")
        if "y" in inp.lower():
            data = generate_dataset(sample_size=10000)
            write_dataset(data)
        else:
            raise

    dataset = TicTacToeDataset(data)
    return dataset

