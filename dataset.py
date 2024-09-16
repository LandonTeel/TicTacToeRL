from minimax import minimax_algorithm, get_empty_positions
from pathlib import Path
import random

# TODO: negatives will mess up parsing of data

PATH = Path("./data/dataset.txt")

P1, P2 = 0, 1

def generate_dataset(sample_size=100) -> list[tuple[list[list[int]], tuple[int, int]]]:
    # some duplicate data is fine
    data = []
    for i in range(sample_size):
        print(i)
        state = [[-1 for _ in range(3)] for _ in range(3)]
        depth = random.randint(0, 8)
        for j in range(depth):
            if j % 2 == 0:
                _, move = minimax_algorithm(state)
                if move is not None:
                    row, col = move
                    state[row][col] = P2
            else:
                options = get_empty_positions(state)
                move = random.choice(options)
                row, col = move
                state[row][col] = P1
        
        _, move = minimax_algorithm(state)
        if move is not None:
            data.append(tuple([state, move]))
    
    print(data)

    return data
    
def write_dataset(data: list[tuple[list[list[int]], tuple[int, int]]]) -> None:
    with open(PATH, "w") as f:
        for p in data:
            state, move = p
            r, c = move
            str_state = "".join([str(v) for row in state for v in row])
            f.write(f"{str_state}:{r}{c}\n")


def read_dataset() -> list[tuple[list[list[int]], tuple[int, int]]]:
    data = []
    with open(PATH, "r") as f:
        for line in f:
            line = line.rstrip()
            split_line = line.split(":")
            state, move = split_line[0], split_line[1]
            state = [int(i) for i in state]
            move = tuple([int(i) for i in move])
            data.append(tuple([state, move]))

    return data


if __name__ == "__main__":
    try:
        data = read_dataset()
    except Exception:
        data = generate_dataset(sample_size=5)
        write_dataset(data)
    print(len(data))


