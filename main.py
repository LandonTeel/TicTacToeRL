# import torch  

# torch.device("cuda" if torch.cuda.is_available() else "cpu")


import math

P1, P2 = 0, 1

def is_winner(board: list[list[int]], player: int) -> bool:
    for row in board:
        if all(cell == player for cell in row):
            return True
    for col in range(3):
        if all(board[row][col] == player for row in range(3)):
            return True
    if all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3)):
        return True
    return False

def get_empty_positions(board: list[list[int]]) -> list[tuple[int, int]]:
    return [(i, j) for i in range(3) for j in range(3) if board[i][j] == -1]

def minimax_helper(current_state: list[list[int]], turn: int) -> tuple[int, tuple[int, int]]:
    if is_winner(current_state, P1):
        return 1, None
    if is_winner(current_state, P2):
        return -1, None
    if not get_empty_positions(current_state):
        return 0, None

    best_score = -math.inf if turn == P1 else math.inf
    best_move = None

    for i, j in get_empty_positions(current_state):
        new_state = [row[:] for row in current_state]
        new_state[i][j] = turn
        score, _ = minimax_helper(new_state, 1 - turn)

        if turn == P1:
            if score > best_score:
                best_score = score
                best_move = (i, j)
        else:
            if score < best_score:
                best_score = score
                best_move = (i, j)

    return best_score, best_move

def minimax(state: list[list[int]]) -> tuple[int, tuple[int, int]]:
    score, best_move = minimax_helper(state, P2) 
    return score, best_move

def print_board(board: list[list[int]]):
    players = ["O", "X", " "]
    for row in board:
        print(" | ".join(players[cell] for cell in row))
        print("---------")

def main():
    state = [[-1 for _ in range(3)] for _ in range(3)]
    current_player = P1 

    while True:
        print("current state:")
        print_board(state)
        
        if current_player == P1:
            while True:
                try:
                    row = int(input("Enter row (0-2): "))
                    col = int(input("Enter column (0-2): "))
                    if state[row][col] == -1:
                        state[row][col] = P1
                        break
                    else:
                        print("cell taken")
                except (ValueError, IndexError):
                    print("invalid input")
        else:
            _, move = minimax(state)
            if move is not None:
                row, col = move
                state[row][col] = P2

        if is_winner(state, P1):
            print("\nO wins\n")
            break
        if is_winner(state, P2):
            print("\nX wins\n")
            break
        if not get_empty_positions(state):
            print("\ndraw\n")
            break
        
        current_player = 1 - current_player

    print_board(state)

if __name__ == "__main__":
    main()

