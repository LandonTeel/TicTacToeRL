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

def minimax_algorithm(state: list[list[int]]) -> tuple[int, tuple[int, int]]:
    score, best_move = minimax_helper(state, P2)
    return score, best_move

def test():
    _, move = minimax_algorithm(state)
    if move is not None:
        row, col = move
        state[row][col] = P2

