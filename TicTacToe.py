class Board:
    def __init__(self) -> None:
        self.board = [[-1 for i in range(3)] for j in range(3)]
        print(self.board)