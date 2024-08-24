import torch  
import pygame
import TicTacToe


torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("Hello")
    board = TicTacToe.Board()

if __name__ == "__main__":
    main()
