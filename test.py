import pygame
import numpy as np

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
LINE_WIDTH = 5

pygame.init()
clock = pygame.time.Clock()
font = pygame.font.Font('Cascadia.ttf', 12)
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("TicTacToeRL")

def to_screen_coords(point, scale):
    return (int(SCREEN_WIDTH / 2 + scale * point[0]), int(SCREEN_HEIGHT / 2 + scale * point[1]))

def draw_grid(screen, scale):
    pygame.draw.line(screen, BLACK, to_screen_coords([-3, 1], scale), to_screen_coords([3, 1], scale), LINE_WIDTH)
    pygame.draw.line(screen, BLACK, to_screen_coords([-3, -1], scale), to_screen_coords([3, -1], scale), LINE_WIDTH)
    pygame.draw.line(screen, BLACK, to_screen_coords([-1, 3], scale), to_screen_coords([-1, -3], scale), LINE_WIDTH)
    pygame.draw.line(screen, BLACK, to_screen_coords([1, 3], scale), to_screen_coords([1, -3], scale), LINE_WIDTH)

def draw_shapes(screen, circles, xs):
    for circle_coords in circles:
        pygame.draw.circle(screen, BLACK, circle_coords, 50, LINE_WIDTH)
    for x_coords in xs:
        pygame.draw.line(screen, BLACK, (x_coords[0] - 50, x_coords[1] - 50), (x_coords[0] + 50, x_coords[1] + 50), LINE_WIDTH)
        pygame.draw.line(screen, BLACK, (x_coords[0] + 50, x_coords[1] - 50), (x_coords[0] - 50, x_coords[1] + 50), LINE_WIDTH)

def draw_winning_line(screen, start, end):
    pygame.draw.line(screen, BLACK, start, end, LINE_WIDTH)

class TicTacToeGame:
    def __init__(self):
        self.clear_button = self.setup_clear_button()
        self.boxes = self.setup_boxes()
        self.square_array = [[2 for _ in range(3)] for _ in range(3)]
        self.circles_to_draw = []
        self.xs_to_draw = []
        self.state = 1  # 0 = O turn, 1 = X turn
        self.winner = None
        self.winning_line = None

    def setup_clear_button(self):
        cleartext = font.render('Clear', True, BLACK, WHITE)
        cleartext_rect = cleartext.get_rect()
        cleartext_rect.bottomleft = (20, 580)
        return cleartext, cleartext_rect

    def setup_boxes(self):
        boxes = []
        offset = [-175, -50, 75]
        for y in offset:
            row = []
            for x in offset:
                rect = pygame.Rect(SCREEN_WIDTH / 2 + x, SCREEN_HEIGHT / 2 + y, 100, 100)
                row.append(rect)
            boxes.append(row)
        return boxes

    def clear_board(self):
        self.square_array = [[2 for _ in range(3)] for _ in range(3)]
        self.circles_to_draw.clear()
        self.xs_to_draw.clear()
        self.state = 1
        self.winner = None
        self.winning_line = None
        self.render()

    def check_win(self):
        lines = [
            [(0, 0), (0, 1), (0, 2)],  # top row
            [(1, 0), (1, 1), (1, 2)],  # middle row
            [(2, 0), (2, 1), (2, 2)],  # bottom row
            [(0, 0), (1, 0), (2, 0)],  # left col
            [(0, 1), (1, 1), (2, 1)],  # middle col
            [(0, 2), (1, 2), (2, 2)],  # right col
            [(0, 0), (1, 1), (2, 2)],  # diagonal tl to br
            [(0, 2), (1, 1), (2, 0)],  # diagonal
        ]

        for line in lines:
            p1, p2, p3 = line
            if self.square_array[p1[0]][p1[1]] == self.square_array[p2[0]][p2[1]] == self.square_array[p3[0]][p3[1]] != 2:
                self.winner = self.square_array[p1[0]][p1[1]]
                self.winning_line = (self.boxes[p1[0]][p1[1]].center, self.boxes[p3[0]][p3[1]].center)
                return True
        return False

    def render(self):
        screen.fill(WHITE)
        
        draw_grid(screen, 67)
        draw_shapes(screen, self.circles_to_draw, self.xs_to_draw)
        pygame.draw.rect(screen, BLACK, self.clear_button[1], 3)
        screen.blit(*self.clear_button)
        
        if self.winning_line:
            draw_winning_line(screen, *self.winning_line)
        
        pygame.display.flip()

    def handle_click(self, mouse_pos):
        if self.winner:
            return  # stop game if won

        if self.clear_button[1].collidepoint(mouse_pos):
            self.clear_board()
            return

        for i, row in enumerate(self.boxes):
            for j, box in enumerate(row):
                if box.collidepoint(mouse_pos) and self.square_array[i][j] == 2:
                    if self.state == 0:
                        self.circles_to_draw.append(box.center)
                        self.square_array[i][j] = 0
                    else:
                        self.xs_to_draw.append(box.center)
                        self.square_array[i][j] = 1
                    self.state = 1 - self.state
                    if self.check_win():
                        self.render()
                    return

def main():
    game = TicTacToeGame()
    game.clear_board()
    while True:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                game.handle_click(event.pos)
        game.render()

if __name__ == "__main__":
    main()

