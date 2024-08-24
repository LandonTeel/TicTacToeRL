import pygame
import numpy as np

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

clock = pygame.time.Clock()

pygame.display.set_caption("TicTacToeERL")
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

scale = 67

points = []
points.append(np.array([-1, 3]))
points.append(np.array([1, 3]))
points.append(np.array([3, 1]))
points.append(np.array([3, -1]))
points.append(np.array([1, -3]))
points.append(np.array([-1, -3]))
points.append(np.array([-3, -1]))
points.append(np.array([-3, 1]))

def to_screen_coords(point):
    return(int(SCREEN_WIDTH/2 + scale*point[0]), int(SCREEN_HEIGHT/2 + scale*point[1]))

def main():
    while True:

        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    exit()
        
        screen.fill(WHITE)

        pygame.draw.line(screen, BLACK, to_screen_coords(points[0]), to_screen_coords(points[5]), 5)
        pygame.draw.line(screen, BLACK, to_screen_coords(points[1]), to_screen_coords(points[4]), 5)   
        pygame.draw.line(screen, BLACK, to_screen_coords(points[2]), to_screen_coords(points[7]), 5)   
        pygame.draw.line(screen, BLACK, to_screen_coords(points[3]), to_screen_coords(points[6]), 5)
        pygame.display.flip()

main()