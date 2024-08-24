import pygame
import numpy as np

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

clock = pygame.time.Clock()
pygame.init()
font = pygame.font.Font('Cascadia.ttf', 12)

pygame.display.set_caption("TicTacToeRL")
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

scale = 67
scale2 = 140

points = []
points.append(np.array([-1, 3]))
points.append(np.array([1, 3]))
points.append(np.array([3, 1]))
points.append(np.array([3, -1]))
points.append(np.array([1, -3]))
points.append(np.array([-1, -3]))
points.append(np.array([-3, -1]))
points.append(np.array([-3, 1]))

boxes = []
boxes.append(np.array([-3,3]))
boxes.append(np.array([0,3]))
boxes.append(np.array([3,3]))
boxes.append(np.array([-3,0]))
boxes.append(np.array([0,0]))
boxes.append(np.array([3,0]))
boxes.append(np.array([-3,-3]))
boxes.append(np.array([0,-3]))
boxes.append(np.array([3,-3]))

circles_to_draw = []
xs_to_draw = []

cleartext = font.render('Clear', True, BLACK, WHITE)
cleartext_rect = cleartext.get_rect()
cleartext_rect.bottomleft = (20, 580)
clear_rect = pygame.Rect(cleartext_rect.left - 5, cleartext_rect.top - 5, cleartext_rect.width + 10, cleartext_rect.height + 10)

#0 = O, 1 = X, 2 = Clear, 3 = FinishedGame
square_array = [
    [2, 2, 2],
    [2, 2, 2],
    [2, 2, 2],
]


def to_screen_coords(point):
    return(int(SCREEN_WIDTH/2 + scale*point[0]), int(SCREEN_HEIGHT/2 + scale*point[1]))

def boxes_to_screen_coords(point):
    return(int(SCREEN_WIDTH/2 + scale2*point[0]), int(SCREEN_HEIGHT/2 + scale2*point[1]))

def draw_circle(box):
    screen_coords = boxes_to_screen_coords(box)
    circles_to_draw.append(screen_coords)

def draw_x(box):
    screen_coords = boxes_to_screen_coords(box)
    xs_to_draw.append(screen_coords)

def clear():
    global square_array
    global box0_rect, box1_rect, box2_rect, box3_rect, box4_rect, box5_rect, box6_rect, box7_rect, box8_rect
    square_array = [
    [2, 2, 2],
    [2, 2, 2],
    [2, 2, 2],
    ]

    circles_to_draw.clear()
    xs_to_draw.clear()

    screen.fill(WHITE)

    pygame.draw.line(screen, BLACK, to_screen_coords(points[0]), to_screen_coords(points[5]), 5)
    pygame.draw.line(screen, BLACK, to_screen_coords(points[1]), to_screen_coords(points[4]), 5)   
    pygame.draw.line(screen, BLACK, to_screen_coords(points[2]), to_screen_coords(points[7]), 5)   
    pygame.draw.line(screen, BLACK, to_screen_coords(points[3]), to_screen_coords(points[6]), 5)

    pygame.draw.rect(screen, BLACK, clear_rect, 3)
    screen.blit(cleartext, cleartext_rect)

    box0_rect = pygame.draw.rect(screen, WHITE, pygame.Rect((SCREEN_WIDTH/2 - 175), (SCREEN_HEIGHT/2 - 175), 100, 100))
    box1_rect = pygame.draw.rect(screen, WHITE, pygame.Rect((SCREEN_WIDTH/2 - 50), (SCREEN_HEIGHT/2 - 175), 100, 100))
    box2_rect = pygame.draw.rect(screen, WHITE, pygame.Rect((SCREEN_WIDTH/2 + 75), (SCREEN_HEIGHT/2 - 175), 100, 100))
    box3_rect = pygame.draw.rect(screen, WHITE, pygame.Rect((SCREEN_WIDTH/2 - 175), (SCREEN_HEIGHT/2 - 50), 100, 100))
    box4_rect = pygame.draw.rect(screen, WHITE, pygame.Rect((SCREEN_WIDTH/2 - 50), (SCREEN_HEIGHT/2 - 50), 100, 100))
    box5_rect = pygame.draw.rect(screen, WHITE, pygame.Rect((SCREEN_WIDTH/2 + 75), (SCREEN_HEIGHT/2 - 50), 100, 100))
    box6_rect = pygame.draw.rect(screen, WHITE, pygame.Rect((SCREEN_WIDTH/2 - 175), (SCREEN_HEIGHT/2 + 75), 100, 100))
    box7_rect = pygame.draw.rect(screen, WHITE, pygame.Rect((SCREEN_WIDTH/2 - 50), (SCREEN_HEIGHT/2 + 75), 100, 100))
    box8_rect = pygame.draw.rect(screen, WHITE, pygame.Rect((SCREEN_WIDTH/2 + 75), (SCREEN_HEIGHT/2 + 75), 100, 100))

    
    pygame.display.update()

def win():
    global box0_rect, box1_rect, box2_rect, box3_rect, box4_rect, box5_rect, box6_rect, box7_rect, box8_rect
    global square_array
    if square_array[0] == [1, 1, 1]:
        pygame.draw.line(screen, BLACK, ((SCREEN_WIDTH/2 - 200), (SCREEN_HEIGHT/2 - 140)), ((SCREEN_WIDTH/2 + 200), (SCREEN_HEIGHT/2 - 140)) ,5)
    elif square_array[0] == [0, 0, 0]:
        pygame.draw.line(screen, BLACK, ((SCREEN_WIDTH/2 - 200), (SCREEN_HEIGHT/2 - 140)), ((SCREEN_WIDTH/2 + 200), (SCREEN_HEIGHT/2 - 140)) ,5)
    elif square_array[1] == [1, 1, 1]:
        pygame.draw.line(screen, BLACK, ((SCREEN_WIDTH/2 - 200), (SCREEN_HEIGHT/2 - 0)), ((SCREEN_WIDTH/2 + 200), (SCREEN_HEIGHT/2 - 0)) ,5)
    elif square_array[1] == [0, 0, 0]:
        pygame.draw.line(screen, BLACK, ((SCREEN_WIDTH/2 - 200), (SCREEN_HEIGHT/2 - 0)), ((SCREEN_WIDTH/2 + 200), (SCREEN_HEIGHT/2 - 0)) ,5)
    elif square_array[2] == [1, 1, 1]:
        pygame.draw.line(screen, BLACK, ((SCREEN_WIDTH/2 - 200), (SCREEN_HEIGHT/2 + 140)), ((SCREEN_WIDTH/2 + 200), (SCREEN_HEIGHT/2 + 140)) ,5)
    elif square_array[2] == [0, 0, 0]:
        pygame.draw.line(screen, BLACK, ((SCREEN_WIDTH/2 - 200), (SCREEN_HEIGHT/2 + 140)), ((SCREEN_WIDTH/2 + 200), (SCREEN_HEIGHT/2 + 140)) ,5)
    elif [square_array[i][0] for i in range(3)] == [1, 1, 1]:
        pygame.draw.line(screen, BLACK, ((SCREEN_WIDTH/2 - 140), (SCREEN_HEIGHT/2 - 200)), ((SCREEN_WIDTH/2 - 140), (SCREEN_HEIGHT/2 + 200)) ,5)
    elif [square_array[i][0] for i in range(3)] == [0, 0, 0]:
        pygame.draw.line(screen, BLACK, ((SCREEN_WIDTH/2 - 140), (SCREEN_HEIGHT/2 - 200)), ((SCREEN_WIDTH/2 - 140), (SCREEN_HEIGHT/2 + 200)) ,5)
    elif [square_array[i][1] for i in range(3)] == [1, 1, 1]:
        pygame.draw.line(screen, BLACK, ((SCREEN_WIDTH/2 - 0), (SCREEN_HEIGHT/2 - 200)), ((SCREEN_WIDTH/2 - 0), (SCREEN_HEIGHT/2 + 200)) ,5)
    elif [square_array[i][1] for i in range(3)] == [0, 0, 0]:
        pygame.draw.line(screen, BLACK, ((SCREEN_WIDTH/2 - 0), (SCREEN_HEIGHT/2 - 200)), ((SCREEN_WIDTH/2 - 0), (SCREEN_HEIGHT/2 + 200)) ,5)
    elif [square_array[i][2] for i in range(3)] == [1, 1, 1]:
        pygame.draw.line(screen, BLACK, ((SCREEN_WIDTH/2 + 140), (SCREEN_HEIGHT/2 + 200)), ((SCREEN_WIDTH/2 + 140), (SCREEN_HEIGHT/2 - 200)) ,5)
    elif [square_array[i][2] for i in range(3)] == [0, 0, 0]:
        pygame.draw.line(screen, BLACK, ((SCREEN_WIDTH/2 + 140), (SCREEN_HEIGHT/2 + 200)), ((SCREEN_WIDTH/2 + 140), (SCREEN_HEIGHT/2 - 200)) ,5)
    elif square_array[0][0] == 1 and square_array[1][1] == 1 and square_array[2][2] == 1:
        pygame.draw.line(screen, BLACK, ((SCREEN_WIDTH/2 -200), (SCREEN_HEIGHT/2 - 200)), ((SCREEN_WIDTH/2 + 200), (SCREEN_HEIGHT/2 + 200)) ,5)
    elif square_array[0][0] == 0 and square_array[1][1] == 0 and square_array[2][2] == 0:
        pygame.draw.line(screen, BLACK, ((SCREEN_WIDTH/2 -200), (SCREEN_HEIGHT/2 - 200)), ((SCREEN_WIDTH/2 + 200), (SCREEN_HEIGHT/2 + 200)) ,5)
    elif square_array[0][2] == 1 and square_array[1][1] == 1 and square_array[2][0] == 1:
        pygame.draw.line(screen, BLACK, ((SCREEN_WIDTH/2 +200), (SCREEN_HEIGHT/2 - 200)), ((SCREEN_WIDTH/2 - 200), (SCREEN_HEIGHT/2 + 200)) ,5)
    elif square_array[0][2] == 0 and square_array[1][1] == 0 and square_array[2][0] == 0:
        pygame.draw.line(screen, BLACK, ((SCREEN_WIDTH/2 +200), (SCREEN_HEIGHT/2 - 200)), ((SCREEN_WIDTH/2 - 200), (SCREEN_HEIGHT/2 + 200)) ,5)
    pygame.display.update()

def main():
    state = 1
    global square_array

    while True:

        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if state == 0 and square_array[0][0] == 2 and box0_rect.collidepoint(mouse_pos):
                    draw_circle(np.array([-1,-1]))
                    state = 1
                    square_array[0][0] = 0
                elif state == 0 and square_array[0][1] == 2 and box1_rect.collidepoint(mouse_pos):
                    draw_circle(np.array([0,-1]))
                    state = 1
                    square_array[0][1] = 0
                elif state == 0 and square_array[0][2] == 2 and box2_rect.collidepoint(mouse_pos):
                    draw_circle(np.array([1,-1]))
                    state = 1
                    square_array[0][2] = 0
                elif state == 0 and  square_array[1][0] == 2 and box3_rect.collidepoint(mouse_pos):
                    draw_circle(np.array([-1,0]))
                    state = 1
                    square_array[1][0] = 0
                elif state == 0 and  square_array[1][1] == 2 and box4_rect.collidepoint(mouse_pos):
                    draw_circle(np.array([0,0]))
                    state = 1
                    square_array[1][1] = 0
                elif state == 0 and  square_array[1][2] == 2 and box5_rect.collidepoint(mouse_pos):
                    draw_circle(np.array([1,0]))
                    state = 1
                    square_array[1][2] = 0
                elif state == 0 and  square_array[2][0] == 2 and box6_rect.collidepoint(mouse_pos):
                    draw_circle(np.array([-1,1]))
                    state = 1
                    square_array[2][0] = 0
                elif state == 0 and  square_array[2][1] == 2 and box7_rect.collidepoint(mouse_pos):
                    draw_circle(np.array([0,1]))
                    state = 1
                    square_array[2][1] = 0
                elif state == 0 and  square_array[2][2] == 2 and box8_rect.collidepoint(mouse_pos):
                    draw_circle(np.array([1,1]))
                    state = 1
                    square_array[2][2] = 0
                elif state == 1 and  square_array[0][0] == 2 and box0_rect.collidepoint(mouse_pos):
                    draw_x(np.array([-1,-1]))
                    state = 0
                    square_array[0][0] = 1
                elif state == 1 and  square_array[0][1] == 2 and box1_rect.collidepoint(mouse_pos):
                    draw_x(np.array([0,-1]))
                    state = 0
                    square_array[0][1] = 1
                elif state == 1 and  square_array[0][2] == 2 and box2_rect.collidepoint(mouse_pos):
                    draw_x(np.array([1,-1]))
                    state = 0
                    square_array[0][2] = 1
                elif state == 1 and  square_array[1][0] == 2 and box3_rect.collidepoint(mouse_pos):
                    draw_x(np.array([-1,0]))
                    state = 0
                    square_array[1][0] = 1
                elif state == 1 and  square_array[1][1] == 2 and box4_rect.collidepoint(mouse_pos):
                    draw_x(np.array([0,0]))
                    state = 0
                    square_array[1][1] = 1
                elif state == 1 and  square_array[1][2] == 2 and box5_rect.collidepoint(mouse_pos):
                    draw_x(np.array([1,0]))
                    state = 0
                    square_array[1][2] = 1
                elif state == 1 and  square_array[2][0] == 2 and box6_rect.collidepoint(mouse_pos):
                    draw_x(np.array([-1,1]))
                    state = 0
                    square_array[2][0] = 1
                elif state == 1 and  square_array[2][1] == 2 and box7_rect.collidepoint(mouse_pos):
                    draw_x(np.array([0,1]))
                    state = 0
                    square_array[2][1] = 1
                elif state == 1 and  square_array[2][2] == 2 and box8_rect.collidepoint(mouse_pos):
                    draw_x(np.array([1,1]))
                    state = 0
                    square_array[2][2] = 1

                elif clear_rect.collidepoint(mouse_pos):
                    clear()
                

            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    exit()       
        
        screen.fill(WHITE)

        pygame.draw.line(screen, BLACK, to_screen_coords(points[0]), to_screen_coords(points[5]), 5)
        pygame.draw.line(screen, BLACK, to_screen_coords(points[1]), to_screen_coords(points[4]), 5)   
        pygame.draw.line(screen, BLACK, to_screen_coords(points[2]), to_screen_coords(points[7]), 5)   
        pygame.draw.line(screen, BLACK, to_screen_coords(points[3]), to_screen_coords(points[6]), 5)

        pygame.draw.rect(screen, BLACK, clear_rect, 3)
        screen.blit(cleartext, cleartext_rect)
        
        box0_rect = pygame.draw.rect(screen, WHITE, pygame.Rect((SCREEN_WIDTH/2 - 175), (SCREEN_HEIGHT/2 - 175), 100, 100))
        box1_rect = pygame.draw.rect(screen, WHITE, pygame.Rect((SCREEN_WIDTH/2 - 50), (SCREEN_HEIGHT/2 - 175), 100, 100))
        box2_rect = pygame.draw.rect(screen, WHITE, pygame.Rect((SCREEN_WIDTH/2 + 75), (SCREEN_HEIGHT/2 - 175), 100, 100))
        box3_rect = pygame.draw.rect(screen, WHITE, pygame.Rect((SCREEN_WIDTH/2 - 175), (SCREEN_HEIGHT/2 - 50), 100, 100))
        box4_rect = pygame.draw.rect(screen, WHITE, pygame.Rect((SCREEN_WIDTH/2 - 50), (SCREEN_HEIGHT/2 - 50), 100, 100))
        box5_rect = pygame.draw.rect(screen, WHITE, pygame.Rect((SCREEN_WIDTH/2 + 75), (SCREEN_HEIGHT/2 - 50), 100, 100))
        box6_rect = pygame.draw.rect(screen, WHITE, pygame.Rect((SCREEN_WIDTH/2 - 175), (SCREEN_HEIGHT/2 + 75), 100, 100))
        box7_rect = pygame.draw.rect(screen, WHITE, pygame.Rect((SCREEN_WIDTH/2 - 50), (SCREEN_HEIGHT/2 + 75), 100, 100))
        box8_rect = pygame.draw.rect(screen, WHITE, pygame.Rect((SCREEN_WIDTH/2 + 75), (SCREEN_HEIGHT/2 + 75), 100, 100))

        for circle_coords in circles_to_draw:
            pygame.draw.circle(screen, BLACK, circle_coords, 50, 5)
        
        for x_coords in xs_to_draw:
            pygame.draw.line(screen, BLACK, ((x_coords[0] - 50, x_coords[1] - 50)), ((x_coords[0] + 50, x_coords[1] + 50)), 5)
            pygame.draw.line(screen, BLACK, ((x_coords[0] + 50, x_coords[1] - 50)), ((x_coords[0] - 50, x_coords[1] + 50)), 5)
        win()
        pygame.display.flip()


if __name__ == "__main__":
    main()