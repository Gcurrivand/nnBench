import pygame
# Colors
BLACK = (0, 0, 0)

def square(surface, x, y, width, height, color=BLACK, fill=False):
    line_width = 0 if fill else 2
    pygame.draw.rect(surface, color, (x, y, width, height), line_width)

def line(surface, start, end, width):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = max(abs(dx), abs(dy))
    for i in range(distance):
        x = int(start[0] + float(i) / distance * dx)
        y = int(start[1] + float(i) / distance * dy)
        pygame.draw.circle(surface, BLACK, (x, y), width // 2)