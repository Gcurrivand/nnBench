import pygame
import sys
import os
from PIL import Image
import re

# Initialize Pygame
pygame.init()

# Set up the drawing window
size = 600
screen = pygame.display.set_mode((size, size))
pygame.display.set_caption("Square Drawing with 30x30 Save")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Set up the drawing surface
drawing_surface = pygame.Surface((size, size))
drawing_surface.fill(WHITE)

# Pen size
pen_size = 2
max_pen_size = 20

# Buttons
button_width = 90
button_height = 30
save_button_rect = pygame.Rect(size - 100, 10, button_width, button_height)
clear_button_rect = pygame.Rect(size - 100, 50, button_width, button_height)
button_color = (100, 100, 100)
font = pygame.font.Font(None, 24)

# Custom event for stopping drawing
STOP_DRAWING_EVENT = pygame.USEREVENT + 1
# Timer variables
drawing_timer = None
DRAWING_TIMEOUT = 500
# Flag to track if drawing has stopped
drawing_stopped = False

# Function to start the drawing timer
def start_drawing_timer():
    global drawing_timer, drawing_stopped
    if drawing_timer:
        pygame.time.set_timer(drawing_timer, 0)  # Cancel existing timer
    drawing_timer = pygame.time.set_timer(STOP_DRAWING_EVENT, DRAWING_TIMEOUT)
    drawing_stopped = False

# Function to stop the drawing timer
def stop_drawing_timer():
    global drawing_timer
    if drawing_timer:
        pygame.time.set_timer(drawing_timer, 0)
        drawing_timer = None

# Ensure the folders exist
folders = ["Dataset", "Dataset/Meaning", "Dataset/Numbers"]
for folder in folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Main game loop
drawing = False
last_pos = None
# Function to get the latest save count
def get_latest_save_count():
    numbers = []
    for filename in os.listdir("Dataset/Numbers"):
        if filename.startswith("drawing_") and filename.endswith(".png"):
            match = re.search(r'drawing_(\d+)\.png', filename)
            if match:
                numbers.append(int(match.group(1)))
    return max(numbers) if numbers else 0

# Initialize save_count with the latest number
save_count = get_latest_save_count()

def clear_board():
    drawing_surface.fill(WHITE)

def save_image_and_meaning():
    global save_count
    save_count += 1
    filename = f"Dataset_DRAW/Numbers/drawing_{save_count}.png"
    pygame.image.save(drawing_surface, filename)
    image = Image.open(filename)
    resized_image = image.resize((28, 28), Image.LANCZOS)
    resized_image.save(filename)

    # Ask for keyboard input
    pygame.display.set_caption("Enter the meaning of the drawing")
    meaning = ""
    input_active = True
    while input_active:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    input_active = False
                elif event.key == pygame.K_BACKSPACE:
                    meaning = meaning[:-1]
                else:
                    meaning += event.unicode
        
        screen.fill(WHITE)
        text_surface = font.render(meaning, True, BLACK)
        screen.blit(text_surface, (10, 10))
        pygame.display.flip()
    # Save meaning to text file
    with open(f"Dataset_DRAW/Meaning/drawing_{save_count}.txt", "w") as f:
        f.write(meaning)
    pygame.display.set_caption("Square Drawing with 30x30 Save")
    clear_board()

def is_clicking_button(pos):
    return save_button_rect.collidepoint(pos) or clear_button_rect.collidepoint(pos)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                if save_button_rect.collidepoint(event.pos):
                    save_image_and_meaning()
                elif clear_button_rect.collidepoint(event.pos):
                    clear_board()
                elif not is_clicking_button(event.pos):
                    drawing = True
                    last_pos = event.pos
                    stop_drawing_timer()
                    drawing_stopped = False
            elif event.button == 4:  # Mouse wheel up
                pen_size = min(pen_size + 1, max_pen_size)
            elif event.button == 5:  # Mouse wheel down
                pen_size = max(pen_size - 1, 1)
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left mouse button
                drawing = False
                last_pos = None
                if not is_clicking_button(event.pos):
                    start_drawing_timer()
        elif event.type == pygame.MOUSEMOTION:
            if drawing and last_pos and not is_clicking_button(event.pos):
                current_pos = event.pos
                distance = max(abs(current_pos[0] - last_pos[0]), abs(current_pos[1] - last_pos[1]))
                for i in range(distance):
                    x = int(last_pos[0] + (current_pos[0] - last_pos[0]) * i / distance)
                    y = int(last_pos[1] + (current_pos[1] - last_pos[1]) * i / distance)
                    pygame.draw.circle(drawing_surface, BLACK, (x, y), pen_size)
                last_pos = event.pos
                start_drawing_timer()
            elif drawing and is_clicking_button(event.pos):
                drawing = False
                last_pos = None
        elif event.type == STOP_DRAWING_EVENT and not drawing_stopped:
            print("Stopped drawing for 1/2 seconds")
            save_image_and_meaning()
            drawing_stopped = True
            stop_drawing_timer()

    # Clear the screen
    screen.fill(WHITE)

    # Copy the drawing surface to the main screen
    screen.blit(drawing_surface, (0, 0))

    # Draw the save button
    pygame.draw.rect(screen, button_color, save_button_rect)
    save_text = font.render("Save", True, WHITE)
    screen.blit(save_text, (save_button_rect.x + 10, save_button_rect.y + 5))

    # Draw the clear button
    pygame.draw.rect(screen, button_color, clear_button_rect)
    clear_text = font.render("Clear", True, WHITE)
    screen.blit(clear_text, (clear_button_rect.x + 10, clear_button_rect.y + 5))

    # Draw the cursor
    mouse_pos = pygame.mouse.get_pos()
    pygame.draw.circle(screen, RED, mouse_pos, pen_size, 1)

    pygame.display.flip()