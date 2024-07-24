import pygame
import sys
import os
import numpy as np
from PIL import Image
package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, package_path)
from NN import LeNet5, CNN_V1,cnn_run_single_inference
from draw_form import square, line
from Drawing.pygame_image import pygame_resize_image 
from OBJD import bruteforce_detection

# Initialize Pygame
pygame.init()

# Set up the drawing window
size = 560
screen = pygame.display.set_mode((size, size))
pygame.display.set_caption("Drawing App with Circular Pen")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
RED = (255, 0, 0)

# Set up the drawing surface
drawing_surface = pygame.Surface((size, size))
drawing_surface.fill(WHITE)

# Pen size
pen_size = 42
min_pen_size = 1
max_pen_size = 90

# Buttons
button_width, button_height = 90, 30
save_button = pygame.Rect(size - 100, 10, button_width, button_height)
clear_button = pygame.Rect(size - 100, 50, button_width, button_height)
font = pygame.font.Font(None, 24)

# Ensure the "Drawings" folder exists
if not os.path.exists("Drawings"):
    os.makedirs("Drawings")

def get_next_filename():
    i = 1
    while True:
        filename = f"Drawings/drawing_{i}.png"
        if not os.path.exists(filename):
            return filename
        i += 1

inference_button = pygame.Rect(size - 100, 90, button_width, button_height)
detect_button = pygame.Rect(size - 100, 130, button_width, button_height)
# Modify the draw_buttons function:
def draw_buttons():
    pygame.draw.rect(screen, GRAY, save_button)
    pygame.draw.rect(screen, GRAY, clear_button)
    pygame.draw.rect(screen, GRAY, inference_button)
    pygame.draw.rect(screen, GRAY, detect_button)
    save_text = font.render("Save", True, WHITE)
    clear_text = font.render("Clear", True, WHITE)
    inference_text = font.render("Run Inference", True, WHITE)
    hello_text = font.render("Detect", True, WHITE)
    screen.blit(save_text, (save_button.x + 10, save_button.y + 5))
    screen.blit(clear_text, (clear_button.x + 10, clear_button.y + 5))
    screen.blit(inference_text, (inference_button.x + 10, inference_button.y + 5))
    screen.blit(hello_text, (detect_button.x + 10, detect_button.y + 5))

def clear_board():
    drawing_surface.fill(WHITE)

def save_drawing():
    filename = get_next_filename()
    #pygame.image.save(drawing_surface, filename)
    print(f"Original drawing saved as {filename}")
    # Resize the image
    resized_image = pygame_resize_image(drawing_surface)
    # Save the resized image
    resized_filename = filename.replace('.png', '_28x28.png')
    Image.fromarray((resized_image * 255).astype(np.uint8)).save(resized_filename)
    print(f"Resized drawing (28x28) saved as {resized_filename}")

# Main game loop
drawing = False
last_pos = None
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                if save_button.collidepoint(event.pos):
                    save_drawing()
                elif clear_button.collidepoint(event.pos):
                    clear_board()
                elif inference_button.collidepoint(event.pos):
                    save_drawing()
                    cnn_run_single_inference(CNN_V1(),os.path.join(os.path.dirname(__file__), "../Drawings/drawing_1_28x28.png"))
                elif detect_button.collidepoint(event.pos):  # Add this block
                    bruteforce_detection(screen)
                else:
                    drawing = True
                    last_pos = event.pos
            elif event.button == 4:  # Scroll up
                pen_size = min(pen_size + 1, max_pen_size)
            elif event.button == 5:  # Scroll down
                pen_size = max(pen_size - 1, min_pen_size)
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:  # Left mouse button
                drawing = False
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                current_pos = event.pos
                if last_pos:
                    line(drawing_surface, last_pos, current_pos, pen_size)
                last_pos = current_pos

    # Clear the screen
    screen.fill(WHITE)

    # Copy the drawing surface to the main screen
    screen.blit(drawing_surface, (0, 0))

    # Draw buttons
    draw_buttons()

    # Draw the cursor to show current pen size
    pygame.draw.circle(screen, RED, pygame.mouse.get_pos(), pen_size // 2, 1)

    pygame.display.flip()