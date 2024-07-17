import pygame
import sys
import os
import numpy as np
import torch
from PIL import Image
package_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, package_path)
from NN import LeNet5, CNN_V1, load_and_prepare_image



def resize_image(surface):
    # Convert the pygame surface to a PIL Image
    pil_string_image = pygame.image.tostring(surface, "RGB", False)
    pil_image = Image.frombytes("RGB", surface.get_size(), pil_string_image)
    # Resize the image to 28x28
    pil_image = pil_image.resize((28, 28), Image.LANCZOS)
    # Convert to grayscale
    pil_image = pil_image.convert('L')
    # Convert to numpy array
    numpy_image = np.array(pil_image)
    # Invert the colors (assuming drawing is black on white)
    numpy_image = 255 - numpy_image
    # Normalize the values to be between 0 and 1
    numpy_image = numpy_image / 255.0
    return numpy_image

# Initialize Pygame
pygame.init()

# Set up the drawing window
size = 600
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
pen_size = 2
min_pen_size = 1
max_pen_size = 50

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

hello_button = pygame.Rect(size - 100, 90, button_width, button_height)
# Add this function to display the "Hello" message:
def run_inference():
    model = LeNet5()
    # Load the model weights
    model_path = os.path.join(os.path.join(os.path.dirname(__file__),"../Models/CNN/lenet5/best_lenet5_weights.pth"))
    model.load_state_dict(torch.load(model_path))
    # Set the model to evaluation mode
    model.eval()
    image = load_and_prepare_image(os.path.join(os.path.dirname(__file__), "../Drawings/drawing_1_28x28.png"))
    image = image.unsqueeze(0)
    image = image.unsqueeze(0)
    output = model(image)
    # If you need to round the predictions (since you used a RoundLayer in the model)
    prediction = torch.round(output)
    print(prediction)

# Modify the draw_buttons function:
def draw_buttons():
    pygame.draw.rect(screen, GRAY, save_button)
    pygame.draw.rect(screen, GRAY, clear_button)
    pygame.draw.rect(screen, GRAY, hello_button)
    save_text = font.render("Save", True, WHITE)
    clear_text = font.render("Clear", True, WHITE)
    hello_text = font.render("Run Inference", True, WHITE)
    screen.blit(save_text, (save_button.x + 10, save_button.y + 5))
    screen.blit(clear_text, (clear_button.x + 10, clear_button.y + 5))
    screen.blit(hello_text, (hello_button.x + 10, hello_button.y + 5))

def clear_board():
    drawing_surface.fill(WHITE)

def save_drawing():
    filename = get_next_filename()
    #pygame.image.save(drawing_surface, filename)
    print(f"Original drawing saved as {filename}")
    # Resize the image
    resized_image = resize_image(drawing_surface)
    # Save the resized image
    resized_filename = filename.replace('.png', '_28x28.png')
    Image.fromarray((resized_image * 255).astype(np.uint8)).save(resized_filename)
    print(f"Resized drawing (28x28) saved as {resized_filename}")

def draw_line(surface, start, end, width):
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = max(abs(dx), abs(dy))
    for i in range(distance):
        x = int(start[0] + float(i) / distance * dx)
        y = int(start[1] + float(i) / distance * dy)
        pygame.draw.circle(surface, BLACK, (x, y), width // 2)

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
                elif hello_button.collidepoint(event.pos):
                    run_inference()
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
                    draw_line(drawing_surface, last_pos, current_pos, pen_size)
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