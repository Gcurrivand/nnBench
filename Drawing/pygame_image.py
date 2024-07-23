from PIL import Image
import pygame
import numpy as np
import torch
import os


def pygame_resize_image(surface):
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

def pygame_load_and_prepare_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale ('L' mode)
    if image.size != (28, 28):
        raise ValueError("Image must be 28x28 in size.")
    image_array = np.array(image)
    image_tensor = torch.tensor(image_array, dtype=torch.float32)
    return image_tensor.unsqueeze(0)

def pygame_rectangle_canvas(surface, x1, y1, x2, y2, filename):
    # Ensure the coordinates are in the correct order
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    # Calculate the width and height of the rectangle
    width = x2 - x1 + 1
    height = y2 - y1 + 1

    # Create a new surface for the rectangle
    rect_surface = pygame.Surface((width, height))

    # Copy the rectangular region to the new surface
    rect_surface.blit(surface, (0, 0), (x1, y1, width, height))

    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Save the surface as an image
    pygame.image.save(rect_surface, "Drawings/"+filename)
    print(f"Image saved as {filename}")

# Example usage:
# pygame.init()
# screen = pygame.display.set_mode((400, 300))
# screen.fill((255, 255, 255))  # Fill with white
# pygame.draw.circle(screen, (255, 0, 0), (200, 150), 100)  # Draw a red circle
# save_rectangle_from_pygame_canvas(screen, 100, 50, 300, 250, 'pygame_output.png')
# pygame.quit()