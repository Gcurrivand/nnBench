from PIL import Image
import pygame
import numpy as np
import os


def pygame_resize_image(surface):
    pil_string_image = pygame.image.tostring(surface, "RGB", False)
    pil_image = Image.frombytes("RGB", surface.get_size(), pil_string_image)
    pil_image = pil_image.resize((28, 28), Image.LANCZOS)
    pil_image = pil_image.convert('L')
    numpy_image = np.array(pil_image)
    numpy_image = 255 - numpy_image
    numpy_image = numpy_image / 255.0
    return numpy_image

def pygame_rectangle_extract(surface, x1, y1, x2, y2, filename):
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    width = x2 - x1
    height = y2 - y1
    rect_surface = pygame.Surface((width, height))
    rect_surface.blit(surface, (0, 0), (x1, y1, width, height))
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)
    pygame.image.save(rect_surface, filename)
    print(f"Image saved as {filename}")

# Example usage:
# pygame.init()
# screen = pygame.display.set_mode((400, 300))
# screen.fill((255, 255, 255))  # Fill with white
# pygame.draw.circle(screen, (255, 0, 0), (200, 150), 100)  # Draw a red circle
# save_rectangle_from_pygame_canvas(screen, 100, 50, 300, 250, 'pygame_output.png')
# pygame.quit()