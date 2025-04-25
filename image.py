from PIL import Image, ImageDraw
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np


image_size = 384
grid_rows = 7
grid_cols = 7
cell_w = image_size // grid_cols
cell_h = image_size // grid_cols



def draw_grid(img: Image.Image):
    """Zeichnet das numerische Grid zur Visualisierung."""
    draw = ImageDraw.Draw(img)
    for r in range(grid_rows):
        for c in range(grid_cols):
            x1, y1 = c*cell_w, r*cell_h
            x2, y2 = x1+cell_w, y1+cell_h
            draw.rectangle([x1, y1, x2, y2], outline="gray", width=1)
            idx = r * grid_cols + c
            draw.text((x1+2, y1+2), str(idx), fill="gray")
    # Matplotlib-Visualisierung statt img.show()
    arr = np.array(img)
    plt.imshow(arr)
    plt.axis("off")
    plt.title("Patch-Index Grid")
    plt.show()

if __name__ == "__main__":
    image = (
    Image
    .open("images/dog.jpg")
    .convert("RGB")
    .resize((image_size, image_size))
)

    draw_grid(image)