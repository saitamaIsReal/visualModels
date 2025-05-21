#!/usr/bin/env python3
import sys
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image, ImageDraw
import requests
from io import BytesIO
from typing import Union
from shapiq.approximator.regression import RegressionFSII
from shapiq.interaction_values import InteractionValues
from IPython.display import display
from shapiq.plot.upset import upset_plot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# Model selection
models = [
    "google/vit-base-patch32-384",
    "facebook/deit-tiny-patch16-224",
    "akahana/vit-base-cats-vs-dogs",
]

image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device:", device)
if device.type == "cuda":
    print(" CUDA device name:", torch.cuda.get_device_name())


# Helper Functions
def create_heatmap_overlay(image: Image.Image, shapley_values: np.ndarray, n_patches_per_row: int, cell: int) -> Image.Image:
    norm_vals = (shapley_values - np.min(shapley_values)) / (np.ptp(shapley_values) + 1e-8)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for idx, val in enumerate(norm_vals):
        r, c = divmod(idx, n_patches_per_row)
        x1, y1 = c * cell, r * cell
        x2, y2 = x1 + cell, y1 + cell
        red = int(val * 255)
        draw.rectangle([x1, y1, x2, y2], fill=(red, 0, 0, 120))
    return Image.alpha_composite(image.convert("RGBA"), overlay)

def combine_side_by_side(images):
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)
    combined = Image.new("RGBA", (total_width, max_height))
    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width
    return combined

def mask_image_grid(
    img: Image.Image,
    coalition: Union[np.ndarray, list],
    image_size: int,
    n_patches_per_row: int,
    cell: int
) -> Image.Image:
    """
    Graying out patches where coalition[i] is False (0).
    """
    arr  = np.array(img.resize((image_size, image_size))).copy()
    coal = np.asarray(coalition, dtype=bool)
    for i, keep in enumerate(coal):
        if not keep:
            r, c = divmod(i, n_patches_per_row)
            y1, y2 = r*cell, (r+1)*cell
            x1, x2 = c*cell, (c+1)*cell
            arr[y1:y2, x1:x2] = 128
    return Image.fromarray(arr)

def draw_grid(
    img: Image.Image,
    n_patches_per_row: int,
    cell: int
):
    """
    Draw numeric patch-grid for visualization.
    """
    draw = ImageDraw.Draw(img)
    for r in range(n_patches_per_row):
        for c in range(n_patches_per_row):
            x1, y1 = c*cell, r*cell
            x2, y2 = x1+cell, y1+cell
            draw.rectangle([x1,y1,x2,y2], outline="gray", width=1)
            idx = r*n_patches_per_row + c
            draw.text((x1+2, y1+2), str(idx), fill="gray")
    display(img)

def value_function(
    coalitions: np.ndarray,
    processor,
    model,
    device,
    image,
    predicted_class,
    image_size,
    n_patches_per_row,
    cell
) -> np.ndarray:
    """
    Given array (n_coalitions, n_patches), return logits for `predicted_class`.
    """
    out = []
    for coalition in coalitions:
        masked = mask_image_grid(image, coalition, image_size, n_patches_per_row, cell)
        batch  = processor(images=masked, return_tensors="pt").to(device)
        with torch.no_grad():
            logit = model(**batch).logits[0, predicted_class].item()
        out.append(logit)
    return np.array(out)

#=======MAIN========
heatmaps_per_model = []  # Liste vorbereiten
upset_plots_per_model = []  # Speichert die UpSet-Plot-Bilder

for model_name in models:
    print(f"\n--- Evaluating {model_name} ---")

# 1) load processor & model
    processor = AutoImageProcessor.from_pretrained(model_name)
    model     = AutoModelForImageClassification.from_pretrained(model_name).to(device).eval()

# 2) extract patch & image config
    patch_size        = model.config.patch_size
    image_size        = model.config.image_size
    n_patches_per_row = image_size // patch_size
    n_patches         = n_patches_per_row**2
    cell = patch_size

# 3) load & show grid
    resp  = requests.get(image_url)
    image = Image.open(BytesIO(resp.content)).convert("RGB").resize((image_size,image_size))
    draw_grid(image.copy(), n_patches_per_row, cell)

 # 4) determine target class
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = int(logits.argmax(-1))
    print("Predicted class:", model.config.id2label[predicted_class])
    print("predicted logits:", logits[0,predicted_class])

# 5) quick-check full vs. empty
    full  = value_function(
        np.array([np.ones(n_patches, bool)]),
        processor, model, device, image,
        predicted_class,
        image_size, n_patches_per_row, cell
    )[0]
    empty = value_function(
        np.array([np.zeros(n_patches, bool)]),
        processor, model, device, image,
        predicted_class,
        image_size, n_patches_per_row, cell
    )[0]
    print(f" Logit full:  {full:.2f}")
    print(f" Logit empty: {empty:.2f}")


# 6) FSII approx
    approximator = RegressionFSII(
        n=n_patches,
        max_order=2,
        pairing_trick=False,
        random_state=42
    )
    result = approximator.approximate(
        budget=100,
        game=lambda c: value_function(
            c,
            processor, model, device, image,
            predicted_class,
            image_size, n_patches_per_row, cell
        )
    )

    # === Heatmap (nur 1st-Order Shapley Values) ===
    first_order = result.get_n_order(order=1)
    shapley_values = np.array([first_order[(i,)] for i in range(n_patches)])
    heatmap_img = create_heatmap_overlay(image, shapley_values, n_patches_per_row, cell)
    heatmaps_per_model.append(heatmap_img)

    # === UpSet Plot (2nd-Order Interactions) ===
    # 2nd-Order Interaktionen laden
    second_order = result.get_n_order(order=2)

    # Top 10 nach Betrag
    top10_items = sorted(second_order.dict_values.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    top10_pairs = [item[0] for item in top10_items]
    top10_values = [item[1] for item in top10_items]

    # Max Player-ID korrekt setzen (z. B. 191 für 14×14)
    max_id = max(i for pair in top10_pairs for i in pair)

    subset = InteractionValues(
        values=np.array(top10_values),
        index="sparse",
        n_players=max_id + 1,
        max_order=2,
        min_order=2,
        interaction_lookup={pair: idx for idx, pair in enumerate(top10_pairs)},
        baseline_value=0.0
    )
    #=========UPSET PLOT PRINT============
    fig = upset_plot(subset, show=False)
    fig.suptitle(f"UpSet Plot – Top 10 ({model_name})", fontsize=10)

    # Bild als PIL.Image umwandeln
    canvas = FigureCanvas(fig)
    fig.set_canvas(canvas)  # sehr wichtig!
    canvas.draw()

    width, height = canvas.get_width_height()
    image_bytes = canvas.buffer_rgba()  # nicht tostring_rgb!
    upset_img = Image.frombuffer("RGBA", (width, height), image_bytes, "raw", "RGBA", 0, 1)
    upset_img.save(f"debug_upset_{model_name}.png")

    # Links 80 Pixel abschneiden
    cropped_img = upset_img.crop((80, 0, upset_img.width, upset_img.height))
    cropped_img.save(f"debug_cropped_{model_name}.png")
    upset_plots_per_model.append(cropped_img)
    print("Upset width:", upset_img.width, "height:", upset_img.height)


    # Zur Liste hinzufügen
    upset_plots_per_model.append(upset_img)
    plt.close(fig)

#=========HEAT PRINT==========
# 🔁 Standardgröße definieren (z. B. 384x384, weil ViT damit arbeitet)
standard_size = (384, 384)
# 📐 Alle Heatmaps auf gleiche Größe bringen
resized_heatmaps = [img.resize(standard_size) for img in heatmaps_per_model]
# 🔗 Jetzt sauber nebeneinander kombinieren
final_img = combine_side_by_side(resized_heatmaps)
final_img.save("vergleich_heatmaps.png")
final_img.show()

#=========UPSET PLOT BILDER KOMBINIEREN==========
# 🔁 Einheitliche Größe (z. B. erste Plotgröße als Referenz)
upset_standard_size = upset_plots_per_model[0].size
resized_upsets = [img.resize(upset_standard_size) for img in upset_plots_per_model]
combined_upset_img = combine_side_by_side(resized_upsets)
combined_upset_img.save("vergleich_upsetplots.png")
combined_upset_img.show()

