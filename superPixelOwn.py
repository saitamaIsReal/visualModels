#!/usr/bin/env python3
# ─── Monkey-Patch für SHAPIQ: Border-Trick ausschalten ────────────────────
import shapiq.approximator.sampling as _sampling
def _no_border(self, sampling_budget: int) -> int:
    # Niemals Border-Trick ausführen
    return sampling_budget
_sampling.CoalitionSampler.execute_border_trick = _no_border
# ──
import sys
import torch
import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image, ImageDraw
import requests
from io import BytesIO
from typing import Union
from shapiq.approximator.regression import RegressionFSII

# === Parameter ===
model_name      = "google/vit-base-patch32-384"
image_url       = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"
image_size      = 384
# Grid-Dimensionen (z.B. 7×7)
grid_rows       = 7
grid_cols       = 7
n_patches       = grid_rows * grid_cols
SHOW_INDEX_GRID = True

# Zellgröße berechnen
cell_h = image_size // grid_rows
cell_w = image_size // grid_cols

# === Hilfsfunktionen ===

def mask_image_grid(img: Image.Image, coalition: Union[np.ndarray, list]) -> Image.Image:
    """
    Maskiert einheitliche rechteckige Zellen im Grid.
    coalition: Array der Länge n_patches (grid_rows*grid_cols), bools.
    """
    img_resized = img.resize((image_size, image_size))
    arr = np.array(img_resized).copy()
    coal = np.asarray(coalition, dtype=bool)
    for idx, keep in enumerate(coal):
        if keep:
            continue
        r = idx // grid_cols
        c = idx % grid_cols
        y1, y2 = r * cell_h, (r + 1) * cell_h
        x1, x2 = c * cell_w, (c + 1) * cell_w
        arr[y1:y2, x1:x2] = 128
    return Image.fromarray(arr)


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
    img.show(title="Patch-Index Grid")


def value_function(coalitions: np.ndarray) -> np.ndarray:
    out = []
    for coalition in coalitions:
        masked = mask_image_grid(image, coalition)
        inputs = processor(images=masked, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, predicted_class]
        out.append(logits.item())
    return np.array(out)


# === Hauptprogramm ===
def main():
    global processor, model, image, predicted_class, device

    # Device & Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device:", device)
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        print("CUDA device name:", torch.cuda.get_device_name(idx))

    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name).to(device).eval()

    # Bild laden
    resp = requests.get(image_url)
    image = Image.open(BytesIO(resp.content)).convert("RGB").resize((image_size, image_size))

    # Grid anzeigen
    if SHOW_INDEX_GRID:
        draw_grid(image.copy())

    # Zielklasse bestimmen
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = int(logits.argmax(-1))
    print(f"Predicted class: {model.config.id2label[predicted_class]}")

    # Quick-Check Full vs Empty
    full = value_function([np.ones(n_patches)])[0]
    empty = value_function([np.zeros(n_patches)])[0]
    print(f"Logit Full  : {full:.2f}")
    print(f"Logit Empty : {empty:.2f}")

    # Sampling-Gewichte (Singles + Paare)
    w = np.zeros(n_patches + 1)
    w[1] = 0.2
    w[2] = 0.8
    w /= w.sum()

    # RegressionFSII
    approximator = RegressionFSII(
        n=n_patches,
        max_order=2,
        pairing_trick=False,
        sampling_weights=w,
        random_state=42
    )
    result = approximator.approximate(
        budget=100,
        game=value_function,
        return_interactions=False
    )

    # FSII auslesen
    fsii_map = result.dict_values
    second_order = {p: v for p, v in fsii_map.items() if len(p) == 2}
    top5 = sorted(second_order.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    print("\nTop-5 normierte FSII-Werte (Grid):")
    for rank, (pair, val) in enumerate(top5, start=1):
        print(f" {rank}. Grid-Paar {pair} → FSII = {val:.6f}")

        # === Visualisiere die Top-5 Paare im Image ===
    # Farben für die fünf Paare
    colors = ["red", "blue", "green", "yellow", "purple"]
    img_vis = image.copy().resize((image_size, image_size))
    draw = ImageDraw.Draw(img_vis)

    # Zeichne Rechtecke um jede Zelle der Top-5 Paare
    for idx, (i, j) in enumerate([pair for pair, _ in top5]):
        # erste Zelle
        r_i, c_i = divmod(i, grid_cols)
        x1_i, y1_i = c_i*cell_w,    r_i*cell_h
        x2_i, y2_i = x1_i+cell_w,   y1_i+cell_h
        draw.rectangle([x1_i, y1_i, x2_i, y2_i], outline=colors[idx], width=3)
        # zweite Zelle
        r_j, c_j = divmod(j, grid_cols)
        x1_j, y1_j = c_j*cell_w,    r_j*cell_h
        x2_j, y2_j = x1_j+cell_w,   y1_j+cell_h
        draw.rectangle([x1_j, y1_j, x2_j, y2_j], outline=colors[idx], width=3)
        # Nummern-Overlay
        draw.text((x1_i+2, y1_i+2), str(idx+1), fill=colors[idx])
        draw.text((x1_j+2, y1_j+2), str(idx+1), fill=colors[idx])

    # Und jetzt anzeigen
    img_vis.show(title="Top-5 Grid-Interaktionen")

if __name__ == "__main__":
    main()
