#!/usr/bin/env python3
import sys
import torch
import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image, ImageDraw
import requests
from io import BytesIO
from typing import Union
from shapiq.approximator.regression import RegressionFSII

# === Parameter & Device ===
model_name = "google/vit-base-patch32-384"
image_url  = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device:", device)
if device.type == "cuda":
    print(" CUDA device name:", torch.cuda.get_device_name())

# === Model & Grid-Konfiguration auslesen ===
processor = ViTImageProcessor.from_pretrained(model_name)
model     = ViTForImageClassification.from_pretrained(model_name).to(device).eval()

# Bild- und Patch-Größen aus Modell-Config
patch_size        = model.config.patch_size            # z.B. 32
image_size        = model.config.image_size            # z.B. 384
n_patches_per_row = image_size // patch_size           # => 12
n_patches         = n_patches_per_row ** 2             # => 144

cell_h = cell_w = patch_size  # da wir auf (image_size × image_size) resize

# === Hilfsfunktionen ===

def mask_image_grid(img: Image.Image, coalition: Union[np.ndarray, list]) -> Image.Image:
    """
    Maskiert die Patches, für die coalition[i] == False ist.
    coalition: bool-Array der Länge n_patches.
    """
    arr  = np.array(img.resize((image_size, image_size))).copy()
    coal = np.asarray(coalition, dtype=bool)
    for i, keep in enumerate(coal):
        if not keep:
            r, c = divmod(i, n_patches_per_row)
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            arr[y1:y2, x1:x2] = 128
    return Image.fromarray(arr)

def draw_grid(img: Image.Image):
    """Zeigt das Patch-Grid mit Indizes."""
    draw = ImageDraw.Draw(img)
    for r in range(n_patches_per_row):
        for c in range(n_patches_per_row):
            x1, y1 = c*cell_w, r*cell_h
            x2, y2 = x1+cell_w, y1+cell_h
            draw.rectangle([x1, y1, x2, y2], outline="gray", width=1)
            idx = r * n_patches_per_row + c
            draw.text((x1+2, y1+2), str(idx), fill="gray")
    img.show(title="Patch-Index Grid")

def value_function(coalitions: np.ndarray) -> np.ndarray:
    """
    Erwartet ein Array shape (n_coalitions, n_patches).
    Gibt logits[target_class] je Coalition zurück.
    """
    out = []
    for coalition in coalitions:
        masked = mask_image_grid(image, coalition)
        batch  = processor(images=masked, return_tensors="pt").to(device)
        with torch.no_grad():
            logit = model(**batch).logits[0, predicted_class].item()
        out.append(logit)
    return np.array(out)

# === Hauptprogramm ===

def main():
    global image, predicted_class

    # 1) Bild laden
    resp  = requests.get(image_url)
    image = Image.open(BytesIO(resp.content)).convert("RGB").resize((image_size, image_size))
    
    # 2) Optional: Grid anzeigen
    draw_grid(image.copy())

    # 3) Zielklasse bestimmen (full image)
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = int(logits.argmax(-1))
    print(f"Predicted class: {model.config.id2label[predicted_class]}")

    # 4) Quick-Check Full vs Empty
    full  = value_function([np.ones(n_patches, bool)])[0]
    empty = value_function([np.zeros(n_patches, bool)])[0]
    print(f"Logit Full  : {full:.2f}")
    print(f"Logit Empty : {empty:.2f}")

    # 5) FSII-Approximation (ohne eigenes Sampling-Gewicht, ohne Monkey-Patch)
    approximator = RegressionFSII(
        n=n_patches,
        max_order=2,
        pairing_trick=True,    # kannst hier True/False probieren
        random_state=42
    )
    result = approximator.approximate(
        budget=300,
        game=value_function
    )

    # 6) Top-5 2nd-Order Interaktionen ausgeben
    fsii_map    = result.dict_values
    second      = {p: v for p, v in fsii_map.items() if len(p) == 2}
    top5        = sorted(second.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    print("\nTop-5 FSII 2nd-Order:")
    for rank, (pair, val) in enumerate(top5, start=1):
        print(f" {rank}. Patch-Paar {pair} → FSII = {val:.4f}")

    # 7) Top-5 auf dem Bild visualisieren
    colors = ["red","blue","green","yellow","purple"]
    vis    = image.copy().resize((image_size,image_size))
    draw   = ImageDraw.Draw(vis)
    drawn  = set()
    for idx,(i,j) in enumerate([p for p,_ in top5]):
        col = colors[idx]
        for patch in (i,j):
            if patch in drawn: continue
            drawn.add(patch)
            r, c = divmod(patch, n_patches_per_row)
            x1, y1 = c*cell_w, r*cell_h
            x2, y2 = x1+cell_w, y1+cell_h
            draw.rectangle([x1,y1,x2,y2], outline=col, width=3)
            draw.text((x1+2,y1+2), str(patch), fill=col)
    vis.show(title="Top-5 Interactions")

if __name__ == "__main__":
    main()
