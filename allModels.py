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

# === Parameter & Device ===
models = [
    "google/vit-base-patch32-384",
    "facebook/deit-tiny-patch16-224",
    "akahana/vit-base-cats-vs-dogs",
    # füge hier weitere Modell-IDs von HuggingFace ein
]

image_url  = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on device:", device)
if device.type == "cuda":
    print(" CUDA device name:", torch.cuda.get_device_name())

# Hilfsfunktionen bleiben unverändert
def mask_image_grid(img: Image.Image, coalition: Union[np.ndarray, list],
                    image_size: int, n_patches_per_row: int, cell_h: int, cell_w: int) -> Image.Image:
    arr  = np.array(img.resize((image_size, image_size))).copy()
    coal = np.asarray(coalition, dtype=bool)
    for i, keep in enumerate(coal):
        if not keep:
            r, c = divmod(i, n_patches_per_row)
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            arr[y1:y2, x1:x2] = 128
    return Image.fromarray(arr)

def draw_grid(img: Image.Image, n_patches_per_row: int, cell_h: int, cell_w: int):
    draw = ImageDraw.Draw(img)
    for r in range(n_patches_per_row):
        for c in range(n_patches_per_row):
            x1, y1 = c*cell_w, r*cell_h
            x2, y2 = x1+cell_w, y1+cell_h
            draw.rectangle([x1, y1, x2, y2], outline="gray", width=1)
            idx = r * n_patches_per_row + c
            draw.text((x1+2, y1+2), str(idx), fill="gray")
    img.show(title="Patch-Index Grid")

def value_function(coalitions: np.ndarray, processor, model, device, image, predicted_class,
                   image_size, n_patches_per_row, cell_h, cell_w) -> np.ndarray:
    out = []
    for coalition in coalitions:
        masked = mask_image_grid(image, coalition, image_size, n_patches_per_row, cell_h, cell_w)
        batch  = processor(images=masked, return_tensors="pt").to(device)
        with torch.no_grad():
            logit = model(**batch).logits[0, predicted_class].item()
        out.append(logit)
    return np.array(out)

# === Hauptschleife über alle Modelle ===
for model_name in models:
    print(f"\n=== Auswertung für Modell: {model_name} ===")

    # 1) Prozessor & Modell laden
    processor = AutoImageProcessor.from_pretrained(model_name)
    model     = AutoModelForImageClassification.from_pretrained(model_name).to(device).eval()

    # 2) Bild- und Patch-Konfiguration auslesen
    patch_size        = model.config.patch_size
    image_size        = model.config.image_size
    n_patches_per_row = image_size // patch_size
    n_patches         = n_patches_per_row ** 2
    cell_h = cell_w = patch_size

    # 3) Bild laden
    resp  = requests.get(image_url)
    image = Image.open(BytesIO(resp.content)).convert("RGB").resize((image_size, image_size))

    # 4) Optional: Grid anzeigen
    draw_grid(image.copy(), n_patches_per_row, cell_h, cell_w)

    # 5) Zielklasse bestimmen
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = int(logits.argmax(-1))
    print(f"Predicted class: {model.config.id2label[predicted_class]}")

    # 6) Quick-Check Full vs Empty
    full  = value_function(np.array([np.ones(n_patches, bool)]),
                           processor, model, device, image, predicted_class,
                           image_size, n_patches_per_row, cell_h, cell_w)[0]
    empty = value_function(np.array([np.zeros(n_patches, bool)]),
                           processor, model, device, image, predicted_class,
                           image_size, n_patches_per_row, cell_h, cell_w)[0]
    print(f"  Logit Full  : {full:.2f}")
    print(f"  Logit Empty : {empty:.2f}")

    # 7) FSII-Approximation
    approximator = RegressionFSII(
        n=n_patches,
        max_order=2,
        pairing_trick=False,
        random_state=42
    )
    result = approximator.approximate(
        budget= 5,
        game=lambda coal: value_function(coal,
                                         processor, model, device, image, predicted_class,
                                         image_size, n_patches_per_row, cell_h, cell_w)
    )

    # 8) Top-5 2nd-Order Interaktionen ausgeben
    fsii_map = result.dict_values
    second  = {p: v for p, v in fsii_map.items() if len(p) == 2}
    top5    = sorted(second.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    print("  Top-5 FSII 2nd-Order:")
    for rank, (pair, val) in enumerate(top5, start=1):
        print(f"   {rank}. Pair {pair} → {val:.4f}")

    # 9) Visualisierung
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
    vis.show(title=f"Top-5 Interactions – {model_name}")

print("\nFertig mit allen Modellen.")
