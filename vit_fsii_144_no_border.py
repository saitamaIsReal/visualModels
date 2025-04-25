#!/usr/bin/env python3
# ─── Monkey‐Patch für SHAPIQ: Border‐Trick ausschalten ─────────────────────
import shapiq.approximator.sampling as _sampling

def _no_border(self, sampling_budget: int) -> int:
    # Nie den Border‐Trick ausführen, Budget unverändert zurückgeben
    return sampling_budget

_sampling.CoalitionSampler.execute_border_trick = _no_border
# ─────────────────────────────────────────────────────────────────────────

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
patch_size      = 32
image_size      = 384
n_patches_per_row = image_size // patch_size
n_patches       = n_patches_per_row ** 2
SHOW_INDEX_GRID = True

# === Hilfsfunktionen ===
def mask_image(original_image: Image.Image, coalition: Union[np.ndarray, list]) -> Image.Image:
    img_np = np.array(original_image.resize((image_size, image_size))).copy()
    coalition = np.asarray(coalition, dtype=bool)
    for i, keep in enumerate(coalition):
        if keep:
            continue
        row, col = divmod(i, n_patches_per_row)
        y1, y2 = row * patch_size, (row + 1) * patch_size
        x1, x2 = col * patch_size, (col + 1) * patch_size
        img_np[y1:y2, x1:x2] = 128
    return Image.fromarray(img_np)

def patch_to_coords(patch_idx: int):
    row, col = divmod(patch_idx, n_patches_per_row)
    x1, y1 = col * patch_size, row * patch_size
    return (x1, y1, x1 + patch_size, y1 + patch_size)

def draw_patch_index_grid(img: Image.Image):
    draw = ImageDraw.Draw(img)
    for i in range(n_patches):
        x1, y1, x2, y2 = patch_to_coords(i)
        draw.rectangle([x1, y1, x2, y2], outline="gray", width=1)
        draw.text((x1 + 2, y1 + 2), str(i), fill="gray")
    img.show()

# value_function greift auf globale Variablen: model, processor, predicted_class, device

def value_function(coalitions: np.ndarray) -> np.ndarray:
    out = []
    for coalition in coalitions:
        img_masked = mask_image(image, coalition)
        inputs = processor(images=img_masked, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, predicted_class]
        out.append(logits.item())
    return np.array(out)

# === Hauptprogramm ===
def main():
    global processor, model, image, predicted_class, device

    # 1) Device und Modell laden
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device:", device)
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        print("CUDA device name:", torch.cuda.get_device_name(idx))

    processor = ViTImageProcessor.from_pretrained(model_name)
    model     = ViTForImageClassification.from_pretrained(model_name).to(device).eval()

    # 2) Bild laden
    resp  = requests.get(image_url)
    image = Image.open(BytesIO(resp.content)).convert("RGB").resize((image_size, image_size))

    # 3) Optional: Patch-Index-Grid anzeigen
    if SHOW_INDEX_GRID:
        draw_patch_index_grid(image.copy())

    # 4) Zielklasse bestimmen
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = int(logits.argmax(-1))
    print(f"Predicted class: {model.config.id2label[predicted_class]}")

    # 5) Quick-Check Full vs Empty
    val_full  = value_function([np.ones(n_patches)])
    val_empty = value_function([np.zeros(n_patches)])
    print(f"Logit Full  : {val_full[0]:.2f}")
    print(f"Logit Empty : {val_empty[0]:.2f}")

    # 6) Sampling-Gewichte (Einzelfälle + Paare)
    w = np.zeros(n_patches + 1)
    w[1] = 0.2
    w[2] = 0.8
    w /= w.sum()

    # 7) RegressionFSII aufsetzen und laufen
    approximator = RegressionFSII(
        n=n_patches,
        max_order=2,
        pairing_trick=False,
        sampling_weights=w,
        random_state=42
    )
    result = approximator.approximate(
        budget=11000,
        game=value_function,
        return_interactions=False
    )

    # 8) FSII-Werte holen und 2nd-order filtern
    fsii_map = result.dict_values
    second_order = {p: v for p, v in fsii_map.items() if len(p) == 2}

    # 9) Top-5 Paare ausgeben
    top5 = sorted(second_order.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    print("\nTop-5 normierte FSII-Werte (144 Patches):")
    for rank, (pair, val) in enumerate(top5, start=1):
        print(f" {rank}. Patch-Paar {pair} → FSII = {val:.6f}")

if __name__ == "__main__":
    main()
