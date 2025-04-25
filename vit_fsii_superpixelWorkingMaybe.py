#!/usr/bin/env python3
import shapiq.approximator.sampling as _sampling

def _no_border(self, sampling_budget: int) -> int:
    # niemals Border-Trick ausführen, einfach Budget unverändert zurückgeben
    return sampling_budget

_sampling.CoalitionSampler.execute_border_trick = _no_border
import sys
import torch
import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image, ImageDraw
import requests
from io import BytesIO
from typing import Union
from skimage.segmentation import slic
from skimage.color import rgb2lab
from skimage.util import img_as_float
from shapiq.approximator.regression import RegressionFSII

# === Parameter ===
model_name      = "google/vit-base-patch32-384"
image_url       = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"
image_size      = 384
n_superpixels   = 49        # Anzahl Superpixel
compactness     = 10        # für SLIC
SHOW_INDEX_GRID = False

# === Hilfsfunktionen ===
def compute_superpixels(image: Image.Image):
    # Convert to float RGB and Lab for SLIC
    arr = img_as_float(np.array(image.resize((image_size, image_size))))
    labels = slic(rgb2lab(arr), n_segments=n_superpixels, compactness=compactness)
    return labels

# wird nach compute_superpixels gesetzt
def mask_image_superpixel(original_image: Image.Image, coalition: Union[np.ndarray, list]) -> Image.Image:
    # coalition length == n_superpixels
    img_np = np.array(original_image.resize((image_size, image_size))).copy()
    coalition = np.asarray(coalition, dtype=bool)
    for i, keep in enumerate(coalition):
        if keep:
            continue
        img_np[superpixel_labels == i] = 128  # Grauwert
    return Image.fromarray(img_np)

# optional: draw superpixel boundaries
def draw_superpixel_grid(img: Image.Image, labels: np.ndarray):
    draw = ImageDraw.Draw(img)
    # outline each superpixel
    for region in range(labels.max()+1):
        mask = labels == region
        ys, xs = np.nonzero(mask)
        # bounding box
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        draw.rectangle([x1, y1, x2, y2], outline="gray", width=1)
    img.show()

# value_function greift auf model, processor, predicted_class, device, superpixel_labels zu

def value_function(coalitions: np.ndarray) -> np.ndarray:
    out = []
    for coalition in coalitions:
        masked = mask_image_superpixel(image, coalition)
        inputs = processor(images=masked, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, predicted_class]
        out.append(logits.item())
    return np.array(out)

# === Hauptprogramm ===
def main():
    global processor, model, image, predicted_class, device, superpixel_labels

    # 1) Device und Model laden
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device:", device)

    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        print("CUDA device name:", torch.cuda.get_device_name(idx))

    processor = ViTImageProcessor.from_pretrained(model_name)
    model     = ViTForImageClassification.from_pretrained(model_name).to(device).eval()

    # 2) Bild laden und Superpixel berechnen
    resp  = requests.get(image_url)
    image = Image.open(BytesIO(resp.content)).convert("RGB").resize((image_size, image_size))
    superpixel_labels = compute_superpixels(image)

    # 3) Optional: Superpixel-Gitter anzeigen
    if SHOW_INDEX_GRID:
        draw_superpixel_grid(image.copy(), superpixel_labels)

    # 4) Zielklasse bestimmen
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = int(logits.argmax(-1))
    print(f"Predicted class: {model.config.id2label[predicted_class]}")

    # 5) Quick-Check Full vs Empty
    full  = value_function([np.ones(n_superpixels)])[0]
    empty = value_function([np.zeros(n_superpixels)])[0]
    print(f"Logit Full  : {full:.2f}")
    print(f"Logit Empty : {empty:.2f}")

    # 6) Sampling-Gewichte (Singles + Paare)
    w = np.zeros(n_superpixels+1)
    w[1] = 0.2
    w[2] = 0.8

    # 7) RegressionFSII konfigurieren
    approximator = RegressionFSII(
        n=n_superpixels,
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

    # 8) FSII-Werte auslesen und Paare filtern
    fsii_map = result.dict_values
    second_order = {p: v for p, v in fsii_map.items() if len(p)==2}

    # 9) Top-5 Paare ausgeben
    top5 = sorted(second_order.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    print("\nTop-5 normierte FSII-Werte (Superpixel):")
    for rank, (pair, val) in enumerate(top5, start=1):
        print(f" {rank}. Superpixels {pair} → FSII = {val:.6f}")

    colors = ["red","blue","green","yellow","purple"]
    img_vis = image.copy()
    draw = ImageDraw.Draw(img_vis)
    for idx, (i,j) in enumerate([pair for pair, _ in top5]):
        # Bounding box um Superpixel i und j
        mask_i = (superpixel_labels == i)
        ys, xs = np.nonzero(mask_i); x1,x2=xs.min(),xs.max(); y1,y2=ys.min(),ys.max()
        draw.rectangle([x1,y1,x2,y2], outline=colors[idx], width=3)
        mask_j = (superpixel_labels == j)
        ys, xs = np.nonzero(mask_j); x1,x2=xs.min(),xs.max(); y1,y2=ys.min(),ys.max()
        draw.rectangle([x1,y1,x2,y2], outline=colors[idx], width=3)
        # Labelnummer
        draw.text((x1, y1), str(idx+1), fill=colors[idx])
    img_vis.show()

if __name__ == "__main__":
    main()
