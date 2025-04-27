#!/usr/bin/env python
# coding: utf-8

# # FSII Analysis with ViT and 7×7 Patch Grid
#
# This notebook demonstrates how to:
# 1. Split an input image into a 7×7 patch grid and visualize the indices.
# 2. Define a masking function that applies a gray mask to selected patches.
# 3. Compute model logits on masked images for a chosen target class.
# 4. Approximate 2nd-order Faithful Shapley Interaction Indices (FSII) with SHAPIQ.
# 5. Display the top-5 interacting patch pairs on the image.

# 1. Imports & Monkey‐Patch to disable SHAPIQ’s border-trick
import shapiq.approximator.sampling as _sampling

def _no_border(self, sampling_budget: int) -> int:
    # Never apply SHAPIQ's internal "border-trick", keep the full budget
    return sampling_budget

_sampling.CoalitionSampler.execute_border_trick = _no_border

import sys, os
import torch
import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image, ImageDraw
import requests
from io import BytesIO
from shapiq.approximator.regression import RegressionFSII

# 2. Global configuration
model_name   = "google/vit-base-patch32-384"
image_url    = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"
image_size   = 384  # final resized image resolution
grid_rows    = grid_cols = 7
n_patches    = grid_rows * grid_cols

# Choose device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(device)}")

# 3. Load & display the image with a 7×7 grid
resp  = requests.get(image_url)
image = Image.open(BytesIO(resp.content)).convert("RGB").resize((image_size, image_size))

# Compute cell dimensions and draw grid
cell_h, cell_w = image_size//grid_rows, image_size//grid_cols
grid_viz     = image.copy()
d = ImageDraw.Draw(grid_viz)
for r in range(grid_rows):
    for c in range(grid_cols):
        x1, y1 = c*cell_w, r*cell_h
        d.rectangle([x1, y1, x1+cell_w, y1+cell_h], outline="gray", width=1)
        idx = r*grid_cols + c
        d.text((x1+2, y1+2), str(idx), fill="gray")
grid_viz

# 4. Define the masking function
def mask_image_grid(img: Image.Image, coalition: np.ndarray) -> Image.Image:
    """
    Apply gray mask to patches where coalition[i] is False.
    coalition: boolean array of length n_patches.
    """
    arr = np.array(img.copy())
    mask = np.asarray(coalition, bool)
    for i, keep in enumerate(mask):
        if not keep:
            r, c = divmod(i, grid_cols)
            y1, y2 = r*cell_h, (r+1)*cell_h
            x1, x2 = c*cell_w, (c+1)*cell_w
            arr[y1:y2, x1:x2] = 128
    return Image.fromarray(arr)

# 5. Show the first 6 *actually sampled* coalitions
from IPython.display import display

# configure a tiny FSII‐approximator that retains raw coalitions
small_approx = RegressionFSII(
    n=n_patches,
    max_order=2,
    pairing_trick=False,
    sampling_weights=w,      # same w you’ll use later
    random_state=42
)
# run with return_interactions=True so we get coalitions_matrix
_ = small_approx.approximate(
    budget=50,
    game=value_fn,
    return_interactions=True
)

# coalitions_matrix is now shape (budget, n_patches)
coalitions = small_approx._sampler.coalitions_matrix

print("First 6 sampled coalitions (True=keep, False=mask):")
for idx in range(6):
    mask = coalitions[idx]
    print(f"‒ Sample #{idx:2d}, keeps {mask.sum()} patches")
    display(mask_image_grid(image, mask).resize((192,192)))


# 6. Load ViT processor & model, select target class
processor = ViTImageProcessor.from_pretrained(model_name)
model     = ViTForImageClassification.from_pretrained(model_name).to(device).eval()

inputs = processor(images=image, return_tensors="pt").to(device)
with torch.no_grad():
    logits = model(**inputs).logits
    target_class = int(logits.argmax(-1))
print(f"Target class: {model.config.id2label[target_class]}")

# 7. Define value function for SHAPIQ
def value_fn(coalitions: list[np.ndarray]) -> np.ndarray:
    """
    Compute logits for each masked image in `coalitions`.
    Returns an array of shape (len(coalitions),).
    """
    out = []
    for coal in coalitions:
        masked = mask_image_grid(image, coal)
        batch  = processor(images=masked, return_tensors="pt").to(device)
        with torch.no_grad():
            logit = model(**batch).logits[0, target_class].item()
        out.append(logit)
    return np.array(out)

# 8. Compare full vs empty
print("Logit (full):",  value_fn([np.ones(n_patches)])[0])
print("Logit (empty):", value_fn([np.zeros(n_patches)])[0])

# 9. Mini-run of FSII approximation for demonstration
w = np.zeros(n_patches+1)
w[1], w[2] = 0.2, 0.8  # 20% singles, 80% pairs
w /= w.sum()

approximator = RegressionFSII(
    n=n_patches,
    max_order=2,
    pairing_trick=False,
    sampling_weights=w,
    random_state=42
)
res = approximator.approximate(
    budget=300,            # small demo
    game=value_fn,
    return_interactions=False
)

# 10. Extract and display top-5 2nd-order FSII scores
fsii_map = res.dict_values
pairs   = {p:v for p,v in fsii_map.items() if len(p)==2}
top5    = sorted(pairs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
for rank,(p,val) in enumerate(top5,1):
    print(f"{rank}. Pair {p} → FSII = {val:.4f}")

# 11. Visualize top pairs on image
colors = ["red","blue","green","yellow","purple"]
vis = image.copy().resize((image_size,image_size))
d   = ImageDraw.Draw(vis)
drawn = set()
for idx,(i,j) in enumerate([p for p,_ in top5]):
    col = colors[idx]
    for patch in (i,j):
        if patch in drawn: continue
        drawn.add(patch)
        r,c = divmod(patch, grid_cols)
        x1,y1 = c*cell_w, r*cell_h
        x2,y2 = x1+cell_w, y1+cell_h
        d.rectangle([x1,y1,x2,y2], outline=col, width=3)
        d.text((x1+2,y1+2), str(patch), fill=col)

vis

# **Notes**
# - **Monkey-Patch**: Disables the border-trick to avoid dimension errors.
# - **sampling_weights**: Controls the fraction of budget drawn for single vs. pair masks.
# - **return_interactions=False**: Return only final dict_values of FSII scores (no raw samples).
