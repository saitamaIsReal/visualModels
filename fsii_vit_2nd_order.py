import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image, ImageDraw
import numpy as np
from typing import Union
from shapiq import RegressionFSII
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import sys

sys.stdout.reconfigure(encoding='utf-8')

# === Parameter ===
model_name = "google/vit-base-patch32-384"
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"
patch_size = 32
image_size = 384
n_patches_per_row = image_size // patch_size
n_patches = n_patches_per_row ** 2
SWAP_ROW_COL = False
SHOW_INDEX_GRID = True

# === Hilfsfunktionen ===
def mask_image(original_image: Image.Image, coalition: Union[np.ndarray, list]) -> Image.Image:
    img = original_image.resize((384, 384))
    img_np = np.array(img).copy()
    coalition = np.array(coalition)
    for i, keep in enumerate(coalition):
        if keep == 1:
            continue
        row, col = divmod(i, n_patches_per_row)
        if SWAP_ROW_COL:
            row, col = col, row
        y1, y2 = row * patch_size, (row + 1) * patch_size
        x1, x2 = col * patch_size, (col + 1) * patch_size
        img_np[y1:y2, x1:x2] = [128, 128, 128]
    return Image.fromarray(img_np)

def patch_to_coords(patch_idx): #patch in coords umrechnen
    row, col = divmod(patch_idx, n_patches_per_row)
    if SWAP_ROW_COL:
        row, col = col, row
    x1, y1 = col * patch_size, row * patch_size
    x2, y2 = x1 + patch_size, y1 + patch_size
    return (x1, y1, x2, y2)

def value_function(coalitions: np.ndarray) -> np.ndarray:
    values = []
    for idx, coalition in enumerate(coalitions):
        masked_img = mask_image(image, coalition)
        inputs = processor(images=masked_img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logit = outputs.logits[0, predicted_class].item()
        values.append(logit)
    return np.array(values)

def draw_patch_index_grid(img: Image.Image): #grid zeichnen
    draw = ImageDraw.Draw(img)
    for i in range(n_patches):
        x1, y1, x2, y2 = patch_to_coords(i)
        draw.rectangle([x1, y1, x2, y2], outline="gray", width=1)
        draw.text((x1 + 2, y1 + 2), str(i), fill="gray")
    img.show(title="Patch-Index-Grid")

def analyze_top_interactions(values, top_k=5): #analyse der top 5 interaktionen
    print("\n--- Detaillierte Interaktionsanalyse der Top-{} Paare ---".format(top_k))
    second_order = {k: v for k, v in values.interaction_lookup.items() if len(k) == 2}
    top_interactions = sorted(second_order.items(), key=lambda kv: abs(np.mean(kv[1])), reverse=True)[:top_k]
    base_logit = value_function([np.ones(n_patches)])[0]
    for idx, ((i, j), val_array) in enumerate(top_interactions):
        coalition_none = np.ones(n_patches)
        coalition_i = coalition_none.copy(); coalition_i[i] = 0
        coalition_j = coalition_none.copy(); coalition_j[j] = 0
        coalition_ij = coalition_none.copy(); coalition_ij[i] = 0; coalition_ij[j] = 0
        logit_i = value_function([coalition_i])[0]
        logit_j = value_function([coalition_j])[0]
        logit_ij = value_function([coalition_ij])[0]
        print(f"\n{idx+1}. Patches ({i}, {j}) — Interaktionswert: {np.mean(val_array):.6f}")
        print(f"  🔹 Original Logit:    {base_logit:.4f}")
        print(f"  🔸 Nur {i} maskiert:  {logit_i:.4f}")
        print(f"  🔸 Nur {j} maskiert:  {logit_j:.4f}")
        print(f"  🔻 Beide maskiert:    {logit_ij:.4f}")

# === Hauptprogramm ===
def main():
    global processor, model, image, predicted_class
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
    model.eval()

    image = Image.open(BytesIO(requests.get(image_url).content)).convert("RGB").resize((384, 384))
    image.show()

    if SHOW_INDEX_GRID:
        draw_patch_index_grid(image.copy())

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = outputs.logits.argmax(-1).item()

    print(f"Vorhergesagte Klasse: {model.config.id2label[predicted_class]}") #bis hierhin wurde Originalbild vorhergesagt, schwärzung hat noch nicht begonnen

    val_full = value_function([np.ones(n_patches)])[0]
    val_none = value_function([np.zeros(n_patches)])[0]
    print(f"Logit Originalbild: {val_full:.2f}")
    print(f"Logit Komplett maskiert: {val_none:.2f}")

    # Beispiel für das Testen eines einzelnen maskierten Patches (z.B. Patch 130)
    for i in range(0, n_patches, 10):  # Alle 10ten Patches durchtesten
        coalition = np.ones(n_patches)
        coalition[i] = 0  # Maskiere den Patch
        logit_value = value_function([coalition])[0]
        print(f"Logit für Patch {i} maskiert: {logit_value}")


    approximator = RegressionFSII(n=n_patches, max_order=2, random_state=None)
    values = approximator.approximate(game=value_function, budget=100, return_interactions=True)

    second_order = {k: v for k, v in values.interaction_lookup.items() if len(k) == 2}

    ''' #hier <------------------------------------------------------------
    # === Detaillierte Interaktionsanalyse ===

    # Extrahiere alle Interaktionen zwischen den Patches (die Paare)
    second_order = {k: v for k, v in values.interaction_lookup.items() if len(k) == 2}

    # Sortiere die Interaktionen nach dem Durchschnittswert der Interaktion
    sorted_interactions = sorted(second_order.items(), key=lambda kv: abs(np.mean(kv[1])), reverse=True)

    # Drucke die Top 10 Interaktionen
    # Ändere die Print-Ausgabe, um mit Unicode-Problemen umzugehen
    print(f"→ Top 10 Interaktionen:")

    for idx, ((i, j), val_array) in enumerate(sorted_interactions[:10], start=1):
        mean_interaction = np.mean(val_array)
        print(f"{idx}. Patch-Paar ({i}, {j}) → Mittelwert Interaktion: {mean_interaction:.6f}")

    # Zeichne die Interaktionspaare im Bild
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    colors = ["red", "blue", "green", "yellow", "purple"]

    # Visualisiere die Top 5 Interaktionen
    top_interactions = sorted_interactions[:5]

    for idx, ((i, j), val_array) in enumerate(top_interactions):
        coords_i = patch_to_coords(i)
        coords_j = patch_to_coords(j)
        
        # Zeichne Rechtecke um die Interaktions-Patches
        draw.rectangle(coords_i, outline=colors[idx], width=3)
        draw.rectangle(coords_j, outline=colors[idx], width=3)
        draw.text((coords_i[0], coords_i[1]), str(idx + 1), fill=colors[idx])
        draw.text((coords_j[0], coords_j[1]), str(idx + 1), fill=colors[idx])

    # Zeige das Bild mit den Interaktionsmarkierungen
    img_draw.show()

    # === Untersuche die Interaktionen für die unteren rechten Patches ===
    print("\n→ Interaktionen zwischen Patches in der unteren rechten Ecke (z. B. Patches 130–143):")
    for (i, j), val_array in second_order.items():
        if i >= 130 or j >= 130:  # Fokussiere auf Patches im unteren rechten Bereich
            print(f"Interaktion zwischen Patch {i} und Patch {j}: {np.mean(val_array):.6f}")



    #hier <------------------------------------------------------------ '''

    analyze_top_interactions(values, top_k=5)

    top_5_pairs = [pair for pair, _ in sorted(second_order.items(), key=lambda kv: abs(np.mean(kv[1])), reverse=True)[:5]]

    print("\n→ Top 5 Interaktionen und Koordinaten:")
    for idx, (i, j) in enumerate(top_5_pairs):
        print(f"{idx+1}. ({i}, {j}) → {patch_to_coords(i)} | {patch_to_coords(j)}")

    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    colors = ["red", "blue", "green", "yellow", "purple"]
    for idx, (i, j) in enumerate(top_5_pairs):
        coords_i = patch_to_coords(i)
        coords_j = patch_to_coords(j)
        draw.rectangle(coords_i, outline=colors[idx], width=3)
        draw.rectangle(coords_j, outline=colors[idx], width=3)
        draw.text((coords_i[0], coords_i[1]), str(idx+1), fill=colors[idx])
        draw.text((coords_j[0], coords_j[1]), str(idx+1), fill=colors[idx])
    img_draw.show()

    

if __name__ == "__main__":
    main()
