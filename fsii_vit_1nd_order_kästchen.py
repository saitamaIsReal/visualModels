import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from shapiq import RegressionFSII
from PIL import ImageDraw, ImageFont



# === Globale Parameter ===
model_name = "google/vit-base-patch32-384"
image_path = "C:/Users/enton/Desktop/Bachelor/vit_model/images/KOA_Nassau_2697x1517.jpg"
target_class = 208  # Beispiel: Labrador
patch_size = 32
image_size = 384
n_patches_per_row = image_size // patch_size
n_patches = n_patches_per_row ** 2


# === Hilfsfunktionen ===
def mask_image(original_image: Image.Image, coalition: Union[np.ndarray, list], patch_size=32):
    img = original_image.resize((384, 384))
    img_np = np.array(img).copy()

    coalition = np.array(coalition)
    assert len(coalition) == n_patches, "Koalition hat falsche Länge"

    for i, keep in enumerate(coalition):
        if keep == 1:
            continue
        row = i // n_patches_per_row
        col = i % n_patches_per_row
        y1, y2 = row * patch_size, (row + 1) * patch_size
        x1, x2 = col * patch_size, (col + 1) * patch_size
        img_np[y1:y2, x1:x2] = 0

    return Image.fromarray(img_np)


def value_function(coalitions: np.ndarray) -> np.ndarray:
    values = []
    for coalition in coalitions:
        masked_img = mask_image(image, coalition)
        inputs = processor(images=masked_img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logit = outputs.logits[0, target_class].item()
        values.append(logit)
    return np.array(values)


# === Hauptprogramm ===
def main():
    global processor, model, image

    # Modell und Processor laden
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
    model.eval()

    # Bild laden
    image = Image.open(image_path).convert("RGB").resize((384, 384))

    # Vorhersage für Originalbild
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()

    print(f"Vorhergesagte Klasse: {model.config.id2label[predicted_class_idx]}")

    # Logit-Vergleich für Vollbild vs. maskiert
    full_logit = value_function([np.ones(n_patches)])[0]
    masked_logit = value_function([np.zeros(n_patches)])[0]
    print(f"Original Logit: {full_logit:.2f}, Maskiertes Bild Logit: {masked_logit:.2f}")

    # FSII Approximation
    approximator = RegressionFSII(n=n_patches, max_order=1)
    values = approximator.approximate(game=value_function, budget=100)
    print("FSII-Werte berechnet:", values.values.shape)

   # Normiere für Vergleichbarkeit
    importance_map = values.get_n_order_values(order=1).reshape(n_patches_per_row, n_patches_per_row)
    norm_map = (importance_map - importance_map.min()) / (importance_map.max() - importance_map.min())

    # Bild vorbereiten
    image_resized = image.resize((384, 384))

    # Transparente Heatmap über Bild
    plt.figure(figsize=(6, 6))
    plt.imshow(image_resized)
    plt.imshow(norm_map, cmap='jet', alpha=0.7, interpolation='nearest', extent=[0, 384, 384, 0])
    plt.title("Overlay: Wichtigkeit auf Bild (1. Ordnung)")
    plt.colorbar(label="Normierte Wichtigkeit")
    plt.axis("off")
    plt.show()

    top_k = 5
    importance_values = values.get_n_order_values(order=1)
    top_indices = np.argsort(-np.abs(importance_values))[:top_k]

    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    font = ImageFont.load_default()
    colors = ["red", "blue", "green", "yellow", "purple"]
    for idx, patch_idx in enumerate(top_indices):
        x1 = (patch_idx % n_patches_per_row) * patch_size
        y1 = (patch_idx // n_patches_per_row) * patch_size
        x2 = x1 + patch_size
        y2 = y1 + patch_size
        draw.rectangle([x1, y1, x2, y2], outline=colors[idx], width=3)
        draw.text((x1 + 2, y1 + 2), str(idx + 1), fill=colors[idx], font=font) 
        
    draw_img.show(title="Top 1st-Order Patches")




if __name__ == "__main__":
    main()
