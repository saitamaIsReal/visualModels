#!/usr/bin/env python3
import sys
import os
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
from PIL import ImageFont

# Ordnerstruktur anlegen
output_dirs = [
    "output/heatmaps",
    "output/histograms",
    "output/upsets_2er",
    "output/upsets_1er2er",
    "output/top_interactions",
]

for d in output_dirs:
    os.makedirs(d, exist_ok=True)

image_paths = [
    "images/cat1.jpg",
    "images/cat2.jpg",
    "images/cat3.jpg",
    "images/cat4.jpg",
    "images/lucky.jpeg",
    "images/dog2.jpg",
    "images/dog3.jpg",
    "images/dog4.jpg",

]

# Model selection
models = [
    "google/vit-base-patch32-384",
    "facebook/deit-tiny-patch16-224",
    "akahana/vit-base-cats-vs-dogs",
]

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
            font = ImageFont.truetype("arial.ttf", size=7)  # kleiner und lesbar
            draw.text((x1+2, y1+2), str(idx), fill="gray", font=font)

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

heatmaps_all = []
histograms_all = []
upset2_all = []
upset12_all = []
top_patches_all = []
upset_plots_2er_only = []
upset_plots_combined = []
histograms = []
# Farbcodes für Modelle (eine Farbe pro Modell)
model_colors = [(255, 0, 0), (0, 180, 0), (0, 128, 255)]

for img_idx, image_path in enumerate(image_paths):
    heatmaps_per_image = []
    histograms_per_image = []
    upsets2_per_image = []
    upsets12_per_image = []
    top_patches_per_image = []
    print(f"\n===== Bild {img_idx+1}/{len(image_paths)}: {image_path} =====")
    

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

        image = Image.open(image_path).convert("RGB").resize((image_size, image_size))

    # 3) load & show grid
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
            random_state=42,
            n_jobs=-1
        )
        result = approximator.approximate(
            budget = 64000,
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
        # Grid über Heatmap legen
        heatmap_with_grid = heatmap_img.copy()
        draw_grid(heatmap_with_grid, n_patches_per_row, cell)


        
        # === UpSet Plot (2nd-Order Interactions) ===
        second_order = result.get_n_order(order=2)

        # Top 10 nach Betrag
        top10_items = sorted(second_order.dict_values.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        # Top 3 Paare für spätere gemeinsame Visualisierung sichern
        if 'top_interaction_patches' not in locals():
            top_interaction_patches = []

        top3_pairs = top10_items[:3]  # (key, value)
        patch_ids = [i for pair, _ in top3_pairs for i in pair]
        top_interaction_patches.append(patch_ids)  # Liste von Listen
        

        top10_pairs = [item[0] for item in top10_items]
        top10_values = [item[1] for item in top10_items]
        for (i, j), val in zip(top10_pairs, top10_values):
            print(f"yes shapiq: Interaction ({i}, {j}): {val:.4f}")
        # Echte Feature-IDs extrahieren
        used_ids = sorted(set(i for pair in top10_pairs for i in pair))

        # Remapping in fester Reihenfolge
        id2new = {real: idx for idx, real in enumerate(used_ids)}
        remapped_pairs = [tuple(sorted((id2new[i], id2new[j]))) for (i, j) in top10_pairs]

        # Neues InteractionValues mit garantierter Paar-Score-Zuordnung
        subset = InteractionValues(
            values=np.array(top10_values),
            index="sparse",
            n_players=len(used_ids),
            max_order=2,
            min_order=2,
            interaction_lookup={pair: idx for idx, pair in enumerate(remapped_pairs)},
            baseline_value=0.0
        )


        # Mapping anzeigen und vorbereiten
        # Mapping anzeigen und vorbereiten
        print("Remapping (neue ID → echte Feature-ID):")
        mapping_text = "\n".join(
            f"{new_id} → Feature {real_id}"
            for new_id, real_id in sorted((v, k) for k, v in id2new.items())
        )
        print(mapping_text)

        # Plot erstellen
        fig = upset_plot(subset, show=False)
        # Modelltitel über dem jeweiligen Plot
        fig.suptitle(model_name, fontsize=12, weight='bold')

        # ⬇️ Großer Abstand unten und größere Schrift
        fig.subplots_adjust(bottom=0.25)  # Platz für Legende
        fig.text(0.01, 0.01, mapping_text, fontsize=16, va="bottom", ha="left", family="monospace")


        print("Top10 original:", top10_items)


        # Bild als PIL.Image umwandeln
        canvas = FigureCanvas(fig)
        fig.set_canvas(canvas)  # sehr wichtig!
        
        canvas.draw()

        width, height = canvas.get_width_height()
        image_bytes = canvas.buffer_rgba()  # nicht tostring_rgb!
        upset_img = Image.frombuffer("RGBA", (width, height), image_bytes, "raw", "RGBA", 0, 1)
        # Zur Liste hinzufügen
        upset_plots_2er_only.append(upset_img)
        plt.close(fig)

    #========= UpSet Plot: 1er + 2er kombiniert ===========

        # 1) Alle Werte holen
        all_items = result.dict_values.items()

        # 2) Nur (i,) und (i,j) behalten
        filtered_items = [(k, v) for k, v in all_items if len(k) in [1, 2]]

        # 3) Top 10 nach Betrag
        top10_items_combined = sorted(filtered_items, key=lambda x: abs(x[1]), reverse=True)[:10]

        # 4) Feature-IDs extrahieren
        used_ids_combined = sorted(set(i for key, _ in top10_items_combined for i in key))
        id2new_combined = {real: idx for idx, real in enumerate(used_ids_combined)}

        # 5) Remapping anwenden
        remapped_keys_combined = [
            tuple(sorted(id2new_combined[i] for i in key))
            for key, _ in top10_items_combined
        ]
        top10_values_combined = [v for _, v in top10_items_combined]

        # 6) Neues InteractionValues-Objekt bauen
        subset_combined = InteractionValues(
            values=np.array(top10_values_combined),
            index="sparse",
            n_players=len(used_ids_combined),
            min_order=1,
            max_order=2,
            interaction_lookup={k: i for i, k in enumerate(remapped_keys_combined)},
            baseline_value=0.0,
        )

        # 7) Mapping anzeigen & als Text vorbereiten
        print("Remapping (1er + 2er):")
        mapping_text_combined = "\n".join(
            f"{new_id} → Feature {real_id}"
            for new_id, real_id in sorted((v, k) for k, v in id2new_combined.items())
        )
        print(mapping_text_combined)

        # 8) Plot erzeugen
        fig_combined = upset_plot(subset_combined, show=False)
        fig_combined.suptitle(f"{model_name} (1er + 2er)", fontsize=12, weight='bold')
        fig_combined.subplots_adjust(bottom=0.25)
        fig_combined.text(0.01, 0.01, mapping_text_combined, fontsize=16, va="bottom", ha="left", family="monospace")

        # 9) In PIL umwandeln
        canvas_combined = FigureCanvas(fig_combined)
        fig_combined.set_canvas(canvas_combined)
        canvas_combined.draw()
        width_c, height_c = canvas_combined.get_width_height()
        img_combined = Image.frombuffer("RGBA", (width_c, height_c), canvas_combined.buffer_rgba(), "raw", "RGBA", 0, 1)
        upset_plots_combined.append(img_combined)
        plt.close(fig_combined)

        #========= HISTOGRAMM (1er + 2er) ==========
        import seaborn as sns

        # Alle Shapley- und Interaktionswerte bis Order 2
        all_items = result.dict_values.items()
        values_1_2 = [abs(v) for k, v in all_items if len(k) in [1, 2]]

        # Histogramm mit seaborn
        fig_hist = plt.figure(figsize=(6, 4))
        sns.histplot(values_1_2, bins=30, kde=False, color='royalblue')
        plt.title(f"Histogramm – Shapley + Interaktionen\n({model_name})", fontsize=10)
        plt.xlabel("Wert (absolut)")
        plt.ylabel("Anzahl")
        plt.tight_layout()

        # Speichern oder als PIL
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        canvas_hist = FigureCanvas(fig_hist)
        canvas_hist.draw()
        w, h = canvas_hist.get_width_height()
        hist_img = Image.frombuffer("RGBA", (w, h), canvas_hist.buffer_rgba(), "raw", "RGBA", 0, 1)
        plt.close(fig_hist)

        # Zur Liste hinzufügen
        heatmaps_per_image.append(heatmap_with_grid)
        heatmap_with_grid.save(f"output/heatmaps/{img_idx}_{model_name.replace('/', '_')}.png")
        histograms_per_image.append(hist_img)
        hist_img.save(f"output/histograms/{img_idx}_{model_name.replace('/', '_')}.png")
        upsets2_per_image.append(upset_img)
        upset_img.save(f"output/upsets_2er/{img_idx}_{model_name.replace('/', '_')}.png")
        upsets12_per_image.append(img_combined)
        img_combined.save(f"output/upsets_1er2er/{img_idx}_{model_name.replace('/', '_')}.png")
        top_patches_per_image.append(patch_ids)

    heatmaps_all.append(heatmaps_per_image)
    histograms_all.append(histograms_per_image)
    upset2_all.append(upsets2_per_image)
    upset12_all.append(upsets12_per_image)
    top_patches_all.append(top_patches_per_image)

    #========= TOP-INTERACTIONS MIT KLARER LEGENDENSPALTE ==========

    base_img = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(base_img)
    draw_grid(base_img, n_patches_per_row, cell)

    for model_idx, (name, patches) in enumerate(zip(models, top_interaction_patches[-len(models):])):
        color = model_colors[model_idx]
        for patch in patches:
            r, c = divmod(patch, n_patches_per_row)
            x1, y1 = c * cell, r * cell
            x2, y2 = x1 + cell, y1 + cell
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)


    # Speichern
    base_img.save(f"output/top_interactions/{img_idx}_top_interactions.png")
    base_img.show()

    print("\n=== Farbzuordnung + Top-3 Interaktionen pro Modell ===")
    for name, color, patches in zip(models, model_colors, top_interaction_patches):
        idx = models.index(name)
        top3_pairs = list(zip(patches[::2], patches[1::2]))
        top3_text = "; ".join(f"{i},{j}" for i, j in top3_pairs)
        print(f"{idx+1}. {name} – RGB{color} – Top-Paare: {top3_text}")

#=========HEAT PRINT==========
    # 🔁 Standardgröße definieren (z. B. 384x384, weil ViT damit arbeitet)
    standard_size = (384, 384)
    # 📐 Alle Heatmaps auf gleiche Größe bringen
    resized_heatmaps = [img.resize(standard_size) for img in heatmaps_per_image]
    # 🔗 Jetzt sauber nebeneinander kombinieren
    final_img = combine_side_by_side(resized_heatmaps)
    final_img.save("vergleich_heatmaps.png")
    final_img.show()

    #=========UPSET PLOT (2er only) BILDER KOMBINIEREN==========
    upset_standard_size = upset_plots_2er_only[0].size
    resized_2er = [img.resize(upset_standard_size) for img in upset_plots_2er_only]
    combined_2er = combine_side_by_side(resized_2er)
    combined_2er.save("vergleich_upsetplots_2er.png")
    combined_2er.show()

    #=========UPSET PLOT (1er + 2er) BILDER KOMBINIEREN==========
    resized_combined = [img.resize(upset_standard_size) for img in upset_plots_combined]
    combined_combined = combine_side_by_side(resized_combined)
    combined_combined.save("vergleich_upsetplots_1er_2er.png")
    combined_combined.show()

    #=========HISTOGRAMM-BILDER KOMBINIEREN==========

    # Einheitliche Größe (z. B. wie erstes Histogramm)
    histogram_standard_size = histograms_all[0][0].size
    resized_histograms = [img.resize(histogram_standard_size) for img in histograms_all[0]]
    combined_histogram_img = combine_side_by_side(resized_histograms)
    combined_histogram_img.save("vergleich_histogramme.png")
    combined_histogram_img.show()




