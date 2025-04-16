import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from vit_model import VisionTransformer, ViTConfig
from shapiq.explainer import TabularExplainer


# CUDA-Gerät setzen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# CIFAR-10 vorbereiten
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
image, label = next(iter(testloader))

# Bild anzeigen
unnorm = transforms.Compose([
    transforms.Normalize(mean=[0., 0., 0.], std=[1/0.5]*3),
    transforms.Normalize(mean=[-0.5]*3, std=[1.]*3),
    transforms.ToPILImage()
])
plt.imshow(unnorm(image[0]))
plt.title(f"Label: {label.item()}")
plt.axis("off")
plt.show()

# Modell laden
config = ViTConfig()
model = VisionTransformer(config)
model.load_state_dict(torch.load("vision_transformer_model.pth", map_location=device))
model.to(device)
model.eval()

# Bild vorbereiten & Patches extrahieren
image = image.to(torch.float32).to(device)
patches = model.patch_embed(image)  # [1, num_patches, dim]
print("Patches shape:", patches.shape)
x = model.pos_embed(patches)
print("Nach pos_embed:", x.shape)

patches_np = patches.squeeze(0).detach().cpu().numpy()


class PatchModelWrapper(torch.nn.Module):
    def __init__(self, model, num_patches=64):
        super().__init__()
        self.model = model
        self.num_patches = num_patches

    def forward(self, x_patches):  # x_patches: [1, n_active_patches, dim]
        device = next(self.model.parameters()).device
        dim = x_patches.shape[-1]

        # Wenn weniger als 64 Patches: mit 0 auffüllen
        if x_patches.shape[1] < self.num_patches:
            pad_len = self.num_patches - x_patches.shape[1]
            pad = torch.zeros((1, pad_len, dim), device=device)
            x_patches = torch.cat([x_patches, pad], dim=1)

        # Weiter durch das Modell
        x = self.model.pos_embed(x_patches)
        x = self.model.encoder(x)
        x = self.model.norm(x)
        out = self.model.mlp_head(x[:, 0])
        return out
    


# Prediction-Funktion, die mit np-Array klarkommt
wrapped_model = PatchModelWrapper(model)

def predict_fn(x_np):
    print(f"[DEBUG] predict_fn Input shape: {x_np.shape}")
    
    if x_np.ndim == 2:
        x_np = x_np.reshape(-1, 64, 64)  # z. B. (8192, 64) → (128, 64, 64)

    x_tensor = torch.tensor(x_np, dtype=torch.float32).to(device)

    outputs = []
    with torch.no_grad():
        for sample in x_tensor:
            sample = sample.unsqueeze(0)  # [1, n_patches, dim]
            out = wrapped_model(sample)
            outputs.append(out.detach().cpu().numpy()[0])

    return np.array(outputs)







# Explainer initialisieren
explainer = TabularExplainer(model=predict_fn, data=patches_np)



# Erklärung berechnen
interactions = explainer.explain(patches_np, budget=128) # Budget anpassen wenn nötig

# Visualisierung
# Debug: Zeig die tatsächliche Form
print("Shape der Interaktionen:", interactions.values.shape)

# Interaktionsmatrix manuell rekonstruieren
def manual_restore_interaction_matrix(values, n_features):
    matrix = np.zeros((n_features, n_features))
    idx = 0
    for i in range(n_features):
        for j in range(i, n_features):
            matrix[i, j] = values[idx]
            if i != j:
                matrix[j, i] = values[idx]
            idx += 1
    return matrix

# Matrix herstellen & Importance berechnen
matrix = manual_restore_interaction_matrix(interactions.values, n_features=64)
importance = matrix.sum(axis=1).reshape(8, 8)

# Visualisierung
plt.figure(figsize=(6, 5))
plt.imshow(importance, cmap='viridis')
plt.colorbar(label="Importance")
plt.title("Patch-wise FSII Importance Map")
plt.axis("off")
plt.show()
