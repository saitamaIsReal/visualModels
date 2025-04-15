import torch
from torchvision import datasets, transforms
from vit_model import ViTConfig, VisionTransformer  # ← Passe ggf. Dateiname an!
import matplotlib.pyplot as plt


# 1. Lade ein Bild
transform = transforms.ToTensor()
dataset = datasets.CIFAR10(root="./data", train=False, download=False, transform=transform)
image, label = dataset[0]  # ein einzelnes Bild + Label
image = image.unsqueeze(0)  # [1, 3, 32, 32]

# 2. Initialisiere das Modell
config = ViTConfig()
model = VisionTransformer(config)

# 3. Mache Vorhersage (Forward Pass)
output = model(image)  # [1, 10]
print("Modellausgabe:", output)

# 4. Vorhergesagte Klasse anzeigen
predicted_class = torch.argmax(output, dim=1).item()
print(f"Vorhergesagte Klasse: {predicted_class}")
print(f"Tatsächliche Klasse: {label}")

# 5. Bild anzeigen
import torchvision.transforms.functional as TF
plt.imshow(TF.to_pil_image(image.squeeze(0)))  # [1,3,32,32] → [3,32,32]
plt.title(f"Label: {label} / Vorhersage: {predicted_class}")
plt.axis("off")
plt.show()

# Die Klassen des CIFAR-10-Datensatzes ausgeben
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
class_names = trainset.classes  # Dies gibt eine Liste der Klassennamen zurück
for i, class_name in enumerate(class_names):
    print(f"Klasse {i}: {class_name}")