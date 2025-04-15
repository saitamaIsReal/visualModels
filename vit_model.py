import torch # alles rund um Tensoren und GPU
import torch.nn as nn #Definieren der Layer
import torch.nn.functional as F # für Funktionen wie Aktivierungen
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

print(torch.__version__)  # Zeigt die PyTorch-Version
print(torch.cuda.is_available())  # Prüft, ob CUDA verfügbar ist


class ViTConfig:
    def __init__(self):
        self.image_size = 32
        self.patch_size = 4
        self.in_channels = 3
        self.num_classes = 10
        self.dim = 64
        self.depth = 6
        self.heads = 8
        self.mlp_dim = 128
        self.dropout = 0.1
        self.num_patches = (self.image_size // self.patch_size) ** 2 


class PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.proj = nn.Conv2d(config.in_channels, config.dim, kernel_size=config.patch_size, stride=config.patch_size) #nn.Conv2d wandelt Bild (B,3,32,32) in (B,64,256) um. Dabei wandelt geht er das 32x32 Bild durch und macht alle 4x4 Pixel einen Patch draus. Diesen 4x4 Patch wandelt er dann in 256 Feature Vektor um

    def forward(self, x): #bringt Bild in richtige Vektorform für Transformer
        x = self.proj(x)  # [B, dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, dim] -> macht aus (B,64,8,8) -> (B,64,64) (flatten) und vertauscht dann mit transpose position 1 und 2 sodass es in richtiger Form für Transformer ist.
        return x


#Bevor der Transformer die Patches verarbeitet, muss er wissen, wo im Bild jeder Patch war. Dafür gibt es Positions-Embeddings.
class AddPositionEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + config.num_patches, config.dim))



    def forward(self, x):
        B, N, D = x.shape  # Batch, Num_Patches, Dim
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, 1 + N, D] cls token enthält zsmfassung über alle patches im bild und wird vorne drangehängt
        x = x + self.pos_embedding  # hängt an jeden patch vektor passenden positionsvektor
        return x



#Ermittlung, wie wichtig Patches untereinander sind (Interaktion)
class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.dim)
        self.attn = nn.MultiheadAttention(embed_dim=config.dim, num_heads=config.heads, batch_first=True)
        self.norm2 = nn.LayerNorm(config.dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.dim, config.mlp_dim),
            nn.GELU(),
            nn.Linear(config.mlp_dim, config.dim),
        )

    def forward(self, x):
        # Multi-Head Attention + Residual
        x_attn, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + x_attn

        # Feedforward + Residual
        x_mlp = self.mlp(self.norm2(x))
        x = x + x_mlp

        return x

class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embed = PatchEmbedding(config)
        self.pos_embed = AddPositionEmbedding(config)
        self.encoder = nn.Sequential(*[TransformerEncoderLayer(config) for _ in range(config.depth)])
        self.norm = nn.LayerNorm(config.dim)
        self.mlp_head = nn.Linear(config.dim, config.num_classes)

    def forward(self, x):
        x = self.patch_embed(x)               # [B, num_patches, dim]
        x = self.pos_embed(x)                 # [B, num_patches+1, dim]
        x = self.encoder(x)                   # Transformer-Blöcke
        x = self.norm(x)                      # Normierung des Outputs
        cls_token_final = x[:, 0]             # Nur das [CLS] Token nehmen
        out = self.mlp_head(cls_token_final)  # Klassifikations-MLP
        return out

# 1. **Datenvorbereitung**
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Größe des Bildes für ViT (32x32)
    transforms.ToTensor(),        # Umwandlung in Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalisierung
])

# Lade den CIFAR-10-Datensatz
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# DataLoader für Trainings- und Testset
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# 2. **Modell-Initialisierung**
config = ViTConfig()  # Konfiguration des ViT-Modells
model = VisionTransformer(config)  # Modell erstellen

# Überprüfe, ob CUDA verfügbar ist und setze das Gerät
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training auf: {device}")

# Das Modell auf das richtige Gerät verschieben
model = model.to(device)
# Debug: Gerät des Modells anzeigen
print(next(model.parameters()).device)


# 3. **Verlustfunktion und Optimierer**
criterion = nn.CrossEntropyLoss()  # Verlustfunktion für Klassifikation
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam Optimierer

# 4. **Trainingsschleife**
num_epochs = 10  # Anzahl der Epochen

for epoch in range(num_epochs):
    model.train()  # Setzt das Modell in den Trainingsmodus
    running_loss = 0.0

    for i, (inputs, labels) in enumerate(trainloader):
        # Verschiebe Eingabedaten und Labels auf das gleiche Gerät und stelle den richtigen Datentyp sicher
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Setzt den Gradienten auf null

        # Erzwungene Typ- und Gerätekompatibilität in der forward-Methode
        outputs = model(inputs.to(device).float())  # Typ explizit setzen für sicherstellen von Float-Tensoren
        loss = criterion(outputs, labels)  # Verlustberechnung
        loss.backward()  # Backpropagation
        optimizer.step()  # Optimierer-Schritt

        running_loss += loss.item()

        # Alle 100 Schritte die Verlustausgabe anzeigen
        if i % 100 == 99:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}], Loss: {running_loss / 100:.4f}")
            running_loss = 0.0

print("Training abgeschlossen")




# 5. **Evaluierung auf dem Testset**
model.eval()  # Setzt das Modell in den Evaluierungsmodus
correct = 0
total = 0

with torch.no_grad():  # Keine Gradientenberechnung für den Test
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)  # Vorhersage mit höchster Wahrscheinlichkeit
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

# Modell speichern (nach dem Training)
torch.save(model.state_dict(), 'vision_transformer_model.pth')
print("Modell wurde gespeichert als 'vision_transformer_model.pth'")
