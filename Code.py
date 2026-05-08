import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# =====================================================
# 1. KONFIGURACJA URZĄDZENIA
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Używane urządzenie: {device}")

# =====================================================
# 2. TRANSFORMACJE (Ulepszone o Augmentację Danych)
# =====================================================
transform_base = transforms.Compose([
    # 1. Konwersja na tensor
    transforms.ToTensor(),

    # 2. FIX ORIENTACJI (Naprawa surowego EMNIST do pionu)
    transforms.Lambda(lambda x: x.transpose(1, 2)),
    transforms.Lambda(lambda x: torch.flip(x, [2])),

    # 3. AUGMENTACJA (Model uczy się, że litera może być niedoskonała)
    # Dodajemy losowy obrót o 10 stopni i lekkie przesunięcie/skalowanie
    transforms.RandomRotation(degrees=10, fill=-1.0), # fill=-1.0 bo tło po normalizacji to -1
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),

    # 4. NORMALIZACJA
    transforms.Normalize((0.5,), (0.5,))
])

# =====================================================
# 3. DATASET (Zbiór 'letters' - 26 wielkich liter)
# =====================================================
train_set = torchvision.datasets.EMNIST(root="./data", split="letters", train=True, download=True,
                                        transform=transform_base)
test_set = torchvision.datasets.EMNIST(root="./data", split="letters", train=False, download=True,
                                       transform=transform_base)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)


# =====================================================
# 4. ROZBUDOWANA ARCHITEKTURA CNN (AdvancedLetterCNN)
# =====================================================
class AdvancedLetterCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Blok 1: 28x28 -> 14x14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Blok 2: 14x14 -> 7x7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Blok 3: 7x7 -> 3x3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 26)  # 26 klas (A-Z)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = AdvancedLetterCNN().to(device)

# =====================================================
# 5. TRENING I STATYSTYKI
# =====================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

epochs = 10
best_acc = 0
history_loss = []
history_acc = []

print(f"\nRozpoczynanie treningu na {len(train_set)} obrazach...")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        # EMNIST 'letters' ma etykiety 1-26, przesuwamy na 0-25
        images, labels = images.to(device), (labels - 1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 400 == 0:
            print(f"Epoka {epoch + 1}/{epochs} | Batch {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

    # Walidacja po każdej epoce
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), (labels - 1).to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(train_loader)

    history_loss.append(avg_loss)
    history_acc.append(accuracy)

    print(f"\n>>> KONIEC EPOKI {epoch + 1} | Skuteczność: {accuracy:.2f}% | Średni Loss: {avg_loss:.4f} <<<\n")

    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.state_dict(), "model.pth")
        print("Zapisano najlepszy model!\n")

    scheduler.step(accuracy)

# =====================================================
# 6. WYKRESY PROGRESSU
# =====================================================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), history_acc, marker='o', color='g', label='Dokładność')
plt.title('Postęp Skuteczności (Accuracy)')
plt.xlabel('Epoka')
plt.ylabel('%')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), history_loss, marker='s', color='r', label='Błąd')
plt.title('Spadek Błędu (Loss)')
plt.xlabel('Epoka')
plt.ylabel('Wartość Loss')
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Trening ukończony. Najlepsze Accuracy: {best_acc:.2f}%")