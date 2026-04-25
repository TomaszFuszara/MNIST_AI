import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. Konfiguracja urządzenia (GPU jeśli dostępne, inaczej CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Używane urządzenie: {device}")

# 2. Transformacje danych
# EMNIST wymaga obrotu o 90 stopni i odbicia (transpozycji), aby litery były czytelne
transform = transforms.Compose([
    transforms.ToTensor(),
    lambda x: x.transpose(1, 2),
    transforms.Normalize((0.5,), (0.5,))
])

# 3. Ładowanie zbioru danych EMNIST (litery)
train_dataset = torchvision.datasets.EMNIST(
    root='./data', split='letters', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.EMNIST(
    root='./data', split='letters', train=False, download=True, transform=transform
)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


# 4. Definicja architektury sieci CNN
class LetterClassifier(nn.Module):
    def __init__(self):
        super(LetterClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 27)  # 26 liter (indeksy 1-26 w EMNIST)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = LetterClassifier().to(device)

# 5. Funkcja kosztu i optymalizator
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Pętla trenowania (3 epoki dla demonstracji)
epochs = 3
print("Rozpoczynanie trenowania...")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoka [{epoch + 1}/{epochs}], Strata: {running_loss / len(train_loader):.4f}")

# 7. Ewaluacja modelu
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Dokładność sieci na zbiorze testowym: {100 * correct / total:.2f}%")

# 8. Wizualizacja przykładowego wyniku
images, labels = next(iter(test_loader))
outputs = model(images.to(device))
_, predicted = torch.max(outputs, 1)

plt.imshow(images[0].squeeze(), cmap='gray')
plt.title(f"Prawda: {chr(labels[0] + 64)} | Predykcja: {chr(predicted[0] + 64)}")
plt.show()