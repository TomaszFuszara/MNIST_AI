import sys
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QVBoxLayout, QFileDialog, QHBoxLayout
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt


# =====================================================
# 1. ROZBUDOWANA ARCHITEKTURA (Musi być 1:1 jak w treningu)
# =====================================================
class AdvancedLetterCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Blok 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Blok 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            # Blok 3
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


class LetterRecognizerApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Rozpoznawanie liter AI - Advanced")
        self.setGeometry(100, 100, 400, 450)

        layout = QVBoxLayout()

        self.image_label = QLabel("Brak obrazu")
        self.image_label.setAlignment(Qt.AlignCenter)

        self.result_label = QLabel("Wynik: -")
        self.result_label.setAlignment(Qt.AlignCenter)

        self.top3_label = QLabel("")
        self.top3_label.setAlignment(Qt.AlignCenter)

        self.button = QPushButton("Wybierz obraz")
        self.button.clicked.connect(self.load_image)

        layout.addWidget(self.image_label)
        layout.addWidget(self.button)
        layout.addWidget(self.result_label)
        layout.addWidget(self.top3_label)

        self.setLayout(layout)

        # 🔹 ŁADOWANIE MODELU
        self.device = torch.device("cpu")

        # Używamy nowej, rozbudowanej klasy
        self.model = AdvancedLetterCNN().to(self.device)
        try:
            self.model.load_state_dict(torch.load("model.pth", map_location=self.device))
            self.model.eval()
            print("Model Advanced załadowany pomyślnie.")
        except Exception as e:
            print(f"Błąd ładowania: {e}")
            self.result_label.setText("BŁĄD: model.pth nie pasuje do architektury!")

        # EMNIST split='letters' to 26 klas (indeksy 0-25)
        self.classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Wybierz obraz",
            "",
            "Images (*.png *.jpg *.jpeg)"
        )

        if file_path:
            self.display_image(file_path)
            processed = self.preprocess(file_path)
            probs = self.predict(processed)
            self.show_results(probs)

    def display_image(self, path):
        pixmap = QPixmap(path)
        pixmap = pixmap.scaled(200, 200, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

    def preprocess(self, path):
        # 1. Wczytanie i konwersja na odcień szarości
        img = Image.open(path).convert('L')

        # 2. Skalowanie do 28x28 (LANCZOS pomaga zachować detale nóżki P)
        img = img.resize((28, 28), Image.Resampling.LANCZOS)

        arr = np.array(img).astype(np.float32) / 255.0

        # 3. Inwersja kolorów (model musi mieć białą literę na czarnym tle)
        if arr.mean() > 0.5:
            arr = 1.0 - arr

        # --- USUŃ LUB ZAKOMENTUJ TE LINIE PONIŻEJ: ---
        # arr = np.transpose(arr, (1, 0))
        # arr = np.fliplr(arr)
        # ---------------------------------------------

        # 4. Normalizacja (identyczna jak w treningu)
        arr = (arr - 0.5) / 0.5

        # Podgląd (zostaw go na chwilę, żeby sprawdzić czy teraz jest prosto)
        import matplotlib.pyplot as plt
        plt.imshow(arr, cmap='gray')
        plt.title("Litera po transpozycji")
        plt.show()

        # 5. Formatowanie pod PyTorch
        arr = arr.reshape(1, 1, 28, 28)
        return torch.tensor(arr.copy())

    def predict(self, arr):
        with torch.no_grad():
            arr = arr.to(self.device)
            outputs = self.model(arr)
            probs = torch.softmax(outputs, dim=1)
            return probs.cpu().numpy()[0]

    def show_results(self, probs):
        top_indices = np.argsort(probs)[::-1]

        # TOP 1 - Twoja oryginalna funkcja
        best_idx = top_indices[0]
        best_letter = self.classes[best_idx]
        best_conf = probs[best_idx] * 100

        self.result_label.setText(
            f"Rozpoznana litera: {best_letter} ({best_conf:.2f}%)"
        )

        # TOP 3 - Twoja oryginalna funkcja
        top3_text = "Top 3:\n"
        for i in range(3):
            idx = top_indices[i]
            letter = self.classes[idx]
            conf = probs[idx] * 100
            top3_text += f"{letter} – {conf:.2f}%\n"

        self.top3_label.setText(top3_text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LetterRecognizerApp()
    window.show()
    sys.exit(app.exec_())