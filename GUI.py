import sys
import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QVBoxLayout, QFileDialog
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt


# 🔹 MODEL (musi być identyczny jak w treningu)
class LetterClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 27)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class LetterRecognizerApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Rozpoznawanie liter AI")
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

        self.model = LetterClassifier().to(self.device)
        self.model.load_state_dict(torch.load("model.pth", map_location=self.device))
        self.model.eval()

        # EMNIST (indeks 1-26)
        self.classes = ["?"] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

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
        img = Image.open(path).convert('L')
        img = img.resize((28, 28))

        arr = np.array(img).astype(np.float32) / 255.0

        # jeśli model się myli, odkomentuj:
        arr = 1 - arr

        # normalizacja EMNIST
        arr = (arr - 0.5) / 0.5

        # transpose jak w treningu
        arr = np.transpose(arr, (1, 0))

        # PyTorch format
        arr = arr.reshape(1, 1, 28, 28)

        return torch.tensor(arr)

    def predict(self, arr):
        with torch.no_grad():
            arr = arr.to(self.device)
            outputs = self.model(arr)

            probs = torch.softmax(outputs, dim=1)

            return probs.cpu().numpy()[0]

    def show_results(self, probs):
        top_indices = np.argsort(probs)[::-1]

        # TOP 1
        best_idx = top_indices[0]
        best_letter = self.classes[best_idx]
        best_conf = probs[best_idx] * 100

        self.result_label.setText(
            f"Rozpoznana litera: {best_letter} ({best_conf:.2f}%)"
        )

        # TOP 3
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