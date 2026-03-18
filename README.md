# MNIST PyTorch Starter

🚀 A simple deep learning project using PyTorch for handwritten digit classification.

---

## 📌 Project Overview

This project implements a basic neural network to classify handwritten digits (0–9) using the MNIST dataset.

It covers the full pipeline:
- Data loading
- Model building
- Training
- Evaluation

---

## 🧠 Model

- Input: 28×28 grayscale images
- Flatten → Fully Connected Layer → ReLU → Fully Connected Layer
- Output: 10 classes (digits 0–9)

---

## ⚙️ Tech Stack

- Python
- PyTorch
- Torchvision

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python train.py