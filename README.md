# 🐶🐱 CNN Image Classification - Dogs vs Cats  

This project implements a **Convolutional Neural Network (CNN)** using **TensorFlow & Keras** to classify images of dogs and cats. The model is trained with real-world image augmentation techniques and evaluated for accuracy.  

---

## 🚀 Features
- Built with **TensorFlow & Keras**
- **Image Augmentation** (rotation, zoom, shear, flips) for better generalization  
- **Binary Classification** (Dog 🐶 vs Cat 🐱)  
- Trained with **CNN architecture (Conv2D + MaxPooling layers)**  
- Uses **train/validation split** for proper evaluation  

---

## 📂 Dataset
- Dataset is organized into subfolders under `Dataset/`
  - `Dataset/train/cats`
  - `Dataset/train/dogs`
- Images are automatically split into **80% training** and **20% validation** using `ImageDataGenerator`.

---

## 🛠️ Tech Stack
- **Python 3.x**
- **TensorFlow / Keras**
- **NumPy**
- **Matplotlib**

---

## ⚙️ Model Architecture
```text
Input: (150 x 150 x 3)
↓ Conv2D (32 filters, ReLU) + MaxPooling
↓ Conv2D (64 filters, ReLU) + MaxPooling
↓ Conv2D (128 filters, ReLU) + MaxPooling
↓ Flatten
↓ Dense (512 units, ReLU)
↓ Dense (1 unit, Sigmoid) → Binary output (Dog or Cat)

Training & Evaluation
🔹 Augmented Images (Sample)

🔹 Training Curves

Accuracy and Loss during training:


▶️ How to Run

Clone this repo & install dependencies:

pip install tensorflow matplotlib numpy


Place your dataset inside the Dataset/ folder (with cats/ and dogs/ subfolders).

Run the script:

python cnn_dog_cat.py


Model will train for given epochs and show accuracy/loss graphs.

📈 Results

Achieved ~85% accuracy on validation data.

Data augmentation helped reduce overfitting.

Model can be further improved with Transfer Learning (VGG16, ResNet, MobileNet).

📌 Future Scope

Add Dropout layers to improve generalization

Use Transfer Learning for higher accuracy

Deploy as a Flask / FastAPI Web App

✨ Author

Noor Alam

💻 AI/ML & Deep Learning Enthusiast

🔍 Exploring Computer Vision for real-world applications