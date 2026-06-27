# American Sign Language (ASL) Alphabet Detection

This repository contains a deep learning project for recognizing American Sign Language (ASL) fingerspelling. It implements a hybrid **CNN + LSTM** model in TensorFlow/Keras to classify images of hand gestures into their corresponding ASL alphabet letters and control signals.

## 📌 Project Overview

American Sign Language (ASL) is a vital tool for communication within the Deaf and hard-of-hearing communities. Automating fingerspelling recognition helps bridge the communication gap. 

This project uses a hybrid neural network architecture:
1. **Convolutional Neural Networks (CNN)** to extract spatial features from hand gesture images.
2. **Long Short-Term Memory (LSTM)** network to analyze sequential patterns in the extracted features.

The model is trained on the popular ASL Alphabet dataset and achieves high accuracy in classifying 29 different classes.

---

## 📊 Dataset Details

The model is trained using the **ASL Alphabet Dataset** from Kaggle (available [here](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)).
* **Total Classes:** 29 classes
  * **Letters:** A to Z (26 classes)
  * **Control Gestures:** `space`, `del` (delete), `nothing`
* **Input Image Size:** Resized and normalized to $64 \times 64$ pixels with 3 color channels (RGB).
* **Data Split:** 80% for training (60,091 images), 20% for validation (15,020 images).

---

## 🧠 Model Architecture

The hybrid **CNN-LSTM** network is structured as follows:

| Layer (Type) | Output Shape | Details / Hyperparameters |
| :--- | :--- | :--- |
| **Conv2D** | (None, 62, 62, 32) | 32 filters, 3x3 kernel, ReLU activation |
| **MaxPooling2D** | (None, 31, 31, 32) | 2x2 pool size |
| **Conv2D** | (None, 29, 29, 64) | 64 filters, 3x3 kernel, ReLU activation |
| **MaxPooling2D** | (None, 14, 14, 64) | 2x2 pool size |
| **Conv2D** | (None, 12, 12, 128) | 128 filters, 3x3 kernel, ReLU activation |
| **MaxPooling2D** | (None, 6, 6, 128) | 2x2 pool size |
| **Flatten** | (None, 4608) | Flattens 2D features to 1D |
| **Reshape** | (None, 36, 128) | Reshapes flattened features into a sequence of 36 timesteps with 128 features each |
| **LSTM** | (None, 128) | 128 LSTM units |
| **Dense** | (None, 256) | 256 units, ReLU activation |
| **Dropout** | (None, 256) | Dropout rate of 0.5 (prevents overfitting) |
| **Dense (Output)** | (None, 29) | 29 units, Softmax activation |

---

## 📈 Training Results

The model is compiled with the **Adam optimizer** and **Categorical Crossentropy** loss. It was trained for 20 epochs:

* **Training Accuracy:** $\approx 99.6\%$
* **Validation Accuracy:** $\approx 86.5\%$

---

## 📁 Repository Structure

* 📓 `train model.ipynb` – Jupyter Notebook containing dataset downloading, preprocessing, model architecture design, training, and accuracy plotting.
* 📓 `Copy of test_model_2 (1).ipynb` – Jupyter Notebook to load the trained model, upload test images, and run predictions with confidence scores.
* 💾 `cnn_lstm_sign_language_model (2).keras` – The saved trained model file.
* 📄 `.gitignore` – Specifies files to ignore (cache, checkpoints, system files).
* 📄 `README.md` – Project documentation.

---

## 🚀 Getting Started

### 📋 Prerequisites

To run this project locally, ensure you have the following libraries installed:
```bash
pip install tensorflow numpy opencv-python matplotlib kagglehub
```

### 🏋️ Training the Model
1. Open the `train model.ipynb` notebook.
2. The notebook downloads the dataset automatically using `kagglehub`. If training on Google Colab, mount your Google Drive to save the model.
3. Run all cells to preprocess the data, train the model, and save it as `cnn_lstm_sign_language_model.keras`.

### 🔍 Running Predictions
1. Open the `Copy of test_model_2 (1).ipynb` notebook.
2. Load the trained model using:
   ```python
   from tensorflow.keras.models import load_model
   model = load_model('path/to/cnn_lstm_sign_language_model.keras')
   ```
3. Upload an image of an ASL hand gesture.
4. The notebook will automatically resize it, make a prediction, and show the confidence score along with the image.
