# DeepFake Audio Detection via Spectrogram Analysis

A machine learning pipeline that distinguishes between human speech and AI-generated audio. This project leverages computer vision techniques by converting audio waveforms into Mel Spectrograms and analyzing them with a Convolutional Neural Network (CNN).

<p align="center">
  <img src="spectrograms/real/sample_spectrogram.png" alt="Mel Spectrogram Sample" width="45%" />
  <img src="plots/accuracy_plot.png" alt="Training Accuracy Plot" width="45%" />
</p>

---

## 🚀 Key Features

* **Audio-to-Image Conversion:** Transforms `.wav` audio files into visual Mel Spectrograms (128 mels) to leverage CNN architectures. [`generate_spectrograms.py`]
* **Custom CNN Architecture:** Implements a 3-layer Convolutional Neural Network using TensorFlow/Keras to extract features from frequency and time domains. [`train_cnn.py`]
* **Automated Pipeline:** Includes scripts for batch processing, stratified dataset splitting (80/10/10), and model training/evaluation. [`split_data.py`, `train_cnn.py`]

## 🛠 Tech Stack

* **Language:** Python 3.x
* **Deep Learning:** TensorFlow, Keras
* **Audio Processing:** Librosa
* **Data Manipulation:** NumPy, Pandas
* **Visualization:** Matplotlib

## 📂 Project Structure

    ├── data_clean/            # Raw .wav audio files (Input)
    ├── spectrograms/          # Processed Mel Spectrogram images
    ├── models/                # Saved .keras models
    ├── src/
    │   └── train_cnn.py       # CNN model training & evaluation
    ├── scripts/
    │   ├── generate_spectrograms.py  # Audio processing pipeline
    │   └── split_data.py             # Stratified train/val/test splitter
    └── plots/                 # Training accuracy and loss visualizations

## 🧠 Model Architecture

The model processes **224x224 RGB spectrogram images** through the following architecture (defined in `train_cnn.py`):

1.  **Input Layer:** Rescaling (`1./255`) for normalization.
2.  **Feature Extraction:** Three Convolutional blocks (32, 64, 128 filters) with ReLU activation and MaxPooling (2x2).
3.  **Classification:** Flatten layer followed by a Dense layer (128 units, ReLU) and a final Sigmoid output for binary classification (Real vs. AI).

## 📊 Workflow

The pipeline consists of three distinct stages:

### 1. Preprocessing
The `generate_spectrograms.py` script loads audio at **22,050 Hz**, computes the Mel spectrogram, converts power to dB, and saves the result as a PNG image using a Magma colormap.

### 2. Data Splitting
The `split_data.py` script shuffles the dataset and performs an **80/10/10 split** (Train/Validation/Test) to ensure robust evaluation.

### 3. Training
The `train_cnn.py` script compiles the model using the **Adam optimizer** and **Binary Crossentropy loss**. It employs **Early Stopping** (patience=3) to prevent overfitting.

## 💻 Getting Started

### 1. Prerequisites

Ensure you have the necessary dependencies installed:

    pip install -r requirements.txt

### 2. Data Generation

Convert your audio dataset into spectrograms:

    python scripts/generate_spectrograms.py

This will populate the `spectrograms/` directory. Next, split the data into training sets:

    python scripts/split_data.py

### 3. Train the Model

Run the training script to generate the model and accuracy plots:

    cd src
    python train_cnn.py

## 📈 Results

* **Validation Accuracy:** ~95% *(Update with your real number)*
* **Test Accuracy:** ~94% *(Update with your real number)*

## 🔮 Future Improvements

* [ ] Implement Recurrent Neural Networks (RNN/LSTM) to analyze raw audio waveforms directly.
* [ ] Add data augmentation (time-stretching, pitch-shifting) to improve model robustness against background noise.
