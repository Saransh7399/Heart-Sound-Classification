# Heart Sound Classification using Audio Features and Machine Learning

This project focuses on classifying heartbeat sounds into one of four categories — `artifact`, `normal`, `murmur`, and `extrahls` — using classical ML models and deep learning with audio feature extraction.

---

## 📦 Dataset

- **Source:** [Kaggle - Heartbeat Sounds Dataset](https://www.kaggle.com/datasets/kinguistics/heartbeat-sounds/data)
- **Subset Used:** Only the `set_a` subset is used in this project.
- **Total Samples:** 176 WAV files across 4 categories:
  - `artifact`: 40
  - `murmur`: 34
  - `normal`: 31
  - `extrahls`: 19

---

## 📊 Project Pipeline

### 🔍 1. Data Understanding & Visualization
- Waveform and spectrogram visualization for individual audio files.
- PCA analysis to evaluate class separability using extracted MFCC features.

### 🎧 2. Feature Engineering
Extracted from audio:
- **MFCCs (13)**
- **Delta + Delta-Delta**
- **Spectral Features**: Centroid, Bandwidth, Roll-off
- **Chroma (12-bin)**
- **ZCR and RMS**

### 🔁 3. Data Augmentation
- Applied **pitch shifting** and **time stretching** to increase training diversity.

### 🧪 4. Classical ML Models
Tested with:
- **Logistic Regression**
- **Random Forest**
- **Gradient Boosting**
- **SVM (with class weights)**

### 🧠 5. Deep Learning Model
- A **1D CNN** trained on MFCC + delta features with augmentations.
- Achieved best macro F1 ~0.80 on validation set.

### ⚖️ 6. Class Imbalance Handling
- Applied **SMOTE** and **class weighting**.
- Improved recall for underrepresented classes (e.g., `extrahls`, `normal`).

---

## 🧪 Evaluation Metrics

- **Confusion Matrix**
- **Precision, Recall, F1-Score**
- **5-Fold Cross Validation**

---

## 🔬 Summary of Results

| Model                     | Accuracy | Best Class   | Weakest Class | Notes                               |
|--------------------------|----------|--------------|----------------|-------------------------------------|
| Logistic Regression       | 56%      | Artifact      | Normal         | Simple baseline using MFCCs only   |
| Random Forest             | 68%      | Artifact      | Normal         | Strong on distinguishable classes  |
| Gradient Boosting         | 60%      | Murmur        | Extrahls       | Better on overlapping classes      |
| RF (SMOTE)                | 68%      | Murmur        | Normal         | Improved recall for `murmur`       |
| GB (SMOTE)                | 68%      | Balanced      | Normal         | Better overall balance             |
| Logistic Regression (CW)  | 64%      | Artifact      | Normal         | Lightweight, interpretable         |
| SVM (CW)                  | 68%      | Extrahls      | Normal         | Excellent on `extrahls` recall     |
| 1D CNN                    | ~77-80%  | Artifact, Murmur | Normal     | Best performance using all features|

---

## 📌 Note
This model has been trained and validated only on the set_a subset. Real-world deployment would require cross-set testing and clinical validation.
Download the data from kaggle and store the python file in the same folders within the stored data.

## 🧰 Requirements

```bash
pip install numpy pandas matplotlib seaborn librosa scikit-learn tensorflow soundfile
