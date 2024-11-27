
# 🖼️ Skin Lesion Classification Project

Welcome to the **Skin Lesion Classification** project! This repository focuses on detecting and classifying various types of skin lesions using the **HAM10000 dataset** and advanced **Machine Learning** techniques.

---

## 🎯 **Project Overview**

Skin cancer is one of the most common cancers globally, and early detection is crucial for effective treatment. This project leverages state-of-the-art machine learning models to classify different types of skin lesions.

---

### 🔍 **Dataset**
- **Name**: [HAM10000 Dataset](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000) 🌐
- **Description**: A large collection of multi-source dermatoscopic images for pigmented lesions.
- **Lesion Types**:
  1. **Melanocytic nevi (nv)**
  2. **Melanoma (mel)**
  3. **Benign keratosis-like lesions (bkl)**
  4. **Basal cell carcinoma (bcc)**
  5. **Actinic keratoses (akiec)**
  6. **Vascular lesions (vasc)**
  7. **Dermatofibroma (df)**

---

### 🚀 **Technologies Used**
- **Libraries**: `NumPy`, `Pandas`, `OpenCV`, `scikit-learn`, `matplotlib`, `seaborn`, `imblearn`
- **Models**:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Random Forest Classifier 🌳

---

### 🛠️ **Steps Involved**

1. **Data Loading and Preprocessing** 📂
   - Loaded images from the dataset folders.
   - Preprocessed images and extracted meaningful features (color, texture).

2. **Feature Extraction** 🌈
   - **Color Statistics**: Mean, Standard Deviation, Histograms.
   - **Texture Features**: GLCM, Local Binary Patterns (LBP), Entropy.

3. **Handling Class Imbalance** ⚖️
   - Applied **Random Oversampling** to balance class distributions.

4. **Model Training and Evaluation** 📊
   - Evaluated three models:
     - Logistic Regression
     - KNN
     - Random Forest (with Hyperparameter Tuning 🎛️)
   - Metrics: Accuracy, ROC AUC Score, Confusion Matrix, Classification Report.

5. **Hyperparameter Tuning** 🔧
   - Performed using **RandomizedSearchCV** to find optimal Random Forest parameters.

6. **Visualization** 🖼️
   - Plotted Confusion Matrices and Multiclass ROC Curves.

---

### 📈 **Results**

| Model                  | Accuracy | ROC AUC Score |
|------------------------|----------|---------------|
| Logistic Regression    | 37.6%    | 0.74          |
| K-Nearest Neighbors    | 83.2%    | 0.97          |
| Random Forest (Tuned)  | **98.31%**| **0.9998**   |

**Note**: Results may vary depending on the train-test split and parameters.
