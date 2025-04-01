# 💊 Drug Classification with Multiple ML Models

This project is a machine learning pipeline built to classify drug types based on patient features like Age, Sex, Blood Pressure, Cholesterol, and Sodium-to-Potassium ratio. It was developed as part of the **END4991 Design II** course at **Yıldız Technical University**.

---

## 📊 Dataset Overview

- Contains 200 records of patient data
- Features:
  - Age
  - Sex
  - Blood Pressure (BP)
  - Cholesterol
  - Na-to-K Ratio
- Target variable: **Drug Type**

---

## 🧠 Algorithms Implemented

1. **K-Nearest Neighbors (KNN)**
   - Manhattan distance performed best
   - Accuracy: ~85%
2. **Support Vector Machines (SVM)**
   - Kernel types tested: linear, polynomial, RBF, sigmoid
   - Best AUC-ROC: ~0.95
3. **Decision Tree Classifier**
4. **Random Forest Classifier**
   - Outperformed Decision Trees in accuracy and generalization
5. **Naive Bayes Classifier**
   - Evaluated using accuracy, confusion matrix, ROC-AUC
6. **Clustering Techniques**
   - KMeans with Silhouette Score & Elbow Method
   - Gaussian Mixture Models (GMM)

---

## 📈 Evaluation Metrics

- Confusion Matrix
- ROC Curve and AUC Score
- Accuracy, Precision, Recall, F1-score
- Silhouette Score (for clustering)
- Elbow Method and PCA (dimensionality reduction)

---

## 📊 Dataset

- **File:** `drug200.csv`
- **Features:**
  - Age
  - Sex
  - Blood Pressure (BP)
  - Cholesterol
  - Na_to_K ratio
  - Target: Drug Type

---

## 📈 Notebooks & Scripts

| File | Model |
|------|-------|
| `KNN.ipynb` | K-Nearest Neighbors |
| `SVM.ipynb` | Support Vector Machine |
| `decision_tree_and_random_forest.ipynb` | Decision Tree + Random Forest |
| `naive_bayes_drug_classification.ipynb` | Naive Bayes |
| `Clustering.py` | Unsupervised Clustering (Silhouette, Elbow, GMM, PCA) |

---

## 📁 Folder Breakdown

- `data/`: Dataset used for training and evaluation
- `notebooks/`: Jupyter notebooks and Python scripts
- `report/`: Full final report (PDF)
- `presentation/`: Project presentation slides

---

## 🧠 Summary of Findings

- **KNN** achieved highest test accuracy with Manhattan distance  
- **SVM (RBF kernel)** showed strong ROC-AUC (~0.95)  
- **Random Forest** outperformed basic Decision Tree  
- **Naive Bayes** gave decent baseline results  
- **Clustering** helped uncover hidden patterns in patient features

---

- Ozgur Gumus

🎓 Advisor: Prof. Dr. Alev Taşkın  
🏫 Yıldız Technical University – 2022–2023 Fall

---

> This project demonstrates practical application of supervised and unsupervised ML algorithms in a healthcare context using Python and scikit-learn.
