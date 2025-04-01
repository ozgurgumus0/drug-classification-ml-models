# 💊 Drug Classification with Multiple ML Models

This project is a machine learning pipeline built to classify drug types based on patient features like Age, Sex, Blood Pressure, Cholesterol, and Sodium-to-Potassium ratio. It was developed as part of the **END4991 Design II** course at **Yıldız Technical University**.

---

## 🧠 Algorithms Implemented

The following models were built, trained, and compared using a shared dataset of 200 records:

- ✅ **K-Nearest Neighbors (KNN)**  
- ✅ **Support Vector Machine (SVM)**  
- ✅ **Decision Tree & Random Forest**  
- ✅ **Naive Bayes Classifier**  
- ✅ **Clustering (K-Means, GMM, PCA)**

Each model was evaluated using:
- Accuracy
- Confusion Matrix
- ROC-AUC
- Silhouette Score (Clustering)

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

## 👥 Team Members

- Özgür Gümüş  
- Melis Kamacıoğlu  
- İlyas Uğur Vural  
- Özlem Yüksel Bilir  
- Nihal Diler  

🎓 Advisor: Prof. Dr. Alev Taşkın  
🏫 Yıldız Technical University – 2022–2023 Fall

---

> This project demonstrates practical application of supervised and unsupervised ML algorithms in a healthcare context using Python and scikit-learn.
