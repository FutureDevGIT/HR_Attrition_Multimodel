# 👩‍💼 HR Attrition Predictor (Multi-Model + SMOTE + Streamlit)

This is an end-to-end ML project that predicts whether an employee is likely to **leave the company** based on HR data.  
It uses multiple machine learning models, handles **imbalanced data** using SMOTE, and offers an interactive **Streamlit UI** for real-time prediction.

---

## 📊 Dataset

- **Source:** IBM HR Analytics Employee Attrition Dataset  
- **Download:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
- Target variable: `Attrition` (Yes / No)
- Problem Type: **Binary Classification** (Highly Imbalanced)

---

## ⚙️ Tech Stack

| Component          | Tools Used                                 |
|--------------------|---------------------------------------------|
| Language           | Python                                      |
| ML Models          | Logistic Regression, Random Forest, XGBoost, KNN, SVM |
| Resampling         | SMOTE (imbalanced-learn)                   |
| Preprocessing      | Label Encoding, StandardScaler             |
| Evaluation         | Confusion Matrix, F1-score, ROC AUC        |
| UI                 | Streamlit                                   |

---

## 🚀 How to Run

### ✅ Install Requirements

```bash
pip install -r requirements.txt
# or manually:
pip install streamlit scikit-learn imbalanced-learn xgboost pandas numpy
```

### ▶️ Launch Streamlit App

```bash
streamlit run hr_attrition_ui.py
```

## 🧠 Features

- Trains and compares multiple models:
  - Logistic Regression
  - Random Forest
  - KNN
  - SVM
  - XGBoost
  - Voting Ensemble (soft voting of top 3)
- Handles class imbalance using SMOTE
- Real-time prediction via Streamlit UI
- Shows probability/confidence of attrition
- Uses label encoding + scaling behind the scenes

## 📈 Sample Output

```
Confusion Matrix:
[[362  11]
 [ 29  30]]

Classification Report:
              precision    recall  f1-score
Attrition 1     0.73       0.51     0.60

ROC AUC Score: 0.91
```

## 🧾 Project Structure

```bash
.
├── dataset/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── hr_attrition_multimodel.py       # Trains & evaluates all models
├── hr_attrition_ui.py               # Streamlit-based prediction UI
├── requirements.txt
└── README.md
```

## 📚 Useful Concepts Covered

- SMOTE (Synthetic Minority Over-sampling)
- Multi-model comparison
- Feature scaling and label encoding
- VotingClassifier (soft voting)
- UI deployment using Streamlit

## 📜 License

MIT © 2025 Mayank Raval

### Built with ❤️ to detect employee attrition using machine learning.
