# Breast Cancer ML Predictor

This project uses Logistic Regression to predict whether a tumor is benign or malignant using the UCI Breast Cancer dataset.

## 📁 Structure

- `data_processing.py`: Loads and prepares the dataset
- `model.py`: Builds and evaluates the logistic regression model
- `train.py`: Main script to execute the pipeline
- `reports/`: Auto-generated visual results (confusion matrix and ROC curve)
- `requirements.txt`: Required libraries

## 🚀 How to Run

```bash
pip install -r requirements.txt
python train.py
```

## 📊 Outputs

After running, plots will be saved in the `reports/` folder:
- `confusion_matrix.png`
- `roc_curve.png`

## 📌 Dataset

We use `sklearn.datasets.load_breast_cancer()`
