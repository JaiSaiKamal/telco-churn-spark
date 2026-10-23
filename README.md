# Telecom Customer Churn Prediction

An end-to-end Big Data customer churn prediction project using **Apache Spark MLlib** and **Streamlit**. It trains and benchmarks multiple Spark machine learning pipelines, visualizes model comparisons, identifies the best-performing algorithm, inspects feature importance, and provides interactive single-customer scoring and CSV batch predictions.

---

## Supported Machine Learning Models

The pipeline now benchmarks four diverse Spark MLlib classification algorithms on the exact same 80/20 train/test split:

1. **Logistic Regression** — Regularized linear baseline model.
2. **Random Forest Classifier** — Ensemble of 50 bagging decision trees with feature sub-sampling.
3. **Gradient-Boosted Trees (GBT)** — Sequential tree boosting algorithm, optimized for tabular data.
4. **Decision Tree Classifier** — Single decision tree for interpretable rules.

The best-performing model (by ROC AUC) is automatically identified and used by default, while the Streamlit UI allows toggling between any model for interactive predictions.

---

## Dataset

This project uses the [Telco Customer Churn dataset by BlastChar on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), an IBM sample dataset. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` and place it at:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

## Run Locally (Windows PowerShell)

Requires **Python 3.10 or 3.11** and **Java 17**. Spark MLlib requires a Java runtime and Hadoop winutils on Windows.

```powershell
# 1. Activate your virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train and benchmark all models
python train.py

# 4. Launch the Streamlit dashboard
streamlit run app.py
```

### Command-line Options for Training
To train only a specific model:
```powershell
python train.py --model-type gbt
python train.py --model-type random_forest
python train.py --model-type logistic_regression
python train.py --model-type decision_tree
```
Or train all (default):
```powershell
python train.py --model-type all
```

---

## Streamlit Features

* **Model Comparison & Overview Tab**: Benchmarks Accuracy, F1-Score, and ROC AUC side-by-side with comparison bar charts, and displays the top feature importances driving customer churn.
* **Predict a Customer Tab**: Interactive form allowing you to select which model to score with and view estimated churn risk probability.
* **Batch Scoring Tab**: Upload a CSV of customers, score them using any chosen model, and download the annotated CSV.
* **Sidebar Controls**: Real-time status of all trained models and a one-click retraining button.

---

## Project Structure

- `app.py` — Streamlit interactive multi-model dashboard and predictor
- `src/model.py` — Spark MLlib pipelines (Logistic Regression, Random Forest, GBT, Decision Tree), feature extraction, evaluation
- `train.py` — Command-line training and benchmarking script
- `requirements.txt` — PySpark, Streamlit, pandas, numpy
- `Dockerfile`, `docker-compose.yml` — Container deployment
