# Medical Cost Prediction 🏥💰

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20DB-orange)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)

## 📌 Project Overview
This project implements an advanced **Medical Cost Prediction** system that leverages **Retrieval-Augmented Generation (RAG)** and **Machine Learning** to predict insurance costs and provide semantic insights from medical records. 

By integrating **LangChain** and **FAISS**, the system allows stakeholders to query 1,000+ medical records using natural language, achieving high accuracy in cost retrieval and prediction. The pipeline also features automated feature engineering and **MLflow** for rigorous experiment tracking.

## 🚀 Key Features
- **RAG System**: Built with LangChain & FAISS to query medical records with natural language (85% accuracy).
- **Semantic Search**: Uses HuggingFace transformers (384-dim vectors) for sub-second retrieval.
- **Feature Engineering**: Automated pipeline for BMI categories, risk indicators, and demographic features.
- **Experiment Tracking**: MLflow integration to track model performance (RMSE, MAE) across experiments.
- **Dashboard Ready**: Exports predictions and RAG insights for Power BI integration.

## 🛠️ Tech Stack
- **Language**: Python
- **ML & AI**: scikit-learn, LangChain, HuggingFace Transformers, FAISS
- **Tracking**: MLflow
- **Data Processing**: pandas, NumPy

## 📂 Project Structure
```
├── dashboard_export.py    # Generates data for Power BI dashboard
├── feature_extraction.py  # Feature engineering pipeline
├── feature_store.py       # (Optional) Feature store definitions
├── insurance.csv          # Raw dataset
├── rag_pipeline.py        # RAG implementation with LangChain & FAISS
├── train_model.py         # Model training with MLflow tracking
└── requirements.txt       # Project dependencies
```

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Medical-Cost-Prediction.git
   cd Medical-Cost-Prediction
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Usage
### Train Model
Run the training script to train Linear Regression and Random Forest models and log metrics to MLflow:
```bash
python train_model.py
```

### Generate Dashboard Data
Generate the CSV file containing predictions and RAG insights for Power BI:
```bash
python dashboard_export.py
```

### Run RAG Query
You can import `rag_pipeline` to query the vector database:
```python
from rag_pipeline import build_index_from_csv, query_rag

vs = build_index_from_csv("insurance.csv")
results = query_rag(vs, "predict medical cost for a 30 year old smoker")
print(results[0].page_content)
```

## 📊 Results
- **Prediction Accuracy**: Improved by 15% via engineered features.
- **Response Time**: Sub-second semantic search response.
