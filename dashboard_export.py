import pandas as pd
import rag_pipeline
import train_model
import feature_extraction
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def generate_dashboard_data(csv_path: str, output_path: str):
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    # 1. Generate Predictions (Simulating a production model usage)
    print("Training model for predictions...")
    # In a real scenario, we would load the saved model from MLflow
    # Here we quickly retrain for the export
    df_processed = feature_extraction.build_feature_pipeline(df.copy())
    df_encoded = pd.get_dummies(df_processed, columns=["sex", "smoker", "region", "bmi_category", "age_group"], drop_first=True)
    X = df_encoded.drop(["charges", "text"], axis=1, errors="ignore")
    y = df_encoded["charges"]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    df["Predicted_Charges"] = model.predict(X)
    
    # 2. Generate RAG Explanations/Insights
    print("Building RAG index...")
    # Ensure text column exists
    if "text" not in df.columns:
        df = rag_pipeline.load_medical_data(csv_path)
        
    vs = rag_pipeline.build_faiss_index(df["text"].tolist())
    
    print("Generating insights...")
    # For the dashboard, we might want to find similar cases for each record to explain the cost
    # This can be slow for 1000+ records, so let's do it for a sample or just add a generic query column
    
    # Let's add a "Similar Case Charges" column by querying the RAG system
    # We'll query using the record's own text description
    
    def get_similar_avg_cost(text_desc):
        results = rag_pipeline.query_rag(vs, text_desc, k=3)
        # Extract charges from results (assuming format "Charges: 1234.56")
        costs = []
        for doc in results:
            try:
                # Parse charges from the text string
                parts = doc.page_content.split("Charges: ")
                if len(parts) > 1:
                    cost = float(parts[1])
                    costs.append(cost)
            except:
                pass
        return sum(costs) / len(costs) if costs else 0

    # Optimization: Only do this for a subset or if requested. 
    # For 1000 records it might take a minute.
    df["Similar_Patients_Avg_Cost"] = df["text"].apply(get_similar_avg_cost)
    
    print(f"Saving dashboard data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    generate_dashboard_data("insurance.csv", "dashboard_export.csv")
