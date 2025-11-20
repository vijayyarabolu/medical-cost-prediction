import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document


def load_medical_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Combine relevant fields into a text representation per record
    df["text"] = (
        "Age: " + df["age"].astype(str)
        + ", Sex: " + df["sex"].astype(str)
        + ", BMI: " + df["bmi"].astype(str)
        + ", Children: " + df["children"].astype(str)
        + ", Smoker: " + df["smoker"].astype(str)
        + ", Region: " + df["region"].astype(str)
        + ", Charges: " + df["charges"].astype(str)
    )
    return df


def build_faiss_index(texts):
    # Use a small, 384-dimensional sentence-transformer to match resume description
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    docs = [Document(page_content=t) for t in texts]
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store


def build_index_from_csv(csv_path: str) -> FAISS:
    df = load_medical_data(csv_path)
    texts = df["text"].tolist()
    vector_store = build_faiss_index(texts)
    return vector_store


def query_rag(vector_store: FAISS, query: str, k: int = 5):
    return vector_store.similarity_search(query, k=k)


if __name__ == "__main__":
    # Minimal demo runner
    csv_path = "insurance.csv"
    vs = build_index_from_csv(csv_path)
    results = query_rag(vs, "predict medical cost for a 30 year old smoker")
    for i, doc in enumerate(results, start=1):
        print(f"--- Result {i} ---")
        print(doc.page_content)
