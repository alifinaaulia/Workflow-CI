import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import mlflow
import mlflow.sklearn
import os
import joblib

print("Tracking URI:", mlflow.get_tracking_uri())

def run_hybrid_modeling(n_components):


    with mlflow.start_run(run_name="Hybrid Recommender") as run:
        # Load data
        df = pd.read_csv("online_retail_preprocessing.csv")

        # Matrix pembelian
        user_item_matrix = df.pivot_table(index='CustomerID', columns='Description', values='TotalPrice', aggfunc='sum', fill_value=0)

        # Collaborative Filtering (SVD)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        latent_matrix = svd.fit_transform(user_item_matrix)
        user_embeddings = normalize(latent_matrix)
        item_embeddings = normalize(svd.components_.T)

        # Content-Based (TF-IDF)
        products = user_item_matrix.columns.tolist()
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(products)
        cosine_sim = cosine_similarity(tfidf_matrix)

        product_to_index = {desc: i for i, desc in enumerate(products)}
        index_to_product = {i: desc for desc, i in product_to_index.items()}

        sample_customer = 17850.0
        sample_product = "WHITE HANGING HEART T-LIGHT HOLDER"

        recommendations = []
        if sample_customer in user_item_matrix.index and sample_product in product_to_index:
            user_idx = user_item_matrix.index.get_loc(sample_customer)
            item_idx = product_to_index[sample_product]

            collab_scores = user_embeddings[user_idx].dot(item_embeddings.T)
            content_scores = cosine_sim[item_idx]
            hybrid_scores = (collab_scores + content_scores) / 2

            top_indices = np.argsort(hybrid_scores)[::-1][:10]
            recommendations = [index_to_product[i] for i in top_indices]

        print("SVD Model:", svd)

        joblib.dump(svd, "model.pkl")
        mlflow.log_artifact("model.pkl", artifact_path="model")




        # Output
        print("\nHybrid Recommender Inference")
        print("Customer:", sample_customer)
        print("Query Product:", sample_product)
        print("Top 10 Recommendations:", recommendations)

        # Cek lokasi model tersimpan
        print(f"\nModel successfully logged. Run ID: {run.info.run_id}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_components", type=int, default=50)
    args = parser.parse_args()

    run_hybrid_modeling(args.n_components)

