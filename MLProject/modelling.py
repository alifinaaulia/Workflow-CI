import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import mlflow
import mlflow.pyfunc
import joblib
import os

print("Tracking URI:", mlflow.get_tracking_uri())

class SVDRecommender(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import joblib
        self.svd = joblib.load(context.artifacts["svd_model"])

    def predict(self, context, model_input):
        return self.svd.transform(model_input)

def run_hybrid_modeling(n_components):
    with mlflow.start_run(run_name="Hybrid Recommender") as run:
        # Load data
        df = pd.read_csv("online_retail_preprocessing.csv")

        # Matrix pembelian
        user_item_matrix = df.pivot_table(index='CustomerID', columns='Description',
                                          values='TotalPrice', aggfunc='sum', fill_value=0)

        # Collaborative Filtering (SVD)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        latent_matrix = svd.fit_transform(user_item_matrix)

        # Simpan model sementara
        model_path = "svd_model.pkl"
        joblib.dump(svd, model_path)

        # Logging model dalam format pyfunc agar bisa di-build Docker
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=SVDRecommender(),
            artifacts={"svd_model": model_path}
        )

        print("\nModel successfully logged. Run ID:", run.info.run_id)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_components", type=int, default=50)
    args = parser.parse_args()

    run_hybrid_modeling(args.n_components)
