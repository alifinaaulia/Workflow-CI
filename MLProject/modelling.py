import pandas as pd
import numpy as np
import argparse
import joblib
import os
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import mlflow
import mlflow.pyfunc


class HybridRecommender(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.svd = joblib.load(context.artifacts["svd_model"])
        self.model_columns = joblib.load(context.artifacts["model_columns"])
        self.tfidf = joblib.load(context.artifacts["tfidf"])
        self.cosine_sim = joblib.load(context.artifacts["cosine_sim"])
        self.product_to_index = joblib.load(context.artifacts["product_to_index"])
        self.index_to_product = joblib.load(context.artifacts["index_to_product"])
        self.user_index_map = joblib.load(context.artifacts["user_index_map"])
        self.user_embeddings = joblib.load(context.artifacts["user_embeddings"])
        self.item_embeddings = normalize(self.svd.components_.T)

    def predict(self, context, model_input):
        customer_id = model_input["customer_id"]
        query_product = model_input["product"]

        if customer_id in self.user_index_map and query_product in self.product_to_index:
            user_idx = self.user_index_map[customer_id]
            item_idx = self.product_to_index[query_product]

            collab_scores = self.user_embeddings[user_idx].dot(self.item_embeddings.T)
            content_scores = self.cosine_sim[item_idx]
            hybrid_scores = (collab_scores + content_scores) / 2

            top_indices = np.argsort(hybrid_scores)[::-1][:10]
            recommendations = [self.index_to_product[i] for i in top_indices]
            return recommendations
        else:
            return ["Invalid input: customer_id or product not found."]


def run_hybrid_modeling(n_components):
    # Cegah run ID ghost error
    if mlflow.active_run():
        mlflow.end_run()
    mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    with mlflow.start_run(run_name="Hybrid Recommender", nested=True) as run:
        # Load from GitHub Actions env or fallback to local
        csv_url = os.getenv("CSV_URL", "MLProject/online_retail_preprocessing.csv")
        df = pd.read_csv(csv_url)

        user_item_matrix = df.pivot_table(index='CustomerID', columns='Description',
                                          values='TotalPrice', aggfunc='sum', fill_value=0)

        svd = TruncatedSVD(n_components=n_components, random_state=42)
        latent_matrix = svd.fit_transform(user_item_matrix)
        user_embeddings = normalize(latent_matrix)
        item_embeddings = normalize(svd.components_.T)

        user_index_map = {cid: idx for idx, cid in enumerate(user_item_matrix.index)}

        products = user_item_matrix.columns.tolist()
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(products)
        cosine_sim = cosine_similarity(tfidf_matrix)

        product_to_index = {desc: i for i, desc in enumerate(products)}
        index_to_product = {i: desc for desc, i in product_to_index.items()}

        # Simpan artefak
        joblib.dump(svd, "svd_model.pkl")
        joblib.dump(user_item_matrix.columns, "model_columns.pkl")
        joblib.dump(tfidf, "tfidf.pkl")
        joblib.dump(cosine_sim, "cosine_sim.pkl")
        joblib.dump(product_to_index, "product_to_index.pkl")
        joblib.dump(index_to_product, "index_to_product.pkl")
        joblib.dump(user_index_map, "user_index_map.pkl")
        joblib.dump(user_embeddings, "user_embeddings.pkl")

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=HybridRecommender(),
            artifacts={
                "svd_model": "svd_model.pkl",
                "model_columns": "model_columns.pkl",
                "tfidf": "tfidf.pkl",
                "cosine_sim": "cosine_sim.pkl",
                "product_to_index": "product_to_index.pkl",
                "index_to_product": "index_to_product.pkl",
                "user_index_map": "user_index_map.pkl",
                "user_embeddings": "user_embeddings.pkl"
            }
        )

        print(f"Model logged. Run ID: {run.info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_components", type=int, default=50)
    args = parser.parse_args()

    run_hybrid_modeling(args.n_components)
