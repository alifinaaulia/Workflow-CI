import os
import argparse
import pandas as pd
import numpy as np
import joblib
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
        self.tfidf = joblib.load(context.artifacts["tfidf_vectorizer"])
        self.cosine_sim = np.load(context.artifacts["cosine_sim"])
        self.product_to_index = joblib.load(context.artifacts["product_to_index"])
        self.index_to_product = joblib.load(context.artifacts["index_to_product"])
        self.user_item_matrix = joblib.load(context.artifacts["user_item_matrix"])

    def predict(self, context, model_input):
        # Ambil input dari request
        customer_id = model_input.iloc[0]["CustomerID"]
        query_product = model_input.iloc[0]["QueryProduct"]

        if customer_id not in self.user_item_matrix.index:
            return [f"CustomerID {customer_id} not found"]

        if query_product not in self.product_to_index:
            return [f"QueryProduct '{query_product}' not found"]

        # Collaborative Filtering
        customer_vector = self.user_item_matrix.loc[customer_id].reindex(self.model_columns, fill_value=0)
        latent_vector = self.svd.transform([customer_vector])
        item_embeddings = normalize(self.svd.components_.T)
        collab_scores = latent_vector.dot(item_embeddings.T).flatten()

        # Content-Based
        content_scores = self.cosine_sim[self.product_to_index[query_product]]

        # Hybrid
        hybrid_scores = (collab_scores + content_scores) / 2
        top_indices = np.argsort(hybrid_scores)[::-1][:10]
        recommendations = [self.index_to_product[i] for i in top_indices]
        return recommendations


def train_and_log_model(n_components):
    df = pd.read_csv("online_retail_preprocessing.csv")

    user_item_matrix = df.pivot_table(index='CustomerID', columns='Description',
                                      values='TotalPrice', aggfunc='sum', fill_value=0)
    products = user_item_matrix.columns.tolist()

    # Collaborative Filtering
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(user_item_matrix)

    # Content-Based Filtering
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(products)
    cosine_sim = cosine_similarity(tfidf_matrix)

    # Mapping produk
    product_to_index = {desc: i for i, desc in enumerate(products)}
    index_to_product = {i: desc for i, desc in enumerate(products)}

    # Simpan semua artifacts
    artifacts = {
        "svd_model": os.path.abspath("svd_model.pkl"),
        "model_columns": os.path.abspath("model_columns.pkl"),
        "tfidf_vectorizer": os.path.abspath("tfidf_vectorizer.pkl"),
        "cosine_sim": os.path.abspath("cosine_sim.npy"),
        "product_to_index": os.path.abspath("product_to_index.pkl"),
        "index_to_product": os.path.abspath("index_to_product.pkl"),
        "user_item_matrix": os.path.abspath("user_item_matrix.pkl")
    }

    joblib.dump(svd, artifacts["svd_model"])
    joblib.dump(products, artifacts["model_columns"])
    joblib.dump(tfidf, artifacts["tfidf_vectorizer"])
    np.save(artifacts["cosine_sim"], cosine_sim)
    joblib.dump(product_to_index, artifacts["product_to_index"])
    joblib.dump(index_to_product, artifacts["index_to_product"])
    joblib.dump(user_item_matrix, artifacts["user_item_matrix"])

    # Logging ke MLflow
    with mlflow.start_run(run_name="Hybrid Recommender") as run:
        mlflow.log_param("n_components", n_components)
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=HybridRecommender(),
            artifacts=artifacts
        )
        print(f"\nModel successfully logged. Run ID: {run.info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_components", type=int, default=50)
    args = parser.parse_args()
    train_and_log_model(args.n_components)
