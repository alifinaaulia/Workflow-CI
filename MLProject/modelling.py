import os
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
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

    def predict(self, context, model_input):
        import pandas as pd
        data = pd.read_csv("online_retail_preprocessing.csv")
        user_item_matrix = data.pivot_table(index='CustomerID', columns='Description',
                                            values='TotalPrice', aggfunc='sum', fill_value=0)

        customer_id = model_input["CustomerID"]
        query_product = model_input["QueryProduct"]

        if customer_id not in user_item_matrix.index:
            return [f"CustomerID {customer_id} not found"]

        if query_product not in self.product_to_index:
            return [f"QueryProduct '{query_product}' not found"]

        # Collaborative
        customer_vector = user_item_matrix.loc[customer_id].reindex(self.model_columns, fill_value=0)
        latent_vector = self.svd.transform([customer_vector])
        item_embeddings = normalize(self.svd.components_.T)
        collab_scores = latent_vector.dot(item_embeddings.T).flatten()

        # Content
        content_scores = self.cosine_sim[self.product_to_index[query_product]]

        # Hybrid
        hybrid_scores = (collab_scores + content_scores) / 2
        top_indices = np.argsort(hybrid_scores)[::-1][:10]
        recommendations = [self.index_to_product[i] for i in top_indices]

        return recommendations


def train_and_log(n_components):
    df = pd.read_csv("online_retail_preprocessing.csv")
    user_item_matrix = df.pivot_table(index='CustomerID', columns='Description',
                                      values='TotalPrice', aggfunc='sum', fill_value=0)
    products = user_item_matrix.columns.tolist()

    # Collaborative
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(user_item_matrix)
    joblib.dump(svd, "svd_model.pkl")
    joblib.dump(products, "model_columns.pkl")

    # Content-Based
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(products)
    cosine_sim = cosine_similarity(tfidf_matrix)
    np.save("cosine_sim.npy", cosine_sim)
    joblib.dump(tfidf, "tfidf_vectorizer.pkl")

    product_to_index = {desc: i for i, desc in enumerate(products)}
    index_to_product = {i: desc for i, desc in enumerate(products)}
    joblib.dump(product_to_index, "product_to_index.pkl")
    joblib.dump(index_to_product, "index_to_product.pkl")

    # Logging model (NO start_run here)
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=HybridRecommender(),
        artifacts={
            "svd_model": "svd_model.pkl",
            "model_columns": "model_columns.pkl",
            "tfidf_vectorizer": "tfidf_vectorizer.pkl",
            "cosine_sim": "cosine_sim.npy",
            "product_to_index": "product_to_index.pkl",
            "index_to_product": "index_to_product.pkl"
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_components", type=int, default=50)
    args = parser.parse_args()

    train_and_log(args.n_components)
