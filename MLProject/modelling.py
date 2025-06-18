import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mlflow
import mlflow.sklearn
import joblib
import os
import argparse

def load_data(data_path):
    return pd.read_csv(data_path)

def train_model(n_components, train_data, test_data):
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    latent_matrix = svd.fit_transform(train_data)
    reconstructed = np.dot(latent_matrix, svd.components_)
    reconstructed_df = pd.DataFrame(reconstructed, index=train_data.index, columns=train_data.columns)

    # Ambil data yang sama-sama tersedia di train dan test
    common_users = train_data.index.intersection(test_data.index)
    common_items = train_data.columns.intersection(test_data.columns)

    y_true = test_data.loc[common_users, common_items].values.flatten()
    y_pred = reconstructed_df.loc[common_users, common_items].values.flatten()

    # Gunakan hanya yang benar-benar ada transaksi
    mask = y_true > 0
    if mask.sum() == 0:
        return svd, float('inf')
    
    rmse = mean_squared_error(y_true[mask], y_pred[mask], squared=False)
    return svd, rmse

def get_hybrid_recommendations(df, svd_model, sample_customer, sample_product):
    user_item = df.pivot_table(index='CustomerID', columns='Description', values='TotalPrice', aggfunc='sum', fill_value=0)

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(user_item.columns)
    cosine_sim = cosine_similarity(tfidf_matrix)

    if sample_product not in user_item.columns or sample_customer not in user_item.index:
        return [], 0.0, 0.0, "cold_start"

    user_embeddings = normalize(svd_model.transform(user_item))
    item_embeddings = normalize(svd_model.components_.T)

    user_idx = user_item.index.get_loc(sample_customer)
    item_idx = list(user_item.columns).index(sample_product)

    collab_scores = user_embeddings[user_idx] @ item_embeddings.T
    content_scores = cosine_sim[item_idx]
    hybrid_scores = (collab_scores + content_scores) / 2

    top_indices = np.argsort(hybrid_scores)[::-1][:10]
    recommendations = [user_item.columns[i] for i in top_indices]

    actual_items = df[df["CustomerID"] == sample_customer]["Description"].unique().tolist()
    hits = len(set(recommendations) & set(actual_items))
    precision = hits / 10
    recall = hits / len(actual_items) if actual_items else 0.0

    return recommendations, precision, recall, "hybrid"

def main(data_path, sample_customer, sample_product):
    df = load_data(data_path)
    user_item = df.pivot_table(index='CustomerID', columns='Description', values='TotalPrice', aggfunc='sum', fill_value=0)

    # Split train-test 80:20
    mask = np.random.rand(len(user_item)) < 0.8
    train_data = user_item[mask]
    test_data = user_item[~mask]

    best_rmse = float("inf")
    best_model = None
    best_n = None
    best_metrics = {}
    fallback_model = None

    for n in [20, 50, 100, 150]:
        with mlflow.start_run(run_name=f"SVD n_components={n}"):
            svd_model, rmse = train_model(n, train_data, test_data)
            recommendations, precision, recall, method = get_hybrid_recommendations(df, svd_model, sample_customer, sample_product)

            mlflow.log_param("n_components", n)
            mlflow.log_param("sample_customer", sample_customer)
            mlflow.log_param("sample_product", sample_product)
            mlflow.log_param("recommendation_method", method)
            for i, rec in enumerate(recommendations):
                mlflow.log_param(f"rec_{i+1}", rec)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("precision_at_10", precision)
            mlflow.log_metric("recall_at_10", recall)

            print(f"Trained SVD with n_components={n}, RMSE={rmse:.4f}")
            fallback_model = svd_model

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = svd_model
                best_n = n
                best_metrics = {
                    "recommendations": recommendations,
                    "precision": precision,
                    "recall": recall
                }

    final_model = best_model if best_model is not None else fallback_model
    final_n = best_n if best_model is not None else n
    final_metrics = best_metrics if best_model is not None else {
        "recommendations": recommendations,
        "precision": precision,
        "recall": recall
    }

    with mlflow.start_run(run_name="Best SVD Model", nested=True):
        mlflow.log_param("best_n_components", final_n)
        mlflow.log_metric("best_rmse", best_rmse)
        mlflow.log_metric("precision_at_10", final_metrics["precision"])
        mlflow.log_metric("recall_at_10", final_metrics["recall"])

        os.makedirs("model_artifacts", exist_ok=True)
        joblib.dump(final_model, "model_artifacts/svd_best_model.pkl")
        mlflow.log_artifact("model_artifacts/svd_best_model.pkl")

        mlflow.sklearn.log_model(final_model, artifact_path="best_model")

        print("\nBest model used for inference")
        print("Customer:", sample_customer)
        print("Query Product:", sample_product)
        print("Top 10 Recommendations:", final_metrics["recommendations"])
        print(f"Precision@10: {final_metrics['precision']:.2f}")
        print(f"Recall@10: {final_metrics['recall']:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="online_retail_preprocessing.csv")
    parser.add_argument("--sample_customer", type=int, default=17850)
    parser.add_argument("--sample_product", type=str, default="WHITE HANGING HEART T-LIGHT HOLDER")
    args = parser.parse_args()

    main(args.data_path, args.sample_customer, args.sample_product)
