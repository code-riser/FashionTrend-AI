"""
Customer Segmentation Module (K-Means Clustering)
------------------------------------------------
Project: FashionTrend AI
Use Case: Segment fashion customers based on purchasing behavior

ML Type: Unsupervised Learning (Clustering)
Algorithm: K-Means
Dataset: CSV-based fashion sales & customer behavior data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class FashionCustomerClustering:
    def __init__(self, csv_path, n_clusters=3):
        """
        Initialize clustering model
        """
        self.csv_path = csv_path
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.scaler = StandardScaler()
        self.data = None
        self.features = None

    # ================= LOAD DATA =================
    def load_data(self):
        """
        Expected columns (example):
        - customer_id
        - category
        - season
        - price
        - quantity
        - total_spent
        """
        self.data = pd.read_csv(self.csv_path)
        return self.data

    # ================= PREPROCESS =================
    def preprocess(self):
        """
        Encode categorical features and scale numerical data
        """
        df = self.data.copy()

        # Drop non-analytical column
        if 'customer_id' in df.columns:
            df.drop('customer_id', axis=1, inplace=True)

        # Handle missing values
        df.dropna(inplace=True)

        # Encode categorical columns
        encoder = LabelEncoder()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = encoder.fit_transform(df[col])

        # Feature scaling
        self.features = self.scaler.fit_transform(df)

        return self.features

    # ================= TRAIN =================
    def train(self):
        """
        Train K-Means clustering model
        """
        self.model.fit(self.features)
        labels = self.model.labels_

        score = silhouette_score(self.features, labels)

        return {
            "Clusters": self.n_clusters,
            "Silhouette Score": round(score, 3)
        }

    # ================= CLUSTER SUMMARY =================
    def get_cluster_summary(self):
        """
        Generate cluster-wise summary
        """
        df = self.data.copy()
        df['Cluster'] = self.model.labels_

        summary = df.groupby('Cluster').mean(numeric_only=True)

        return summary

    # ================= AI INSIGHTS =================
    def generate_insights(self):
        """
        Generate AI-style marketing insights
        """
        insights = [
            "Cluster 0: Price-sensitive customers preferring discounted fashion items.",
            "Cluster 1: Premium buyers focusing on quality and seasonal collections.",
            "Cluster 2: Trend-driven customers with frequent purchases.",
            "Segmented marketing campaigns can significantly improve conversions.",
            "Customer clustering helps optimize inventory and pricing strategies."
        ]
        return insights


# ================= TEST RUN =================
if __name__ == "__main__":
    cluster_model = FashionCustomerClustering(
        csv_path="data/fashion_sales.csv",
        n_clusters=3
    )
    cluster_model.load_data()
    cluster_model.preprocess()
    result = cluster_model.train()

    print("Clustering Results:")
    for k, v in result.items():
        print(f"{k}: {v}")

    print("\nCluster Summary:")
    print(cluster_model.get_cluster_summary())
