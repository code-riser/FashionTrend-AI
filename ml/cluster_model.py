import pandas as pd
from sklearn.cluster import KMeans

def cluster_customers(csv_path):
    df = pd.read_csv(csv_path)
    df = df.select_dtypes(include=["number"]).dropna()

    model = KMeans(n_clusters=3, random_state=42)
    df["Cluster"] = model.fit_predict(df)

    return df.head(20).to_html(classes="data-preview")
