import pandas as pd
from sklearn.linear_model import LinearRegression

def train_and_predict(csv_path):
    df = pd.read_csv(csv_path)
    df = df.select_dtypes(include=["number"]).dropna()

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    model = LinearRegression()
    model.fit(X, y)

    df["Predicted_Sales"] = model.predict(X)
    return df.head(20).to_html(classes="data-preview")
