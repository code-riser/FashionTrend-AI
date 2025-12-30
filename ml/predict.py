"""
Fashion Sales Trend Prediction Module
-------------------------------------
Project: FashionTrend AI
Use Case: Predict future fashion sales trends for marketing decisions

ML Type: Supervised Learning (Regression)
Dataset: CSV-based fashion sales data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class FashionTrendPredictor:
    def __init__(self, csv_path):
        """
        Initialize predictor with dataset path
        """
        self.csv_path = csv_path
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.data = None
        self.features = None
        self.target = None

    # ================= LOAD DATA =================
    def load_data(self):
        """
        Load CSV dataset
        Expected columns (example):
        - category
        - season
        - price
        - sales
        - month
        """
        self.data = pd.read_csv(self.csv_path)
        return self.data

    # ================= PREPROCESS =================
    def preprocess(self):
        """
        Clean and preprocess dataset
        """
        df = self.data.copy()

        # Drop missing values
        df.dropna(inplace=True)

        # Encode categorical columns
        encoder = LabelEncoder()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = encoder.fit_transform(df[col])

        # Feature & target separation
        self.features = df.drop('sales', axis=1)
        self.target = df['sales']

        # Feature scaling
        self.features = self.scaler.fit_transform(self.features)

        return self.features, self.target

    # ================= TRAIN MODEL =================
    def train(self):
        """
        Train regression model
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)

        metrics = {
            "MAE": round(mean_absolute_error(y_test, predictions), 2),
            "MSE": round(mean_squared_error(y_test, predictions), 2),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, predictions)), 2),
            "R2_Score": round(r2_score(y_test, predictions), 2)
        }

        return metrics

    # ================= PREDICT =================
    def predict_sales(self, input_data: dict):
        """
        Predict sales for new fashion input
        input_data example:
        {
            "category": 1,
            "season": 2,
            "price": 1999,
            "month": 6
        }
        """
        input_df = pd.DataFrame([input_data])
        input_scaled = self.scaler.transform(input_df)
        prediction = self.model.predict(input_scaled)

        return round(float(prediction[0]), 2)

    # ================= INSIGHTS =================
    def generate_insights(self):
        """
        Generate AI-style insights for UI
        """
        avg_sales = self.target.mean()
        max_sales = self.target.max()
        min_sales = self.target.min()

        insights = [
            f"Average fashion sales observed: {round(avg_sales, 2)} units.",
            f"Peak sales reached up to {max_sales} units for high-demand items.",
            f"Lowest performing products sold around {min_sales} units.",
            "Season and price significantly impact fashion sales trends.",
            "AI model can assist in planning inventory and marketing campaigns."
        ]

        return insights


# ================= TEST RUN =================
if __name__ == "__main__":
    predictor = FashionTrendPredictor("data/fashion_sales.csv")
    predictor.load_data()
    predictor.preprocess()
    metrics = predictor.train()

    print("Model Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")



from ml.data_loader import FashionDataLoader

loader = FashionDataLoader("data/fashion_sales.csv")
loader.load_csv()
loader.validate(mode="sales")
data = loader.clean()
