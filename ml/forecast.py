"""
Fashion Demand Forecasting Module
--------------------------------
Project: FashionTrend AI
Use Case: Forecast future demand for fashion products

ML Type: Supervised Learning (Regression + Trend Awareness)
Algorithm: Random Forest Regressor
Dataset: CSV-based fashion sales & demand data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


class FashionDemandForecaster:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.model = RandomForestRegressor(
            n_estimators=150,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.data = None
        self.features = None
        self.target = None

    # ================= LOAD DATA =================
    def load_data(self):
        """
        Expected columns (example):
        - category
        - season
        - price
        - month
        - demand
        """
        self.data = pd.read_csv(self.csv_path)
        return self.data

    # ================= PREPROCESS =================
    def preprocess(self):
        df = self.data.copy()
        df.dropna(inplace=True)

        # Encode categorical columns
        encoder = LabelEncoder()
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = encoder.fit_transform(df[col])

        self.features = df.drop('demand', axis=1)
        self.target = df['demand']

        self.features = self.scaler.fit_transform(self.features)
        return self.features, self.target

    # ================= TRAIN =================
    def train(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.target, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)

        metrics = {
            "MAE": round(mean_absolute_error(y_test, predictions), 2),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, predictions)), 2)
        }
        return metrics

    # ================= FORECAST =================
    def forecast_demand(self, input_data: dict):
        """
        input_data example:
        {
            "category": 1,
            "season": 2,
            "price": 2499,
            "month": 9
        }
        """
        input_df = pd.DataFrame([input_data])
        input_scaled = self.scaler.transform(input_df)
        prediction = self.model.predict(input_scaled)

        return round(float(prediction[0]), 2)

    # ================= AI INSIGHTS =================
    def generate_insights(self):
        avg_demand = self.target.mean()
        peak_demand = self.target.max()

        insights = [
            f"Average demand level observed: {round(avg_demand, 2)} units.",
            f"Peak demand reached approximately {peak_demand} units.",
            "Seasonality plays a major role in fashion demand.",
            "Price-sensitive items show fluctuating demand patterns.",
            "Demand forecasting helps reduce overstock and stockouts."
        ]
        return insights


# ================= TEST RUN =================
if __name__ == "__main__":
    forecaster = FashionDemandForecaster("data/fashion_sales.csv")
    forecaster.load_data()
    forecaster.preprocess()
    metrics = forecaster.train()

    print("Demand Forecasting Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")
