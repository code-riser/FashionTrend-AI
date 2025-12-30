import pandas as pd
import os

def load_csv_preview(folder):
    datasets = []
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, file))
            datasets.append({
                "filename": file,
                "uploaded_by": "Admin",
                "date": "Today",
                "preview": df.head().to_html(classes="data-preview")
            })
    return datasets












"""
Data Loader & Preprocessing Engine
---------------------------------
Project: FashionTrend AI
Purpose:
- Validate CSV dataset
- Clean data
- Prepare features for ML models
- Maintain consistency across ML modules

This module is shared by:
- Trend Prediction
- Customer Segmentation
- Demand Forecasting
"""

import pandas as pd


class FashionDataLoader:
    REQUIRED_COLUMNS = {
        "sales": ["category", "season", "price", "month", "sales"],
        "demand": ["category", "season", "price", "month", "demand"],
        "customer": ["category", "season", "price", "quantity", "total_spent"]
    }

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = None

    # ================= LOAD =================
    def load_csv(self):
        """
        Load CSV safely
        """
        try:
            self.data = pd.read_csv(self.csv_path)
            return self.data
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {str(e)}")

    # ================= VALIDATE =================
    def validate(self, mode="sales"):
        """
        Validate required columns based on ML task
        mode options:
        - sales
        - demand
        - customer
        """
        if mode not in self.REQUIRED_COLUMNS:
            raise ValueError("Invalid validation mode selected")

        missing_cols = [
            col for col in self.REQUIRED_COLUMNS[mode]
            if col not in self.data.columns
        ]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return True

    # ================= CLEAN =================
    def clean(self):
        """
        Basic cleaning operations
        """
        df = self.data.copy()

        # Drop duplicates
        df.drop_duplicates(inplace=True)

        # Handle missing values
        df.dropna(inplace=True)

        # Reset index
        df.reset_index(drop=True, inplace=True)

        self.data = df
        return self.data

    # ================= SUMMARY =================
    def summary(self):
        """
        Dataset summary for admin panel
        """
        return {
            "rows": self.data.shape[0],
            "columns": self.data.shape[1],
            "column_names": list(self.data.columns)
        }

    # ================= EXPORT =================
    def export_clean_data(self, output_path):
        """
        Save cleaned dataset
        """
        self.data.to_csv(output_path, index=False)
        return output_path


# ================= TEST RUN =================
if __name__ == "__main__":
    loader = FashionDataLoader("data/fashion_trend.csv")
    loader.load_csv()
    loader.validate(mode="sales")
    loader.clean()

    print("Dataset Summary:")
    print(loader.summary())
