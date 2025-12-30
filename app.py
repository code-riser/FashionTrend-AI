# app.py

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
from datetime import datetime

from ml.data_loader import load_csv_preview
from ml.trend_model import train_and_predict
from ml.cluster_model import cluster_customers

app = Flask(__name__)
app.secret_key = "fashiontrend-ai-secret"

login_manager = LoginManager(app)
login_manager.login_view = "login"

DATA_FOLDER = "data"

# ---------------- USER MODEL ----------------
class User(UserMixin):
    def __init__(self, id, role="Admin"):
        self.id = id
        self.role = role

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# ---------------- AUTH ----------------
@app.route("/", methods=["GET"])
def home():
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User(id=request.form["email"])
        login_user(user)
        return redirect(url_for("dashboard"))
    return render_template("auth.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# ---------------- DASHBOARD ----------------
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template(
        "dashboard.html",
        total_sales="₹12.4M",
        top_category="Ethnic Wear",
        trending_season="Summer 2026",
        low_demand="Winter Jackets",
        categories=[
            {"name": "Casual Wear", "sales": "₹4.1M", "trend": "up"},
            {"name": "Ethnic Wear", "sales": "₹5.6M", "trend": "up"},
            {"name": "Footwear", "sales": "₹2.7M", "trend": "down"},
        ],
        customer_segments=[
            {"name": "Premium Buyers", "count": 420, "insight": "High-value repeat buyers"},
            {"name": "Seasonal Shoppers", "count": 780, "insight": "Festival driven"},
        ]
    )

# ---------------- ADMIN ----------------
def get_datasets():
    datasets = []
    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".csv"):
            datasets.append({
                "filename": file,
                "uploaded_by": "Admin",
                "date": datetime.now().strftime("%d %b %Y")
            })
    return datasets

@app.route("/admin")
@login_required
def admin_dashboard():
    return render_template("admin.html", datasets=get_datasets())

@app.route("/manage-datasets")
@login_required
def manage_datasets():
    previews = load_csv_preview(DATA_FOLDER)
    return render_template("manage_datasets.html", datasets=previews)

# ---------------- TREND PREDICTION ----------------
@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict_route():
    table = None
    if request.method == "POST":
        file = request.files["dataset"]
        path = os.path.join(DATA_FOLDER, file.filename)
        file.save(path)
        table = train_and_predict(path)
        flash("Prediction completed successfully", "success")
    return render_template("predict.html", table=table)

# ---------------- CLUSTERING ----------------
@app.route("/cluster", methods=["GET", "POST"])
@login_required
def cluster_route():
    table = None
    if request.method == "POST":
        file = request.files["dataset"]
        path = os.path.join(DATA_FOLDER, file.filename)
        file.save(path)
        table = cluster_customers(path)
        flash("Clustering completed successfully", "success")
    return render_template("cluster.html", table=table)

# ---------------- INSIGHTS & LOGS ----------------
@app.route("/insights")
@login_required
def insights():
    return "<h2>Insights coming from AI usage analytics</h2>"

@app.route("/admin-logs")
@login_required
def admin_logs():
    return "<h2>System logs page</h2>"

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)













from flask import Flask, render_template, redirect, url_for, send_file
from flask_login import LoginManager, UserMixin, login_required, logout_user, current_user
import os
import pandas as pd
from datetime import datetime

app = Flask(__name__)
app.secret_key = "fashiontrend-secret-key"

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

DATASET_FOLDER = "data"

# ---------------- USER MODEL ----------------
class User(UserMixin):
    def __init__(self, id, role="Admin"):
        self.id = id
        self.role = role

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# ---------------- DATASET MANAGER ----------------
def get_datasets():
    datasets = []
    for i, file in enumerate(os.listdir(DATASET_FOLDER)):
        if file.endswith(".csv"):
            datasets.append({
                "id": i,
                "filename": file,
                "uploaded_by": "Admin",
                "date": datetime.now().strftime("%d %b %Y")
            })
    return datasets

# ---------------- ROUTES ----------------

@app.route("/admin")
@login_required
def admin():
    datasets = get_datasets()
    return render_template(
        "admin.html",
        datasets=datasets
    )

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.route("/manage-datasets")
@login_required
def manage_datasets():
    return render_template("manage_datasets.html")

@app.route("/analysis")
@login_required
def analysis():
    return render_template("analysis.html")

@app.route("/insights")
@login_required
def insights():
    return render_template("insights.html")

@app.route("/admin-logs")
@login_required
def admin_logs():
    return render_template("logs.html")

@app.route("/download-dataset/<int:id>")
@login_required
def download_dataset(id):
    files = [f for f in os.listdir(DATASET_FOLDER) if f.endswith(".csv")]
    return send_file(os.path.join(DATASET_FOLDER, files[id]), as_attachment=True)

@app.route("/delete-dataset/<int:id>")
@login_required
def delete_dataset(id):
    files = [f for f in os.listdir(DATASET_FOLDER) if f.endswith(".csv")]
    os.remove(os.path.join(DATASET_FOLDER, files[id]))
    return redirect(url_for("admin"))

if __name__ == "__main__":
    app.run(debug=True)








"""
FashionTrend AI - Flask Web Application
Integrates dashboard, admin, auth, ML prediction, and clustering
"""

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
import pandas as pd

# Import ML modules
from utils.predict import train_model, predict_trend
from utils.cluster import train_customer_clusters, predict_customer_segment
from ml.data_loader import load_csv, validate_columns, clean_data

# ================= CONFIG =================
app = Flask(__name__)
app.secret_key = "supersecretkey"

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# ================= USER MANAGEMENT =================
# Dummy user store for example purposes
USERS = {
    "admin@example.com": {"password": "admin123", "role": "Admin"},
    "user@example.com": {"password": "user123", "role": "User"},
}

class User(UserMixin):
    def __init__(self, id, email, role):
        self.id = id
        self.email = email
        self.role = role

@login_manager.user_loader
def load_user(user_id):
    for idx, (email, data) in enumerate(USERS.items()):
        if idx == int(user_id):
            return User(id=idx, email=email, role=data['role'])
    return None

# ================= ROUTES =================
@app.route("/")
def home():
    return redirect(url_for("login"))

# -------- AUTH ROUTES --------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        for idx, (user_email, data) in enumerate(USERS.items()):
            if email == user_email and password == data['password']:
                user = User(id=idx, email=email, role=data['role'])
                login_user(user)
                return redirect(url_for("dashboard"))
        flash("Invalid credentials", "danger")
    return render_template("auth.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        flash("Registration feature not implemented in this demo.", "info")
    return render_template("auth.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# -------- DASHBOARD --------
@app.route("/dashboard")
@login_required
def dashboard():
    # Dummy example metrics
    total_sales = 150000
    top_category = "Men's Apparel"
    trending_season = "Summer 2026"
    low_demand = 5
    categories = [
        {"name": "Men's Apparel", "sales": 50000, "trend": "up"},
        {"name": "Women's Apparel", "sales": 40000, "trend": "stable"},
        {"name": "Footwear", "sales": 30000, "trend": "down"},
    ]
    customer_segments = [
        {"name": "Premium Buyers", "count": 120, "insight": "High Value"},
        {"name": "Occasional Buyers", "count": 250, "insight": "Medium Value"},
    ]
    sales_labels = ["Jan", "Feb", "Mar", "Apr", "May"]
    sales_data = [20000, 25000, 30000, 35000, 40000]

    return render_template("dashboard.html",
                           total_sales=total_sales,
                           top_category=top_category,
                           trending_season=trending_season,
                           low_demand=low_demand,
                           categories=categories,
                           customer_segments=customer_segments,
                           sales_labels=sales_labels,
                           sales_data=sales_data)

# -------- PREDICTION ROUTE --------
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    file = request.files.get("dataset")
    if not file:
        flash("Please upload a CSV file", "warning")
        return redirect(url_for("dashboard"))

    df = pd.read_csv(file)
    predictions = predict_trend(df)
    flash(f"Predictions generated for {len(predictions)} records", "success")
    return redirect(url_for("dashboard"))

# -------- CLUSTER ROUTE --------
@app.route("/cluster", methods=["POST"])
@login_required
def cluster():
    file = request.files.get("dataset")
    if not file:
        flash("Please upload a CSV file", "warning")
        return redirect(url_for("dashboard"))

    df = pd.read_csv(file)
    labels = predict_customer_segment(df)
    flash(f"Customer segments generated: {set(labels)}", "success")
    return redirect(url_for("dashboard"))

# -------- ADMIN DASHBOARD --------
@app.route("/admin")
@login_required
def admin_dashboard():
    if current_user.role != "Admin":
        flash("Access denied", "danger")
        return redirect(url_for("dashboard"))
    return render_template("admin.html")




   



# ================= RUN APP =================
if __name__ == "__main__":
    app.run(debug=True)
