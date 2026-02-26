# 📊 B2B Beverage Sales Forecasting & Customer Segmentation

**🔴 Live Dashboard:** [B2B Beverage Sales Predictor](https://b2b-beverage-sales-prediction.streamlit.app/)

## 🎯 Business Problem Statement
In the highly competitive wholesale beverage industry, maximizing revenue requires a deep understanding of purchasing behavior and seasonal demand. This project analyzes approximately 9 million B2B transactions to identify high-value client profiles, forecast peak sales periods, and predict whether incoming orders will be of high monetary value. The goal is to provide actionable intelligence for inventory management, targeted marketing, and dynamic pricing.

## 📈 Economic & Financial Concepts Applied
* **Revenue Optimization & Product Mix:** Our Exploratory Data Analysis (EDA) revealed that "Alcoholic Beverages" hold a disproportionate share of total revenue compared to soft drinks and water. This dictates a supply-side focus on maintaining high inventory levels for these premium margins.
* **Seasonality & Supply Chain Forecasting:** Time-series analysis indicates clear seasonal fluctuations, with demand consistently dipping early in the year and peaking mid-to-late year. This cyclical demand curve informs warehousing and logistics planning.
* **Pricing Strategy & Demand Elasticity:** Through K-Means clustering, we discovered that our most profitable client segments (generating over €230k average revenue) are consistently driven by an 8% average discount. Meanwhile, the bulk of our low-revenue clients receive 0% discount. This indicates that strategic discounting is a primary lever for driving high-volume sales.

## 🤖 AI Techniques Used
1. **Exploratory Data Analysis (EDA):** Time-series resampling and revenue aggregation using Pandas and Matplotlib.
2. **K-Means Clustering (Unsupervised Learning):** Segmented the B2B client base into 3 distinct tiers based on Total Revenue, Order Quantity, and Discount Rate, standardized via `StandardScaler`.
3. **Logistic Regression (Supervised Learning):** Built a predictive classification model to identify "High-Value Orders" based on transaction features. Handled severe class imbalance using `class_weight='balanced'`, achieving an 85.6% overall accuracy and an 83% recall rate for high-value targets.
4. **Interactive Deployment:** Built an interactive web application using Streamlit and Plotly to serve real-time predictions and visualize historical insights.

## 📂 Dataset
* [Beverage Sales Dataset on Kaggle](https://www.kaggle.com/datasets/sebastianwillmann/beverage-sales) (approx. 9 million rows)

## 📸 Project Outputs
*(Replace these text placeholders with actual screenshots of your dashboard and Colab outputs!)*
Streamlit Prediction Engine 
 <img width="1394" height="686" alt="image" src="https://github.com/user-attachments/assets/9081cb74-7796-4eb7-b651-1581d8aa248e" />

Plotly Pie Chart & Line Graph 
<img width="1394" height="686" alt="image" src="https://github.com/user-attachments/assets/8959dff5-a5fa-4dcb-95da-8aa69423785c" />

K-Means Scatter Plot 
<img width="1394" height="686" alt="image" src="https://github.com/user-attachments/assets/3a0cb447-7150-47f8-a9b5-f3223fc4ec3f" />


## 💻 How to Run Locally
1. Clone this repository: `git clone https://github.com/your-username/b2b-beverage-sales-prediction.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Run the Streamlit app: `streamlit run app.py`
