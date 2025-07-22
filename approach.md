
# Project Approach: House Price Prediction

This document details the full technical workflow behind the House Price Prediction project, from raw data to a deployed, interactive Streamlit app.

---

## 1. Data Loading & Initial Exploration

- **Source:** Data is loaded from CSV files in the `data/` folder (`train.csv`, `test.csv`).
- **Tools:** pandas for dataframes, matplotlib/seaborn for quick visual checks.
- **Goal:** Understand data types, missing values, and target (`SalePrice`) distribution.


---

## 2. Data Cleaning

- **Missing Values:**  
  - Numeric columns: filled with median.
  - Categorical columns: filled with mode.
- **Outliers:**  
  - Removed rows where `SalePrice` is more than 3 standard deviations from the mean.
- **Result:** Cleaned data saved as `cleaned_train.csv` for downstream use.


---

## 3. Feature Engineering

- **New Features:**  
  - `TotalSF` = `TotalBsmtSF` + `1stFlrSF` + `2ndFlrSF`
  - `HouseAge` = `YrSold` - `YearBuilt`
  - `RemodAge` = `YrSold` - `YearRemodAdd`
- **Encoding:**  
  - One-hot encoding for categorical variables (e.g., `MSZoning`, `Neighborhood`, `HouseStyle`).
- **Skewness Correction:**  
  - Applied `np.log1p` to highly skewed numeric features (excluding `SalePrice`).

---


## 4. Feature Selection

- **Correlation Analysis:**  
  - Selected features most correlated with `SalePrice`.
- **Dimensionality Reduction:**  
  - Dropped features with low importance or high collinearity.

---

## 5. Model Training & Evaluation

- **Models Tried:**  
  - Linear Regression, Ridge, Lasso, SVR, Random Forest, XGBoost.
- **Validation:**  
  - Used cross-validation (CV) and RMSE as the main metric.
- **Selection:**  
  - Chose the best-performing model based on lowest CV RMSE.
- **Persistence:**  
  - Saved the trained model and feature columns using `joblib` in the `models/` folder.

---

## 6. Final Pipeline & Prediction

- **Test Data:**  
  - Applied the same cleaning and feature engineering steps as training data.
- **Prediction:**  
  - Loaded the saved model and columns, aligned test data, and generated predictions.
- **Export:**  
  - Saved predictions as `final_predictions.csv` in the `output/` folder.

---

## 7. Streamlit App Development

- **App Structure:**  
  - Sidebar navigation: Predict, Batch Predict, EDA, About.
- **Single Prediction:**  
  - User inputs features, app performs feature engineering, predicts price, and shows summary/statistics/visualization.
- **Batch Prediction:**  
  - User uploads CSV, app processes all rows, predicts prices, highlights outliers, and allows result download.
- **EDA:**  
  - Interactive charts: price distribution, feature importance, correlation heatmap.
- **Robustness:**  
  - Uses only relative paths, `st.cache_data` for efficiency, and user-friendly error handling.
- **Deployment:**  
  - Ready for Streamlit Cloud or local use.

---

## 8. Outputs & Documentation

- **Notebooks:**  
  - Each stage (EDA & Cleaning, Feature Engineering, Final Pipeline) is in a separate, well-commented notebook.
- **Outputs:**  
  - All results and predictions are saved in the `output/` folder.
- **Documentation:**  
  - This approach file, a user-friendly README, and requirements.txt for easy setup.

---

## Key Takeaways

- **Data quality and thoughtful feature engineering are critical for accurate predictions.**
- **Model selection is based on robust cross-validation, not just test set performance.**
- **The Streamlit app is designed for both usability and transparency, with clear feedback and visualizations for users.**
- **The project is fully reproducible and ready for deployment.**

---