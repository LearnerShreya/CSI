import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Caching and Model Loading ---
@st.cache_data
def load_model():
    try:
        model = joblib.load('../models/model_house_price_prediction.pkl')
        model_columns = joblib.load('../models/model_columns.pkl')
        return model, model_columns, None
    except Exception as e:
        return None, None, str(e)

@st.cache_data
def load_train_data():
    try:
        df = pd.read_csv('../data/processed_train.csv')
        return df, None
    except Exception as e:
        return pd.DataFrame(), str(e)

model, model_columns, model_err = load_model()
train_df, train_err = load_train_data()

# --- Sidebar Navigation ---
st.sidebar.title('🏠 House Price Prediction')
section = st.sidebar.radio('Navigation', ['Predict', 'Batch Predict', 'EDA', 'About'])

def get_cat_options(col_prefix):
    if train_df.empty:
        return []
    return sorted([c.replace(col_prefix+'_', '') for c in train_df.columns if c.startswith(col_prefix+'_')])

# --- Predict Section ---
if section == 'Predict':
    st.title('🏡 Predict House Sale Price')
    if model_err:
        st.error(f"Model loading error: {model_err}")
    elif train_err:
        st.error(f"Data loading error: {train_err}")
    else:
        st.markdown('### Enter House Features')
feature_inputs = {}
feature_inputs['OverallQual'] = st.slider('Overall Quality (1-10)', 1, 10, 5)
feature_inputs['GrLivArea'] = st.number_input('Above Ground Living Area (sq ft)', min_value=300, max_value=6000, value=1500)
feature_inputs['GarageCars'] = st.slider('Garage Cars', 0, 4, 2)
feature_inputs['TotalBsmtSF'] = st.number_input('Total Basement SF', min_value=0, max_value=3000, value=800)
feature_inputs['1stFlrSF'] = st.number_input('1st Floor SF', min_value=0, max_value=3000, value=900)
feature_inputs['2ndFlrSF'] = st.number_input('2nd Floor SF', min_value=0, max_value=3000, value=500)
feature_inputs['YearBuilt'] = st.number_input('Year Built', min_value=1870, max_value=2023, value=2000)
feature_inputs['YearRemodAdd'] = st.number_input('Year Remodeled', min_value=1870, max_value=2023, value=2010)
feature_inputs['YrSold'] = st.number_input('Year Sold', min_value=2006, max_value=2010, value=2010)
mszoning_opts = get_cat_options('MSZoning')
feature_inputs['MSZoning'] = st.selectbox('MS Zoning', mszoning_opts) if mszoning_opts else 'RL'
neighborhood_opts = get_cat_options('Neighborhood')
feature_inputs['Neighborhood'] = st.selectbox('Neighborhood', neighborhood_opts) if neighborhood_opts else 'NAmes'
housestyle_opts = get_cat_options('HouseStyle')
feature_inputs['HouseStyle'] = st.selectbox('House Style', housestyle_opts) if housestyle_opts else '1Story'
feature_inputs['TotalSF'] = feature_inputs['TotalBsmtSF'] + feature_inputs['1stFlrSF'] + feature_inputs['2ndFlrSF']
feature_inputs['HouseAge'] = feature_inputs['YrSold'] - feature_inputs['YearBuilt']
feature_inputs['RemodAge'] = feature_inputs['YrSold'] - feature_inputs['YearRemodAdd']
if feature_inputs['YearBuilt'] > feature_inputs['YrSold']:
    st.warning('Year Built cannot be after Year Sold!')
if feature_inputs['YearRemodAdd'] > feature_inputs['YrSold']:
    st.warning('Year Remodeled cannot be after Year Sold!')
input_df = pd.DataFrame([feature_inputs])
for col_prefix in ['MSZoning', 'Neighborhood', 'HouseStyle']:
    col = feature_inputs[col_prefix]
    col_name = f'{col_prefix}_{col}'
    input_df[col_name] = 1
for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model_columns]
if st.button('Predict Sale Price'):
            try:
    prediction = model.predict(input_df)[0]
                # --- 1. Summary Card ---
                st.markdown('''
                <div style="display: flex; gap: 2rem; margin-bottom: 1rem; align-items: center;">
                  <div style="background: #e3eafc; color: #1a237e; padding: 1.2rem 2rem; border-radius: 12px; min-width: 200px; text-align: center; font-size: 1.5rem; font-weight: bold; box-shadow: 0 2px 8px #e3eafc;">
                    🏡<br>Predicted Price<br><span style='font-size:2rem;'>${:,.0f}</span>
                  </div>
                </div>
                '''.format(prediction), unsafe_allow_html=True)
                # --- 2. Column Selector for Input Summary ---
                st.markdown('#### 📝 Input Summary')
                input_cols = list(feature_inputs.keys())
                selected_input_cols = st.multiselect('Columns to show in input summary:', input_cols, default=input_cols)
                # --- 3. Styled Preview Table ---
                def highlight_predicted(val):
                    return 'background-color: #fffde7; color: #0d47a1; font-weight: bold;' if val == prediction else ''
                input_preview = pd.DataFrame([feature_inputs])[selected_input_cols]
                st.dataframe(input_preview.style.applymap(highlight_predicted, subset=[col for col in input_preview.columns if 'Price' in col or 'Sale' in col]), use_container_width=True)
                st.download_button('Download Prediction', pd.DataFrame({'SalePrice':[prediction]}).to_csv(index=False), 'prediction.csv')
                # --- 4. Visualize Prediction in Context ---
                if not train_df.empty and 'SalePrice' in train_df.columns:
                    st.markdown('#### 📊 Where does your prediction fall?')
                    fig, ax = plt.subplots()
                    sns.histplot(train_df['SalePrice'], bins=30, kde=True, ax=ax, color='skyblue')
                    ax.axvline(prediction, color='red', linestyle='--', label='Your Prediction')
                    ax.set_title('SalePrice Distribution (Train Data)')
                    ax.set_xlabel('SalePrice')
                    ax.legend()
                    st.pyplot(fig)
            except Exception as e:
                st.error(f'Prediction error: {e}')

# --- Batch Predict Section ---
elif section == 'Batch Predict':
    st.title('📦 Batch House Price Prediction')
    if model_err:
        st.error(f"Model loading error: {model_err}")
    elif train_err:
        st.error(f"Data loading error: {train_err}")
    else:
        st.markdown('Upload a CSV file with house features for batch prediction. The file should have columns matching the model features.')
        uploaded_file = st.file_uploader('Choose a CSV file', type='csv')
        if uploaded_file:
            try:
                batch_df = pd.read_csv(uploaded_file)
                if {'TotalBsmtSF','1stFlrSF','2ndFlrSF'}.issubset(batch_df.columns):
                    batch_df['TotalSF'] = batch_df['TotalBsmtSF'] + batch_df['1stFlrSF'] + batch_df['2ndFlrSF']
                if {'YrSold','YearBuilt'}.issubset(batch_df.columns):
                    batch_df['HouseAge'] = batch_df['YrSold'] - batch_df['YearBuilt']
                if {'YrSold','YearRemodAdd'}.issubset(batch_df.columns):
                    batch_df['RemodAge'] = batch_df['YrSold'] - batch_df['YearRemodAdd']
                for col_prefix in ['MSZoning', 'Neighborhood', 'HouseStyle']:
                    if col_prefix in batch_df.columns:
                        for val in batch_df[col_prefix].unique():
                            col_name = f'{col_prefix}_{val}'
                            batch_df[col_name] = (batch_df[col_prefix] == val).astype(int)
                for col in model_columns:
                    if col not in batch_df.columns:
                        batch_df[col] = 0
                batch_df = batch_df[model_columns]
                preds = model.predict(batch_df)
                batch_df['PredictedSalePrice'] = preds
            except Exception as e:
                st.error(f'Batch prediction error: {e}')
                batch_df = None
            if 'batch_df' in locals() and batch_df is not None:
                flagged_rows = batch_df.isnull().any(axis=1).sum()
                mean_pred = batch_df['PredictedSalePrice'].mean()
                median_pred = batch_df['PredictedSalePrice'].median()
                min_pred = batch_df['PredictedSalePrice'].min()
                max_pred = batch_df['PredictedSalePrice'].max()
                st.markdown('''
                <div style="display: flex; gap: 2rem; margin-bottom: 1rem;">
                  <div style="background: #e3eafc; color: #1a237e; padding: 1rem; border-radius: 8px; min-width: 120px; text-align: center;">
                    <b>Rows</b><br>{}</div>
                  <div style="background: #e3eafc; color: #1a237e; padding: 1rem; border-radius: 8px; min-width: 120px; text-align: center;">
                    <b>Mean</b><br>${:,.0f}</div>
                  <div style="background: #e3eafc; color: #1a237e; padding: 1rem; border-radius: 8px; min-width: 120px; text-align: center;">
                    <b>Median</b><br>${:,.0f}</div>
                  <div style="background: #e3eafc; color: #1a237e; padding: 1rem; border-radius: 8px; min-width: 120px; text-align: center;">
                    <b>Min</b><br>${:,.0f}</div>
                  <div style="background: #e3eafc; color: #1a237e; padding: 1rem; border-radius: 8px; min-width: 120px; text-align: center;">
                    <b>Max</b><br>${:,.0f}</div>
                  <div style="background: #fff3cd; color: #7c4700; padding: 1rem; border-radius: 8px; min-width: 120px; text-align: center;">
                    <b>Flagged Rows</b><br>{}</div>
                </div>
                '''.format(
                    batch_df.shape[0], mean_pred, median_pred, min_pred, max_pred, flagged_rows
                ), unsafe_allow_html=True)
                st.markdown('#### 📝 Batch Input Summary')
                with st.expander('See batch input details', expanded=True):
                    st.markdown(f"**Rows:** {batch_df.shape[0]}  |  **Columns:** {batch_df.shape[1]}  |  **Missing values:** {batch_df.isnull().sum().sum()}")
                    st.dataframe(batch_df.head(), use_container_width=True)
                    st.caption('Preview: First 5 rows of your uploaded data (after feature engineering).')
                # --- 3. Column Selector ---
                st.markdown('#### 🗂️ Select Columns to Display')
                engineered_cols = [col for col in ['TotalSF', 'HouseAge', 'RemodAge'] if col in batch_df.columns]
                try:
                    uploaded_file.seek(0)
                    user_cols = [col for col in pd.read_csv(uploaded_file, nrows=1).columns]
                except pd.errors.EmptyDataError:
                    st.error('Uploaded CSV is empty or invalid. Please upload a valid CSV file with headers.')
                    st.stop()
                if batch_df.empty:
                    st.error('Uploaded CSV contains no data.')
                    st.stop()
                all_cols = list(batch_df.columns)
                preview_cols_default = [col for col in (user_cols + engineered_cols + ['PredictedSalePrice', 'SalePrice']) if col in all_cols]
                selected_cols = st.multiselect('Columns to show in preview/download:', all_cols, default=preview_cols_default)
                st.markdown('#### 📋 Batch Prediction Results Preview')
                q_low = batch_df['PredictedSalePrice'].quantile(0.05)
                q_high = batch_df['PredictedSalePrice'].quantile(0.95)
                preview_df = batch_df[selected_cols].copy()
                def highlight_outlier(val):
                    style = ''
                    if isinstance(val, (int, float)):
                        if val <= q_low:
                            style += 'background-color: #ffe6e6;'
                        elif val >= q_high:
                            style += 'background-color: #e6f7ff;'
                    return style
                def highlight_predicted(val):
                    return 'background-color: #fffde7; color: #0d47a1; font-weight: bold;'
                styled_preview = preview_df.head(10).style.applymap(highlight_outlier, subset=['PredictedSalePrice'])
                styled_preview = styled_preview.applymap(highlight_predicted, subset=['PredictedSalePrice'])
                st.dataframe(styled_preview, use_container_width=True)
                st.markdown('#### 🏆 Top 5 Most Expensive | 💸 5 Cheapest Predictions')
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(batch_df.nlargest(5, 'PredictedSalePrice')[selected_cols].reset_index(drop=True), use_container_width=True)
                with col2:
                    st.dataframe(batch_df.nsmallest(5, 'PredictedSalePrice')[selected_cols].reset_index(drop=True), use_container_width=True)
                st.markdown('#### 📊 Prediction Distribution & Boxplot')
                fig, axs = plt.subplots(1, 2, figsize=(10, 4))
                axs[0].hist(batch_df['PredictedSalePrice'], bins=30, color='deepskyblue', edgecolor='black')
                axs[0].set_title('Histogram')
                axs[0].set_xlabel('Predicted SalePrice')
                axs[0].set_ylabel('Frequency')
                axs[1].boxplot(batch_df['PredictedSalePrice'], vert=False, patch_artist=True, boxprops=dict(facecolor='#e6f7ff'))
                axs[1].set_title('Boxplot')
                axs[1].set_xlabel('Predicted SalePrice')
                st.pyplot(fig)
                if flagged_rows > 0:
                    st.warning(f"{flagged_rows} row(s) had missing or invalid values. These may affect prediction accuracy.")
                    st.dataframe(batch_df[batch_df.isnull().any(axis=1)][selected_cols].head(10), use_container_width=True)
                    st.download_button('Download Flagged Rows', batch_df[batch_df.isnull().any(axis=1)][selected_cols].to_csv(index=False), 'flagged_rows.csv')
                st.markdown('#### ⬇️ Download Results')
                st.download_button('Download Selected Columns (CSV)', batch_df[selected_cols].to_csv(index=False), 'batch_selected_columns.csv')
                try:
                    import io
                    import xlsxwriter
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        batch_df[selected_cols].to_excel(writer, index=False, sheet_name='Predictions')
                    st.download_button('Download Selected Columns (Excel)', output.getvalue(), 'batch_selected_columns.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                except Exception:
                    pass

# --- EDA Section ---
elif section == 'EDA':
    st.title('📊 Exploratory Data Analysis')
    if train_err:
        st.error(f"Data loading error: {train_err}")
    elif train_df.empty:
        st.warning('Training data not available for EDA.')
    else:
        st.markdown('### SalePrice Distribution')
        fig1, ax1 = plt.subplots()
        sns.histplot(train_df['SalePrice'], bins=30, kde=True, ax=ax1, color='lightgreen')
        ax1.set_xlabel('SalePrice')
        st.pyplot(fig1)
        st.markdown('### Feature Importance')
        if model is not None:
            if hasattr(model, 'coef_'):
                importances = pd.Series(model.coef_, index=model_columns)
                top_importances = importances.abs().sort_values(ascending=False).head(10)
                fig2, ax2 = plt.subplots()
                top_importances.plot(kind='barh', ax=ax2)
                ax2.set_title('Top 10 Feature Importances (abs)')
                st.pyplot(fig2)
            elif hasattr(model, 'feature_importances_'):
                importances = pd.Series(model.feature_importances_, index=model_columns)
                top_importances = importances.abs().sort_values(ascending=False).head(10)
                fig2, ax2 = plt.subplots()
                top_importances.plot(kind='barh', ax=ax2)
                ax2.set_title('Top 10 Feature Importances')
                st.pyplot(fig2)
            else:
                st.info('Feature importance not available for this model.')
        st.markdown('### Correlation Heatmap')
        corr = train_df.corr()
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr[['SalePrice']].sort_values(by='SalePrice', ascending=False).head(11), annot=True, cmap='coolwarm', ax=ax3)
        st.pyplot(fig3)

# --- About Section ---
elif section == 'About':
    st.title('ℹ️ About This Project')
    st.markdown('''
    ## Welcome to the House Price Prediction App!
    
    This app empowers you to:
    - Instantly predict house prices using real, historical data and advanced machine learning.
    - Upload your own CSV files for batch predictions—get results for many houses at once.
    - Explore the data and model insights with interactive EDA and feature importance tools.
    
    **How to use:**
    - Use the **Predict** tab for single house price prediction.
    - Use **Batch Predict** to upload a CSV and get predictions for many houses.
    - Explore **EDA** for data insights and model interpretability.
    
    ---
    
    **Note from the Developer:**
    > Thank you for using this app! If you have feedback, suggestions, or want to connect, feel free to reach out on [LinkedIn](https://www.linkedin.com/in/shreya-singh-561a591a5/) – Shreya Singh
    
    ---
    ''')

st.markdown('---')
st.caption('For best results, use realistic values for your house features.')