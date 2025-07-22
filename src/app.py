import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- Caching and Model Loading ----
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
st.sidebar.markdown(
    '''
    <style>
    section[data-testid="stSidebar"] > div:first-child {padding-top: 0.1rem !important;}
    .sidebar-nav-label {font-size: 1.18rem !important; font-weight: 600; margin-bottom: 0 !important; color: #fff; letter-spacing: 0.5px;}
    .sidebar-radio .stRadio > div {font-size: 1.13rem !important;}
    .sidebar-radio .stRadio div[role="radiogroup"] > label {margin-bottom: 0.18rem !important;}
    .sidebar-radio .stRadio div[role="radiogroup"] {margin-top: 0 !important; padding-top: 0 !important;}
    .sidebar-tagline {margin-bottom: 1.2rem;}
    </style>
    <div style="background: #23272b; color: #fff; border-radius: 12px; padding: 0.3rem 0.5rem 0.5rem 0.5rem; margin-bottom: 0.5rem; box-shadow: 0 2px 8px #23272b;">
        <div style="text-align: center;">
            <span style="font-size:2.8rem;">🏠</span>
            <div style="font-size:1.5rem; font-weight: bold; margin-top: 0.2rem; color: #fff;">House Price Prediction</div>
            <div class="sidebar-tagline" style="font-size:1rem; color: #b0b8c1; margin-top: 0.2rem;">Smart, Fast & Easy</div>
        </div>
    </div>
    ''', unsafe_allow_html=True
)
st.sidebar.markdown('<div class="sidebar-nav-label">Navigation</div>', unsafe_allow_html=True)
section = st.sidebar.radio(
    '',
    ['🔮 Predict', '📦 Batch Predict', '📊 EDA', 'ℹ️ About'],
    key='sidebar-radio',
)
st.sidebar.markdown('<hr style="margin: 0.3rem 0 0.2rem 0;">', unsafe_allow_html=True)
st.sidebar.markdown(
    '''
    <div style="text-align:center; font-size:0.95rem; margin-top:1.5rem;">
        Made with <span style="color:#e25555;">❤️</span> by <a href="https://www.linkedin.com/in/shreya-singh-561a591a5/" target="_blank" style="color:#1a237e; text-decoration:underline;">Shreya Singh</a>
    </div>
    ''', unsafe_allow_html=True
)

footer_html = '''<hr style="border: none; border-top: 1px solid #bbb; margin: 1.2rem 0 0.5rem 0;" />
<div style="text-align:left; font-size:0.95rem; color:#e0e0e0; margin-bottom:0.5rem;">
Built with ❤️ by <a href="https://www.linkedin.com/in/shreya-singh-561a591a5/" target="_blank" style="color:#90caf9; text-decoration:underline;">Shreya Singh</a> for data science and real estate enthusiasts.
</div>'''

def get_cat_options(col_prefix):
    if train_df.empty:
        return []
    return sorted([c.replace(col_prefix+'_', '') for c in train_df.columns if c.startswith(col_prefix+'_')])

# --- Predict Section ---
if section == '🔮 Predict':
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

    st.markdown(footer_html, unsafe_allow_html=True)

# --- Batch Predict Section ---
elif section == '📦 Batch Predict':
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

    st.markdown(footer_html, unsafe_allow_html=True)

# --- EDA Section ---
elif section == '📊 EDA':
    st.title('📊 Exploratory Data Analysis')
    if train_err:
        st.error(f"Data loading error: {train_err}")
    elif train_df.empty:
        st.warning('Training data not available for EDA.')
    else:
        with st.expander('SalePrice Distribution', expanded=True):
            fig1, ax1 = plt.subplots()
            sns.histplot(train_df['SalePrice'], bins=30, kde=True, ax=ax1, color='lightgreen')
            ax1.set_xlabel('SalePrice')
            st.pyplot(fig1)
        # Feature selection for scatterplot
        with st.expander('Feature vs. SalePrice (Scatterplot)', expanded=False):
            num_cols = train_df.select_dtypes(include=np.number).columns.tolist()
            feature = st.selectbox('Select a feature to plot against SalePrice:', [col for col in num_cols if col != 'SalePrice'])
            fig2, ax2 = plt.subplots()
            sns.scatterplot(x=train_df[feature], y=train_df['SalePrice'], ax=ax2)
            ax2.set_xlabel(feature)
            ax2.set_ylabel('SalePrice')
            st.pyplot(fig2)
        # Categorical boxplot
        with st.expander('Categorical Feature Analysis (Boxplot)', expanded=False):
            cat_cols = train_df.select_dtypes(include='object').columns.tolist()
            if cat_cols:
                cat_feature = st.selectbox('Select a categorical feature:', cat_cols)
                fig3, ax3 = plt.subplots()
                sns.boxplot(x=train_df[cat_feature], y=train_df['SalePrice'], ax=ax3)
                ax3.set_xlabel(cat_feature)
                ax3.set_ylabel('SalePrice')
                plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
                st.pyplot(fig3)
            else:
                st.info('No categorical features available.')
        # Correlation heatmap
        with st.expander('Correlation Heatmap', expanded=False):
            corr_cols = st.multiselect('Select features for correlation heatmap:', num_cols, default=num_cols[:10])
            if corr_cols:
                corr = train_df[corr_cols + ['SalePrice']].corr()
                fig4, ax4 = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax4)
                st.pyplot(fig4)
            else:
                st.info('Select at least one feature to display the heatmap.')
        # Missing values
        with st.expander('Missing Values', expanded=False):
            missing = train_df.isnull().sum()
            missing = missing[missing > 0]
            if not missing.empty:
                fig5, ax5 = plt.subplots()
                missing.sort_values().plot(kind='barh', ax=ax5)
                ax5.set_title('Missing Values per Feature')
                st.pyplot(fig5)
            else:
                st.info('No missing values in the training data.')
        # Summary statistics
        with st.expander('Summary Statistics', expanded=False):
            st.dataframe(train_df.describe().T)

    st.markdown(footer_html, unsafe_allow_html=True)

# --- About Section ---
elif section == 'ℹ️ About':
    st.title('ℹ️ About This Project')
    st.markdown('''
    <div style="background: #e3eafc; color: #1a237e; border-radius: 10px; padding: 1.2rem 1.5rem; margin-bottom: 1rem;">
        <b>🏠 House Price Prediction App</b><br>
        Predict house prices instantly using real data and advanced machine learning. Explore, analyze, and download results—all in one place.
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('### 🚀 Key Features')
    st.markdown('''
    - 🔮 **Single Prediction:** Enter house details and get an instant price prediction.
    - 📦 **Batch Prediction:** Upload a CSV to predict prices for many houses at once.
    - 📊 **EDA:** Explore data distributions, feature importance, and more.
    - ⬇️ **Download:** Export your results for further analysis.
    ''')

    st.markdown('### 🛠️ Tech Stack')
    st.markdown('''
    - Python, pandas, numpy, scikit-learn, xgboost
    - Streamlit for the interactive web app
    - matplotlib & seaborn for visualizations
    ''')

    st.markdown('### 📋 How to Use')
    st.markdown('''
    1. **Predict:** Use the sidebar to enter house features and get a price.
    2. **Batch Predict:** Upload a CSV file with house data for bulk predictions.
    3. **EDA:** Explore the data and model insights.
    4. **Download:** Save your results as CSV or Excel.
    ''')

    st.markdown('### 💡 Tips for Best Results')
    st.markdown('''
    - Use realistic values for all features.
    - For batch prediction, ensure your CSV columns match the model features.
    - Explore EDA to understand what drives house prices!
    ''')

    with st.expander("👩‍💻 Note from the Developer"):
        st.markdown('''
        Thank you for using this app!  
        If you have feedback, suggestions, or want to connect, reach out on [LinkedIn](https://www.linkedin.com/in/shreya-singh-561a591a5/) – Shreya Singh
        ''')

    st.markdown(footer_html, unsafe_allow_html=True)