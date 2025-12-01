import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import kagglehub
import os

st.title("Cardiovascular Disease EDA Dashboard")
st.markdown("---")


st.sidebar.header("Data")
file = st.sidebar.file_uploader("Upload dataset CSV (optional)", type=["csv"])

@st.cache_data
def load_data(file=None):
    if file is not None:
        df = pd.read_csv(file)
        return df
    else:
        try:
            path = kagglehub.dataset_download("alphiree/cardiovascular-diseases-risk-prediction-dataset")
            df = pd.read_csv(os.path.join(path, "CVD_cleaned.csv"))
            return df
        except Exception as e:
            st.error(f"Could not download dataset from Kaggle: {e}")
            st.info("Please upload the CVD_cleaned.csv file manually.")
            return None


df = load_data(file)

if df is None:
    st.error("""
    ### No dataset found!
    """)
    st.stop()

st.markdown("### Dataset Preview")
st.dataframe(df.head())


st.markdown("### Basic Dataset Info")
col1, col2 = st.columns(2)
with col1:
    st.write("Rows:", df.shape[0])
with col2:
    st.write("Columns:", df.shape[1])


@st.cache_data
def preprocess(df):
    df = df.copy()
    df.drop_duplicates(inplace=True)

    bin_map = {"Yes":1, "No":0, "No ":0, "NO":0, "YES":1}
    binary_cols = [
        "Exercise","Heart_Disease","Skin_Cancer","Other_Cancer","Depression","Diabetes",
        "Arthritis","Smoking_History"
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(bin_map).fillna(0)

    if 'Sex' in df.columns:
        df["Sex"] = df["Sex"].map({"Female":0, "Male":1})

    age_order = {
        "18-24":1,"25-29":2,"30-34":3,"35-39":4,"40-44":5,
        "45-49":6,"50-54":7,"55-59":8,"60-64":9,"65-69":10,"70-74":11,"75-79":12,"80+":13
    }
    if 'Age_Category' in df.columns:
        df["Age_Category"] = df["Age_Category"].map(age_order)

    checkup_map = {
        "Within the past year":1,
        "Within the past 2 years":2,
        "Within the past 5 years":3,
        "5 or more years ago":4,
        "Never":5
    }
    if 'Checkup' in df.columns:
        df["Checkup"] = df["Checkup"].map(checkup_map)

    general_health_map = {
        "Excellent": 5,
        "Very Good": 4,
        "Good": 3,
        "Fair": 2,
        "Poor": 1
    }
    if 'General_Health' in df.columns:
        df["General_Health"] = df["General_Health"].map(general_health_map)

    return df

df_encoded = preprocess(df)


st.sidebar.header("Filters")
sex_filter = st.sidebar.selectbox("Sex", options=['All'] + df["Sex"].dropna().unique().tolist()) if 'Sex' in df.columns else 'All'
age_min, age_max = None, None
if 'Age_Category' in df.columns:
    age_vals = sorted(df_encoded['Age_Category'].dropna().unique())
    age_min, age_max = st.sidebar.select_slider("Age category range", options=age_vals, value=(age_vals[0], age_vals[-1]))


df_filtered = df_encoded.copy()
if sex_filter != 'All' and 'Sex' in df_filtered.columns:
    if sex_filter in [0, 1]:
        df_filtered = df_filtered[df_filtered['Sex'] == sex_filter]
    else:
        df_filtered = df_filtered[df['Sex'] == sex_filter]

if 'Age_Category' in df_filtered.columns and age_min is not None and age_max is not None:
    df_filtered = df_filtered[(df_filtered['Age_Category'] >= age_min) & (df_filtered['Age_Category'] <= age_max)]


tabs = st.tabs(["Target Distribution","Demographics","BMI & Lifestyle","Comorbidities","Correlation","Modeling","PCA & Feature Importance"])


with tabs[0]:
    st.header("Target Distribution")
    if 'Heart_Disease' in df_filtered.columns:
        fig, ax = plt.subplots()
        df_filtered['Heart_Disease'].value_counts().plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
        ax.set_title('Heart Disease Distribution')
        ax.set_xlabel('Heart Disease (0 = No, 1 = Yes)')
        ax.set_ylabel('Count')
        st.pyplot(fig)

        
        fig2, ax2 = plt.subplots()
        df_filtered['Heart_Disease'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax2)
        ax2.set_ylabel('')
        ax2.set_title('Target Distribution (Pie)')
        st.pyplot(fig2)
    else:
        st.warning("`Heart_Disease` column not present in dataset.")


with tabs[1]:
    st.header("Demographics")
    if 'Sex' in df.columns:
        fig, ax = plt.subplots()
        sns.countplot(x='Sex', hue='Heart_Disease', data=df, ax=ax)
        ax.set_title('Heart Disease by Sex')
        st.pyplot(fig)

    if 'Age_Category' in df.columns:
        fig, ax = plt.subplots(figsize=(10,4))
        sns.countplot(x='Age_Category', hue='Heart_Disease', data=df_encoded, ax=ax)
        ax.set_title('Heart Disease by Age Category')
        st.pyplot(fig)


with tabs[2]:
    st.header("BMI & Lifestyle")
    if 'BMI' in df.columns:
        fig, ax = plt.subplots()
        sns.histplot(df['BMI'], kde=True, ax=ax)
        ax.set_title('BMI Distribution')
        st.pyplot(fig)

        fig2, ax2 = plt.subplots()
        sns.boxplot(data=df, x='Heart_Disease', y='BMI', ax=ax2)
        ax2.set_title('BMI vs Heart Disease')
        st.pyplot(fig2)

    st.subheader('Lifestyle Factors')
    lifestyle_cols = ["Exercise","Smoking_History","Alcohol_Consumption","Fruit_Consumption","Green_Vegetables_Consumption","FriedPotato_Consumption"]
    for col in lifestyle_cols:
        if col in df.columns:
            fig, ax = plt.subplots()
            if df[col].dtype == object:
                sns.countplot(data=df, x=col, hue='Heart_Disease', ax=ax)
                ax.set_title(f'Heart Disease vs {col}')
            else:
                sns.boxplot(data=df, x='Heart_Disease', y=col, ax=ax)
                ax.set_title(f'{col} vs Heart Disease')
            st.pyplot(fig)


with tabs[3]:
    st.header('Comorbidities')
    comorbidities = ["Diabetes", "Arthritis", "Depression", "Skin_Cancer", "Other_Cancer"]
    for col in comorbidities:
        if col in df.columns:
            fig, ax = plt.subplots()
            sns.countplot(data=df, x=col, hue='Heart_Disease', ax=ax)
            ax.set_title(f'Heart Disease vs {col}')
            st.pyplot(fig)


with tabs[4]:
    st.header('Correlation Heatmap')
    numeric_df = df_encoded.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(12,6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)


with tabs[5]:
    st.header('Modeling')
    st.markdown('This section trains a Logistic Regression model and a Random Forest model using SMOTE-resampled training data.\nSelect options and click `Train Models` to run.')

    test_size = st.slider('Test size (%)', min_value=10, max_value=50, value=20, step=5)
    run_train = st.button('Train Models')

    if run_train:
        if 'Heart_Disease' not in df_encoded.columns:
            st.error('No target column `Heart_Disease` found.')
        else:
            X = df_encoded.drop('Heart_Disease', axis=1)
            y = df_encoded['Heart_Disease']

            if len(y.unique()) < 2:
                st.error(f'Cannot train model: Only one class present in dataset (class: {y.unique()[0]}). The dataset needs both disease and non-disease cases.')
                st.stop()

            X = pd.get_dummies(X, drop_first=True)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=42, stratify=y)

            sm = SMOTE(random_state=42)
            X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

            logreg = LogisticRegression(max_iter=1000)
            logreg.fit(X_train_res, y_train_res)
            y_pred = logreg.predict(X_test)

            st.subheader('Logistic Regression Results')
            st.write('Accuracy:', accuracy_score(y_test, y_pred))
            st.text(classification_report(y_test, y_pred))

            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix — Logistic Regression')
            st.pyplot(fig)

            rf = RandomForestClassifier(n_estimators=200, random_state=42)
            rf.fit(X_train_res, y_train_res)
            y_pred_rf = rf.predict(X_test)

            st.subheader('Random Forest Results')
            st.write('Accuracy:', accuracy_score(y_test, y_pred_rf))
            st.text(classification_report(y_test, y_pred_rf))


            importances = pd.Series(rf.feature_importances_, index=X.columns)
            st.subheader('Top 15 Important Features (Random Forest)')
            fig, ax = plt.subplots(figsize=(10,6))
            importances.sort_values(ascending=False).head(15).plot(kind='bar', ax=ax)
            ax.set_title('Feature Importances')
            st.pyplot(fig)


with tabs[6]:
    st.header('PCA Visualization')
    numeric_df = df_encoded.select_dtypes(include=[np.number])
    if 'Heart_Disease' in numeric_df.columns:
        X_num = numeric_df.drop('Heart_Disease', axis=1).fillna(0)
        y = numeric_df['Heart_Disease']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_num)
        pca = PCA(n_components=2)
        pca_2d = pca.fit_transform(X_scaled)

        fig, ax = plt.subplots()
        scatter = ax.scatter(pca_2d[:,0], pca_2d[:,1], c=y, cmap='coolwarm', alpha=0.6)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('PCA 2D')
        plt.colorbar(scatter, ax=ax, label='Heart Disease')
        st.pyplot(fig)

    st.success('PCA visualization complete')

st.markdown('---')
st.write('Dashboard created from the notebook EDA in `EDA_Cardio.ipynb`.')
