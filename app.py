import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from scipy.cluster.hierarchy import dendrogram, linkage

st.set_page_config(page_title="EV Analytics App", layout="wide")

st.title("🚗⚡ Electric Vehicle Analytics Dashboard")

st.write("This app analyzes Electric Vehicle population data using Machine Learning.")

# ---------------- Upload Data ----------------
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    df = df.dropna(subset=["Electric Range", "Model Year", "Electric Vehicle Type"])
    df = df[df["Electric Range"] > 0]

    # ---------------- EDA ----------------
    st.header("Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("EV Adoption Over Years")
        fig, ax = plt.subplots()
        df["Model Year"].value_counts().sort_index().plot(kind="line", ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("Top 10 EV Manufacturers")
        fig, ax = plt.subplots()
        df["Make"].value_counts().head(10).plot(kind="bar", ax=ax)
        st.pyplot(fig)

    st.subheader("Electric Range Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["Electric Range"], bins=30, kde=True, ax=ax)
    st.pyplot(fig)

    # ---------------- SUPERVISED LEARNING ----------------
    st.header("Supervised Learning")

    st.subheader("1️⃣ Logistic Regression (EV Type Classification)")

    ml_df = df[["Model Year", "Electric Range", "Base MSRP", "Electric Vehicle Type"]].dropna()
    le = LabelEncoder()
    ml_df["Electric Vehicle Type"] = le.fit_transform(ml_df["Electric Vehicle Type"])

    X = ml_df.drop("Electric Vehicle Type", axis=1)
    y = ml_df["Electric Vehicle Type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    log_model = LogisticRegression()
    log_model.fit(X_train, y_train)

    y_pred = log_model.predict(X_test)

    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.subheader("2️⃣ Linear Regression (Electric Range Prediction)")

    X_lr = df[["Model Year", "Base MSRP"]].dropna()
    y_lr = df.loc[X_lr.index, "Electric Range"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_lr, y_lr, test_size=0.2, random_state=42
    )

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)

    y_pred = lin_model.predict(X_test)

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("Actual Range")
    ax.set_ylabel("Predicted Range")
    st.pyplot(fig)

    # ---------------- UNSUPERVISED LEARNING ----------------
    st.header("Unsupervised Learning")

    st.subheader("3️⃣ K-Means Clustering")

    cluster_data = df[["Model Year", "Electric Range"]].dropna()
    cluster_scaled = StandardScaler().fit_transform(cluster_data)

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(cluster_scaled)

    fig, ax = plt.subplots()
    ax.scatter(cluster_scaled[:, 0], cluster_scaled[:, 1], c=clusters)
    ax.set_xlabel("Model Year")
    ax.set_ylabel("Electric Range")
    st.pyplot(fig)

    st.subheader("4️⃣ Hierarchical Clustering")

    linked = linkage(cluster_scaled[:200], method="ward")

    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(linked, ax=ax)
    st.pyplot(fig)

else:
    st.info("Please upload the Electric Vehicle CSV file to continue.")
