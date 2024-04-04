import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans

# Fungsi untuk membaca dataset
def load_data():
    return pd.read_csv('datacleaned.csv')

# Fungsi untuk menampilkan judul aplikasi
def show_title():
    st.title('Dashboard Analisis Penggunaan Media Sosial')

# Fungsi untuk filter dataset di sidebar
def sidebar_filter(df):
    st.sidebar.header('Filter Dataset')
    age_min, age_max = st.sidebar.slider('Pilih rentang usia:', int(df['age'].min()), int(df['age'].max()), (20, 40))
    selected_gender = st.sidebar.multiselect('Pilih gender:', options=df['gender'].unique(), default=df['gender'].unique())
    return df[(df['age'] >= age_min) & (df['age'] <= age_max) & (df['gender'].isin(selected_gender))]

# Fungsi untuk menampilkan informasi dasar dataset
def show_basic_info(data):
    st.header('Informasi Dasar Dataset')
    st.write(data)

# Fungsi untuk visualisasi distribusi usia berdasarkan gender
def visualize_age_distribution(data):
    st.header('Distribusi Usia Pengguna Berdasarkan Gender')
    fig, ax = plt.subplots()
    sns.histplot(data=data, x='age', hue='gender', kde=True, multiple="stack", palette='pastel')
    ax.set_title('Distribusi Usia Pengguna Media Sosial Berdasarkan Gender')
    ax.set_xlabel("Usia Pengguna")
    ax.set_ylabel("Jumlah Pengguna")
    st.pyplot(fig)

# Fungsi untuk visualisasi tren waktu penggunaan media sosial
def visualize_time_trend(data):
    st.header('Tren Waktu Penggunaan Media Sosial')
    fig, ax = plt.subplots()
    # Implementasi visualisasi tren di sini
    st.pyplot(fig)

# Fungsi untuk visualisasi korelasi antara variabel
def visualize_correlation(data):
    st.header('Korelasi Antara Variabel')
    fig, ax = plt.subplots()
    sns.histplot(data["age"], kde=True)
    plt.title("Usia Pengguna Sosial Media")
    plt.xlabel("Usia Pengguna")
    plt.ylabel("Jumlah Pengguna")
    st.pyplot(fig)

# Fungsi untuk melakukan preprocessing untuk clustering
def preprocess_for_clustering(data):
    x_final = data.drop(['demographics', 'age_category'], axis=1)
    label_encoder = LabelEncoder()
    x_final['gender_encoded'] = label_encoder.fit_transform(x_final['gender'])
    numeric_columns = x_final.select_dtypes(include=['int', 'float']).columns
    scaler = MinMaxScaler()
    x_final_norm = scaler.fit_transform(x_final[numeric_columns])
    return x_final_norm

# Fungsi untuk melakukan clustering dan menampilkan hasil
def perform_clustering(data, x_final_norm):
    st.header('K-Means Clustering')
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(x_final_norm)
    kmeans_clust = kmeans.predict(x_final_norm)

    x_final = pd.DataFrame(data.drop(['demographics', 'age_category'], axis=1)).reset_index(drop=True)
    kmeans_col = pd.DataFrame(kmeans_clust, columns=["kmeans_cluster"])
    combined_data_assoc = pd.concat([x_final, kmeans_col], axis=1)

    # Custom visualization of clustering
    fig, ax = plt.subplots()
    colors = ['red', 'green', 'blue']  # Define colors for each cluster
    for i, cluster in enumerate(combined_data_assoc['kmeans_cluster'].unique()):
        clustered_data = combined_data_assoc[combined_data_assoc['kmeans_cluster'] == cluster]
        ax.scatter(clustered_data['time_spent'], clustered_data['gender'], label=f'Cluster {cluster}', color=colors[i], alpha=0.6)
    ax.set_title('Custom Scatter K-Means')
    ax.set_xlabel('Time Spent')
    ax.set_ylabel('Gender')
    ax.legend()
    st.pyplot(fig)

def main():
    show_title()
    df = load_data()
    filtered_data = sidebar_filter(df)
    show_basic_info(filtered_data)
    visualize_age_distribution(filtered_data)
    visualize_time_trend(filtered_data)
    visualize_correlation(filtered_data)
    x_final_norm = preprocess_for_clustering(filtered_data)
    perform_clustering(filtered_data, x_final_norm)

if __name__ == "__main__":
    main()
