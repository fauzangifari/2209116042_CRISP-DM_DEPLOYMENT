import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans
from streamlit_option_menu import option_menu


def load_data():
    return pd.read_csv('datacleaned.csv')


def show_title():
    st.title('Dashboard Analisis Pengguna Media Sosial')


def sidebar_filter(df):
    st.sidebar.header('Filter Dataset')
    age_min, age_max = st.sidebar.slider('Pilih rentang usia:', int(df['age'].min()), int(df['age'].max()), (20, 40))
    selected_gender = st.sidebar.multiselect('Pilih gender:', options=df['gender'].unique(),
                                             default=df['gender'].unique())

    user_count = st.sidebar.slider('Pilih jumlah data pengguna:', 1, len(df), 50)

    filtered_df = df[(df['age'] >= age_min) & (df['age'] <= age_max) & (df['gender'].isin(selected_gender))]

    if len(filtered_df) > user_count:
        filtered_df = filtered_df.sample(user_count)
    return filtered_df


def show_basic_info(data):
    st.header('Informasi Dasar Dataset')
    st.write(data)


def visualize_age_distribution(data):
    st.header('Distribusi Usia Pengguna Berdasarkan Gender')
    fig, ax = plt.subplots()
    sns.histplot(data=data, x='age', hue='gender', kde=True, multiple="stack", palette='pastel')
    ax.set_title('Distribusi Usia Pengguna Media Sosial Berdasarkan Gender')
    ax.set_xlabel("Usia Pengguna")
    ax.set_ylabel("Jumlah Pengguna")
    st.pyplot(fig)

    male_dominant_age = data[data['gender'] == 'male']['age'].value_counts().idxmax()
    female_dominant_age = data[data['gender'] == 'female']['age'].value_counts().idxmax()
    st.header("Kesimpulan")
    st.markdown(
        f"Usia dominan untuk pengguna pria adalah {male_dominant_age} tahun dan untuk pengguna wanita adalah {female_dominant_age} tahun. Strategi pemasaran dapat ditargetkan secara spesifik berdasarkan informasi ini.")


def visualize_age_composition(data):
    st.header('Usia Pengguna Sosial Media')
    fig, ax = plt.subplots()
    sns.histplot(data["age"], kde=True)
    plt.title("Usia Pengguna Sosial Media")
    plt.xlabel("Usia Pengguna")
    plt.ylabel("Jumlah Pengguna")
    st.pyplot(fig)

    age_dominant = data["age"].value_counts().idxmax()
    st.header("Kesimpulan")
    st.markdown(
        f"Usia yang paling dominan adalah {age_dominant} tahun. strategi pemasaran dapat ditargetkan secara spesifik berdasarkan informasi ini.")


def preprocess_for_clustering(data):
    x_final = data.drop(['demographics', 'age_category'], axis=1)
    label_encoder = LabelEncoder()
    x_final['gender_encoded'] = label_encoder.fit_transform(x_final['gender'])
    numeric_columns = x_final.select_dtypes(include=['int', 'float']).columns
    scaler = MinMaxScaler()
    x_final_norm = scaler.fit_transform(x_final[numeric_columns])
    return x_final_norm


def perform_clustering(data, x_final_norm):
    st.header('K-Means Clustering')
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(x_final_norm)
    kmeans_clust = kmeans.predict(x_final_norm)

    x_final = pd.DataFrame(data.drop(['demographics', 'age_category'], axis=1)).reset_index(drop=True)
    kmeans_col = pd.DataFrame(kmeans_clust, columns=["kmeans_cluster"])
    combined_data_assoc = pd.concat([x_final, kmeans_col], axis=1)

    fig, ax = plt.subplots()
    colors = ['red', 'green', 'blue']
    for i, cluster in enumerate(combined_data_assoc['kmeans_cluster'].unique()):
        clustered_data = combined_data_assoc[combined_data_assoc['kmeans_cluster'] == cluster]
        ax.scatter(clustered_data['time_spent'], clustered_data['gender'], label=f'Cluster {cluster}', color=colors[i],
                   alpha=0.6)
    ax.set_title('Scatter K-Means')
    ax.set_xlabel('Time Spent')
    ax.set_ylabel('Gender Encoded')
    ax.legend()
    st.pyplot(fig)

    insight = combined_data_assoc['kmeans_cluster'].value_counts()
    st.header("Hasil Cluster")
    st.write(insight)

    st.header("Kesimpulan")
    cluster_descriptions = []
    for cluster_num in range(3):
        cluster_data = combined_data_assoc[combined_data_assoc['kmeans_cluster'] == cluster_num]
        avg_time_spent = cluster_data['time_spent'].mean()
        gender_distribution = cluster_data['gender'].value_counts(normalize=True)

        # Mengonversi distribusi gender menjadi dictionary dengan pembulatan 3 angka belakang
        gender_distribution_dict = {gender: round(freq, 3) for gender, freq in gender_distribution.items()}

        cluster_descriptions.append(
            f"Cluster {cluster_num}: Rata-rata waktu yang dihabiskan adalah {avg_time_spent:.2f} jam dengan distribusi gender {gender_distribution_dict}.")

    for desc in cluster_descriptions:
        st.write(desc)

    st.write("""
    Berdasarkan clustering yang dilakukan, teridentifikasi 3 segmen pengguna dengan karakteristik waktu yang dihabiskan dan distribusi gender yang berbeda. Ini menunjukkan bahwa terdapat pola penggunaan media sosial yang variatif di antara pengguna. Perusahaan dapat memanfaatkan insight ini untuk menargetkan konten atau iklan mereka secara lebih efektif kepada masing-masing segmen.
    """)


def visualize_line_chart(data):
    st.header('Waktu yang dihabiskan dengan usia')
    fig, ax = plt.subplots()
    sns.set_style("whitegrid")
    sns.lineplot(data=data, x="time_spent", y="age")
    ax.set_title('Waktu yang dihabiskan dengan usia')
    ax.set_xlabel("Waktu yang dihabiskan")
    ax.set_ylabel("Usia")
    st.pyplot(fig)

    correlation = data['time_spent'].corr(data['age'])
    st.header("Kesimpulan")
    st.markdown(
        f"Terdapat korelasi sebesar {correlation:.2f} antara waktu yang dihabiskan dan usia pengguna. Ini menunjukkan bahwa {('seiring bertambahnya usia, waktu yang dihabiskan meningkat' if correlation > 0 else 'ada kecenderungan waktu yang dihabiskan berkurang seiring dengan bertambahnya usia')}.")


def visualize_relationship(data):
    st.header('Hubungan Waktu yang Dihabiskan di Media Sosial dengan Usia')
    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x='age', y='time_spent', palette='coolwarm', alpha=0.6)
    plt.title('Hubungan Waktu yang Dihabiskan dengan Usia')
    plt.xlabel('Usia')
    plt.ylabel('Waktu yang Dihabiskan di Media Sosial')
    st.pyplot(fig)

    age_relation_value = data['age'].corr(data['time_spent'])
    age_relation = 'positif' if age_relation_value > 0 else 'negatif'

    st.header("Kesimpulan")
    st.markdown(
        f"Hubungan antara usia dan waktu yang dihabiskan di media sosial adalah {age_relation}, "
        f"dengan nilai korelasi sebesar {age_relation_value:.2f}."
    )
def main():
    show_title()
    df = load_data()
    st.sidebar.title("Selamat Datang!")
    with st.sidebar:
        page = option_menu("Pilih Halaman",
                           ["Informasi Dasar", "Distribusi", "Hubungan", "Perbandingan", "Komposisi", "Clustering"])
    if page == "Informasi Dasar":
        show_basic_info(df)
    elif page == "Distribusi":
        filtered_data = sidebar_filter(df)
        visualize_age_distribution(filtered_data)
    elif page == "Hubungan":
        filtered_data = sidebar_filter(df)
        visualize_relationship(filtered_data)
    elif page == "Perbandingan":
        filtered_data = sidebar_filter(df)
        visualize_line_chart(filtered_data)
    elif page == "Komposisi":
        filtered_data = sidebar_filter(df)
        visualize_age_composition(filtered_data)
    elif page == "Clustering":
        filtered_data = sidebar_filter(df)
        x_final_norm = preprocess_for_clustering(filtered_data)
        perform_clustering(filtered_data, x_final_norm)


if __name__ == "__main__":
    main()
