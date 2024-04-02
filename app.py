import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans

# Baca dataset
df = pd.read_csv('datacleaned.csv')

# Set tema
sns.set_theme(style="darkgrid")

# Judul aplikasi
st.title('Dashboard Analisis Media Sosial Berdasarkan Gender dan Waktu Yang Dihabiskan')

# Filter dataset di sidebar
st.sidebar.header('Filter Dataset')
age_min, age_max = st.sidebar.slider('Pilih rentang usia:', int(df['age'].min()), int(df['age'].max()), (20, 40))
selected_gender = st.sidebar.multiselect('Pilih gender:', options=df['gender'].unique(), default=df['gender'].unique())

# Filter data
filtered_data = df[(df['age'] >= age_min) & (df['age'] <= age_max) & (df['gender'].isin(selected_gender))]

# Informasi dataset
st.header('Informasi Dasar Dataset')
st.write(filtered_data)

# Visualisasi distribusi usia berdasarkan gender
st.header('Distribusi Usia Pengguna Berdasarkan Gender')
fig, ax = plt.subplots()
sns.histplot(data=filtered_data, x='age', hue='gender', kde=True, multiple="stack", palette='pastel')
ax.set_title('Distribusi Usia Pengguna Media Sosial Berdasarkan Gender')
ax.set_xlabel("Usia Pengguna")
ax.set_ylabel("Jumlah Pengguna")
st.pyplot(fig)

# Visualisasi distribusi jenis kelamin
st.header('Distribusi Jenis Kelamin Pengguna')
gender_count = filtered_data['gender'].value_counts()
fig2 = px.pie(gender_count, values=gender_count, names=gender_count.index, title='Distribusi Jenis Kelamin Dalam Filter')
st.plotly_chart(fig2)

# Preprocessing untuk clustering
st.header('Elbow Method for Optimal K in K-Means Clustering')
# Label encoding untuk kolom 'gender'
label_encoder = LabelEncoder()
df['gender_encoded'] = label_encoder.fit_transform(df['gender'])

# Menghilangkan kolom yang tidak diperlukan
x_final = df.drop(['gender', 'demographics', 'age_category'], axis=1)

# Scaling data
scaler = MinMaxScaler()
x_final_norm = scaler.fit_transform(x_final)

# Mencari nilai K optimal
inertia_values = []
k_range = range(2, 10)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(x_final_norm)
    inertia_values.append(kmeans.inertia_)

# Plotting Elbow Method
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia_values, marker='o')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.xticks(k_range)
st.pyplot(plt)
