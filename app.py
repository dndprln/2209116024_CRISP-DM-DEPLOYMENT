import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib 
import pickle
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Set option to disable the warning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Sidebar
selected_category = st.sidebar.selectbox("Halaman", 
    ["Dashboard", "Distribusi Data", "Analisis Perbandingan", "Analisis Hubungan",
    "Analisis Komposisi", "Data Clustering"
])

# Data Handling (assuming data is in a CSV file)
URL = 'reviewKPK.csv'
df = pd.read_csv(URL)

# Encode categorical columns
df_encoded = pd.get_dummies(df, columns=['brand_name', 'is_a_buyer'])

# Main Content
if selected_category == "Dashboard":
    # Dashboard Section
    st.title("Review Kosmetik dan Produk Skincare - Top Brand")

    st.image('https://qph.cf2.quoracdn.net/main-qimg-97c70a0bd0fed90eaa5ce532162b705d-lq', use_column_width=True)

    st.write(df)  # Menampilkan seluruh data pada halaman "Dashboard"

    st.write("Peninjauan menyeluruh terhadap kosmetik dan produk kecantikan dari merek-merek terkemuka yaitu seperti seperti Olay, Nivea, NYX Professional Makeup, Maybelline New York, Lakme, L'Oreal Paris, dll dengan memahami review, preferensi, dan rating pembeli dari merek-merek tersebut, tujuan utamanya adalah untuk memberikan wawasan yang berguna kepada konsumen dalam memilih produk kecantikan yang sesuai dengan kebutuhan dan preferensi mereka.")

elif selected_category == "Distribusi Data":
    # Distribusi Data Section
    st.title("Review Kosmetik dan Produk Skincare - Top Brand")

    # Display overall rating distribution
    st.write("**Distribusi Rating Produk**")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Ganti "product_rating" dengan nama kolom rating yang sesuai
    sns.countplot(x='product_rating', data=df)
    plt.title('Distribusi Rating Produk')
    plt.xlabel('Rating Produk')
    plt.ylabel('Jumlah')
    st.pyplot(fig)

    st.write("Dapat dilihat dari visualisasi diatas, bahwa product_rating tertinggi berada di nilai 4.2 dan product_rating terendah berada di nilai 3.3. Dapat di asumsikan bahwa nilai pada kolom review_rating tidak berpengaruh untuk nilai kolom product_rating, karena seberapa pun nilai yang diberikan di review_rating, maka hasil nilai dari product_rating tidak berubah")

elif selected_category == "Analisis Perbandingan":
    # Analisis Kategori Section
    st.title("Review Kosmetik dan Produk Skincare - Top Brand")

    # Menampilkan regression plot antara 'mrp' dan 'price'
    plt.figure(figsize=(10, 6))
    sns.regplot(x='mrp', y='price', data=df)
    plt.title('Regression Plot antara MRP dan Price')
    plt.xlabel('MRP')
    plt.ylabel('Price')
    st.pyplot()

    st.write("Gambar di atas menunjukkan hubungan antara harga maksimum ritel (MRP) dan harga (Price) produk. Regresi plot ini dapat memberikan gambaran tentang seberapa kuat hubungan linier antara kedua variabel tersebut.")

elif selected_category == "Analisis Hubungan":
    # Analisis Hubungan Section
    st.title("Review Kosmetik dan Produk Skincare - Top Brand")

    # Tambahkan kode heatmap korelasi
    df_numerik = df.select_dtypes(include=['number'])
    matriks_korelasi = df_numerik.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(matriks_korelasi, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matrix Plot dari Korelasi Antar Kolom')
    st.pyplot()

    # Konten analisis hubungan
    st.write("Heatmap di atas adalah visualisasi dari matriks korelasi antar kolom data numerik pada dataset kita. \
    Nilai-nilai dalam heatmap ini menunjukkan seberapa kuat hubungan linier antara pasangan kolom numerik. \
    Korelasi yang mendekati 1 menunjukkan hubungan positif yang kuat, sementara korelasi mendekati -1 menunjukkan \
    hubungan negatif yang kuat. Nilai mendekati 0 menunjukkan hubungan yang lemah atau tidak ada hubungan linier.")


elif selected_category == "Analisis Komposisi":
    # Analisis Komposisi Section
    st.title("Review Kosmetik dan Produk Skincare - Top Brand")

    # Top Merek dengan Rating Tinggi
    high_rating_brands = df[df['product_rating'] >= 4.5]['brand_name'].value_counts().head(5)
    st.write("Merek dengan Rating Tinggi :")
    st.write(high_rating_brands)

    st.write("Analisis Jumlah Produk per Merek")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='brand_name', data=df, order=df['brand_name'].value_counts().index)
    plt.title('Jumlah Produk per Merek')
    plt.xlabel('Merek')
    plt.ylabel('Jumlah Produk')
    plt.xticks(rotation=45)
    
    # Simpan plot ke dalam file PNG
    plt.savefig('plot.png')
    st.image('plot.png')  # Tampilkan plot dari file PNG di Streamlit

    st.write("Terdapat identifikasi dan menampilkan top merek yang memiliki rating tinggi, yaitu rating produknya lebih besar atau sama dengan 4.5")

elif selected_category == "Data Clustering":
    # Data Clustering Section
    st.title("Review Kosmetik dan Produk Skincare - Top Brand")
    st.subheader('Clustering Analysis based on Selected Features')

    # Selecting features for clustering
    selected_features = ['price', 'mrp']
    clustering_data = df[selected_features]

    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)

    # Selecting number of clusters with slider
    num_clusters = st.slider("Select number of clusters (2-8):", min_value=2, max_value=8, value=4, step=1)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(scaled_data)
    kmeans_cluster_labels = kmeans.labels_

    # Perform Hierarchical clustering
    hierarchical = AgglomerativeClustering(n_clusters=num_clusters)
    hierarchical_cluster_labels = hierarchical.fit_predict(scaled_data)

    # Visualizing the clusters
    plt.figure(figsize=(16, 6))

    # Plot KMeans clustering
    plt.subplot(1, 2, 1)
    plt.scatter(clustering_data['price'], clustering_data['mrp'], c=kmeans_cluster_labels, cmap='viridis', s=50)
    plt.title(f'KMeans Clustering (Number of Clusters: {num_clusters})')
    plt.xlabel('Price')
    plt.ylabel('MRP')
    plt.grid(True)

    # Plot Hierarchical clustering
    plt.subplot(1, 2, 2)
    plt.scatter(clustering_data['price'], clustering_data['mrp'], c=hierarchical_cluster_labels, cmap='viridis', s=50)
    plt.title(f'Hierarchical Clustering (Number of Clusters: {num_clusters})')
    plt.xlabel('Price')
    plt.ylabel('MRP')
    plt.grid(True)

    st.pyplot(plt)

    # Interpretation of clusters
    st.write(f"*Number of Clusters: {num_clusters}*")
    st.write("Kategori klustering yang berbeda berdasarkan harga dan MRP. Setiap cluster mewakili segmen produk berbeda di pasar, yang ditandai dengan titik harga dan MRP yang berbeda-beda. Menganalisis karakteristik masing-masing cluster dapat memberikan wawasan tentang preferensi pelanggan dan tren pasar, memungkinkan strategi pemasaran yang ditargetkan dan inisiatif pengembangan produk.")