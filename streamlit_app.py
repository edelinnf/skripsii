import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import random_center_initializer
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.express as px
import io
import warnings
warnings.filterwarnings('ignore')

# ---------------- Sidebar ---------------- #
# Sidebar - Header Branding
st.sidebar.markdown("## 🧠 Propalyze")
st.sidebar.title("Menu")

fitur = st.sidebar.radio("Pilih Halaman", ["📘 Penjelasan", "📊 Analisis & Klasterisasi"])

# ---------------- Fitur 1: Tentang Aplikasi ---------------- #
if fitur == "📘 Penjelasan":
    st.title("Analisis Klasterisasi Pelanggan Properti")
    st.markdown("""
    Aplikasi ini dibuat menggunakan **Streamlit** untuk melakukan *klasterisasi pelanggan properti* 
    menggunakan algoritma **X-Means**.
    
    ### 🔍 Fitur Utama:
    - Preprocessing data pelanggan properti
    - Normalisasi dan identifikasi keterlambatan
    - Klasterisasi menggunakan X-Means
    - Visualisasi hasil dengan **t-SNE 2D dan 3D**
    - Evaluasi model (Silhouette, DBI, CH Score)
    - Unduh hasil akhir ke Excel

    ### 📁 Format Data yang Dibutuhkan:
    - `angsuran.xlsx` (berisi transaksi)
    - `data master.xlsx` (berisi harga properti)

    Silakan masuk ke halaman **Analisis & Klasterisasi** untuk memulai.
    """)

# ---------------- Fitur 2: Analisis & Klasterisasi ---------------- #
elif fitur == "📊 Analisis & Klasterisasi":
    st.title("📊 Analisis & Klasterisasi Pelanggan Properti")

    # Upload file
    uploaded_angsuran = st.file_uploader("📤 Upload file angsuran.xlsx", type="xlsx", key="angsuran")
    uploaded_master = st.file_uploader("📤 Upload file data utama.xlsx", type="xlsx", key="data_master")

    if uploaded_angsuran and uploaded_master:
        df1 = pd.read_excel(uploaded_angsuran)
        df2 = pd.read_excel(uploaded_master)
        df3 = pd.read_excel(uploaded_angsuran)

        # Proses Data
        df1 = df1[['Nomor Unit', 'Nominal']].dropna()
        df2 = df2[['F', 'Unnamed: 3']].dropna()
        df3 = df3[['Nomor Unit', 'Tanggal Diterima', 'Tanggal Pembayaran']].dropna()

        df3['Tanggal Diterima'] = pd.to_datetime(df3['Tanggal Diterima'], dayfirst=True)
        df3['Tanggal Pembayaran'] = pd.to_datetime(df3['Tanggal Pembayaran'], dayfirst=True)

        agg_df = df1.groupby('Nomor Unit').agg({'Nominal': ['count', 'sum']}).reset_index()
        agg_df.columns = ['Nomor Unit', 'Jumlah Transaksi', 'Total Pembayaran']

        df2 = df2.iloc[2:].rename(columns={'F': 'Nomor Unit', 'Unnamed: 3': 'Harga'})
        df2['Harga'] = df2['Harga'].astype(float)

        dataset = pd.merge(agg_df, df2, on='Nomor Unit', how='inner')
        dataset['Selisih'] = dataset['Harga'] - dataset['Total Pembayaran']
        dataset['Status Pembayaran'] = dataset['Selisih'].apply(lambda x: 1 if x <= 0 else 0)

        df3['Selisih Hari'] = (df3['Tanggal Diterima'] - df3['Tanggal Pembayaran']).dt.days
        df4 = df3[df3['Selisih Hari'] > 0]
        df5 = df4.groupby('Nomor Unit').size().reset_index(name='Jumlah Terlambat')
        df6 = df3[['Nomor Unit']].drop_duplicates()
        data_terlambat = pd.merge(df6, df5, on='Nomor Unit', how='left')

        dataset = pd.merge(dataset, data_terlambat, on='Nomor Unit', how='left')
        dataset['Jumlah Terlambat'] = dataset['Jumlah Terlambat'].fillna(0).astype(int)

        # Preprocessing
        fitur = dataset[['Jumlah Transaksi', 'Total Pembayaran', 'Harga', 'Selisih', 'Status Pembayaran', 'Jumlah Terlambat']]
        scaler = RobustScaler()
        dataset_nrmlzd = pd.DataFrame(scaler.fit_transform(fitur), columns=fitur.columns)

        fitur_klaster = dataset_nrmlzd[['Jumlah Transaksi', 'Jumlah Terlambat', 'Selisih', 'Status Pembayaran']]
        fitur_np = fitur_klaster.values

        # X-Means Clustering
        initial_centers = random_center_initializer(fitur_np, 2).initialize()
        xmeans_model = xmeans(fitur_np, initial_centers, kmin=2, kmax=5)
        xmeans_model.process()
        clusters = xmeans_model.get_clusters()

        dataset['Klaster'] = -1
        for cluster_idx, instance_indices in enumerate(clusters):
            for instance_idx in instance_indices:
                if instance_idx < len(dataset):
                    dataset.loc[instance_idx, 'Klaster'] = cluster_idx

        # Evaluasi
        st.subheader("📈 Evaluasi Klaster")
        st.write(f"**Silhouette Score**: {silhouette_score(fitur_np, dataset['Klaster']):.3f}")
        st.write(f"**Davies-Bouldin Index**: {davies_bouldin_score(fitur_np, dataset['Klaster']):.3f}")
        st.write(f"**Calinski-Harabasz Score**: {calinski_harabasz_score(fitur_np, dataset['Klaster']):.3f}")

        perplexity = st.sidebar.slider("Perplexity t-SNE", 5, 50, 30)
        max_iter = st.sidebar.slider("Max Iterasi t-SNE", 250, 1000, 300)
        
        # t-SNE Visualisasi
        st.subheader("🧬 Visualisasi t-SNE 2D")
        tsne_2d = TSNE(n_components=2, perplexity=perplexity, max_iter=max_iter, random_state=42)
        tsne_result = tsne_2d.fit_transform(fitur_np)
        dataset['TSNE-1'], dataset['TSNE-2'] = tsne_result[:, 0], tsne_result[:, 1]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=dataset, x='TSNE-1', y='TSNE-2', hue='Klaster', palette='tab10', s=60)
        plt.title("Visualisasi Klaster dengan t-SNE 2D")
        st.pyplot(fig)

        # Visualisasi 3D Interaktif
        st.subheader("Visualisasi t-SNE 3D Interaktif")
        tsne_3d = TSNE(n_components=3, perplexity=30, n_iter=300, random_state=42)
        tsne_3d_result = tsne_3d.fit_transform(fitur_np)
        tsne_3d_df = pd.DataFrame(tsne_3d_result, columns=['TSNE1_3D', 'TSNE2_3D', 'TSNE3_3D'])
        tsne_3d_df['Klaster'] = dataset['Klaster']

        fig_3d = px.scatter_3d(tsne_3d_df, x='TSNE1_3D', y='TSNE2_3D', z='TSNE3_3D', color='Klaster',
                               color_discrete_sequence=px.colors.qualitative.T10,
                               title='Visualisasi 3D Klaster dengan t-SNE')
        st.plotly_chart(fig_3d)

        st.subheader("📊 Dataset Final")
        st.dataframe(dataset)

        # Download Button
        st.subheader("📥 Unduh Hasil Klasterisasi")
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            dataset.to_excel(writer, index=False, sheet_name='Hasil Klaster')
            writer.save()
            processed_data = output.getvalue()

        st.download_button(
            label="📁 Download Excel",
            data=processed_data,
            file_name='hasil_klaster.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    else:
        st.info("📂 Silakan upload kedua file Excel.")
