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
import warnings
import io

warnings.filterwarnings('ignore')

st.title("Analisis Klasterisasi Pelanggan Properti (X-Means)")

# Upload file
uploaded_angsuran = st.file_uploader("Upload file angsuran.xlsx", type="xlsx")
uploaded_utama = st.file_uploader("Upload file data utama.xlsx", type="xlsx")

if uploaded_angsuran and uploaded_utama:
    df1 = pd.read_excel(uploaded_angsuran)
    df2 = pd.read_excel(uploaded_utama)
    df3 = pd.read_excel(uploaded_angsuran)

    # --- Pemilihan Data ---
    df1 = df1[['Nomor Unit', 'Nominal']].dropna()
    df2 = df2[['F', 'Unnamed: 3']].dropna()
    df3 = df3[['Nomor Unit', 'Tanggal Diterima', 'Tanggal Pembayaran']].dropna()

    # Konversi tanggal
    df3['Tanggal Diterima'] = pd.to_datetime(df3['Tanggal Diterima'], dayfirst=True)
    df3['Tanggal Pembayaran'] = pd.to_datetime(df3['Tanggal Pembayaran'], dayfirst=True)

    # Agregasi
    agg_df = df1.groupby('Nomor Unit').agg({'Nominal': ['count', 'sum']}).reset_index()
    agg_df.columns = ['Nomor Unit', 'Jumlah Transaksi', 'Total Pembayaran']

    # Ubah nama kolom
    df2 = df2.iloc[2:].rename(columns={'F': 'Nomor Unit', 'Unnamed: 3': 'Harga'})

    # Merge
    dataset = pd.merge(agg_df, df2, on='Nomor Unit', how='inner')
    dataset['Harga'] = dataset['Harga'].astype(float)
    dataset['Selisih'] = dataset['Harga'] - dataset['Total Pembayaran']
    dataset['Status Pembayaran'] = dataset['Selisih'].apply(lambda x: "Lunas" if x <= 0 else "Belum Lunas")

    # Keterlambatan
    df3['Selisih Hari'] = (df3['Tanggal Diterima'] - df3['Tanggal Pembayaran']).dt.days
    df4 = df3[df3['Selisih Hari'] > 0]
    df5 = df4.groupby('Nomor Unit').size().reset_index(name='Jumlah Terlambat')
    df6 = df3[['Nomor Unit']].drop_duplicates()
    data_terlambat = pd.merge(df6, df5, on='Nomor Unit', how='left')
    dataset = pd.merge(dataset, data_terlambat, on='Nomor Unit', how='left')
    dataset['Jumlah Terlambat'] = dataset['Jumlah Terlambat'].fillna(0).astype(int)

    # Preprocessing
    dataset['Status Pembayaran'] = dataset['Status Pembayaran'].apply(lambda x: 1 if x == "Lunas" else 0)
    fitur = dataset[['Jumlah Transaksi', 'Total Pembayaran', 'Harga', 'Selisih', 'Status Pembayaran', 'Jumlah Terlambat']]
    scaler = RobustScaler()
    dataset_nrmlzd = pd.DataFrame(scaler.fit_transform(fitur), columns=fitur.columns)

    # VIF
    vif_data = pd.DataFrame()
    vif_data["Fitur"] = dataset_nrmlzd.columns
    vif_data["VIF"] = [variance_inflation_factor(dataset_nrmlzd.values, i) for i in range(dataset_nrmlzd.shape[1])]
    st.subheader("Multikolinearitas (VIF)")
    st.dataframe(vif_data)

    # Klasterisasi
    fitur_klaster = dataset_nrmlzd[['Jumlah Transaksi', 'Jumlah Terlambat', 'Selisih', 'Status Pembayaran']]
    fitur_np = fitur_klaster.values
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
    st.subheader("Evaluasi Klaster")
    st.write(f"**Silhouette Score:** {silhouette_score(fitur_np, dataset['Klaster']):.3f}")
    st.write(f"**Davies-Bouldin Index:** {davies_bouldin_score(fitur_np, dataset['Klaster']):.3f}")
    st.write(f"**Calinski-Harabasz Score:** {calinski_harabasz_score(fitur_np, dataset['Klaster']):.3f}")

    # Interpretasi
    st.subheader("Rata-Rata Fitur per Klaster")
    cluster_means = dataset.groupby('Klaster')[['Jumlah Transaksi', 'Jumlah Terlambat', 'Selisih', 'Status Pembayaran']].mean()
    st.dataframe(cluster_means)

    # Visualisasi t-SNE
    st.subheader("Visualisasi t-SNE 2D")
    tsne_2d = TSNE(n_components=2, perplexity=30, max_iter=300, random_state=42)
    tsne_result = tsne_2d.fit_transform(fitur_np)
    dataset['TSNE-1'], dataset['TSNE-2'] = tsne_result[:, 0], tsne_result[:, 1]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=dataset, x='TSNE-1', y='TSNE-2', hue='Klaster', palette='tab10', s=60)
    plt.title("Visualisasi Klaster dengan t-SNE 2D")
    st.pyplot(fig)

    # Visualisasi 3D Interaktif
    st.subheader("Visualisasi t-SNE 3D Interaktif")
    tsne_3d = TSNE(n_components=3, perplexity=30, max_iter=300, random_state=42)
    tsne_3d_result = tsne_3d.fit_transform(fitur_np)
    tsne_3d_df = pd.DataFrame(tsne_3d_result, columns=['TSNE1_3D', 'TSNE2_3D', 'TSNE3_3D'])
    tsne_3d_df['Klaster'] = dataset['Klaster']

    fig_3d = px.scatter_3d(tsne_3d_df, x='TSNE1_3D', y='TSNE2_3D', z='TSNE3_3D', color='Klaster',
                           color_discrete_sequence=px.colors.qualitative.T10,
                           title='Visualisasi 3D Klaster dengan t-SNE')
    st.plotly_chart(fig_3d)

    # Tampilkan dataset final
    st.subheader("Dataset Final dengan Label Klaster")
    st.dataframe(dataset[['Nomor Unit', 'Jumlah Transaksi', 'Total Pembayaran', 'Harga',
                          'Selisih', 'Status Pembayaran', 'Jumlah Terlambat', 'Klaster']])

    st.subheader("📥 Unduh Hasil Klasterisasi")

    # Konversi ke Excel dan buat buffer
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        dataset.to_excel(writer, index=False, sheet_name='Hasil Klaster')
        writer.save()
        processed_data = output.getvalue()

    # Tombol download
    st.download_button(
        label="📁 Download Hasil sebagai Excel",
        data=processed_data,
        file_name='hasil_klaster.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )


else:
    st.info("Silakan unggah kedua file Excel terlebih dahulu.")

