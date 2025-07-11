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

# ----------------- Custom CSS Styling ----------------- #

# ----------------- Sidebar: Branding dan Navigasi ----------------- #
st.sidebar.title("Menu")
st.sidebar.markdown("## 🧠 Propalyze")

# CSS untuk style menu kotak
st.sidebar.markdown(
    """
    <style>
    .sidebar-box {
        background-color: #a3d8f4;
        padding: 10px 16px;
        margin-bottom: 10px;
        border-radius: 12px;
        color: #002b5c;
        font-weight: 600;
        cursor: pointer;
        text-align: left;
        border: 2px solid transparent;
    }
    .sidebar-box:hover {
        background-color: #74c0fc;
        border: 2px solid #1c7ed6;
    }
    .active {
        background-color: #1c7ed6 !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Inisialisasi halaman aktif
if "page" not in st.session_state:
    st.session_state.page = "Penjelasan"

# Tombol menu sidebar
def render_button(label, icon):
    active = "active" if st.session_state.page == label else ""
    st.sidebar.markdown(
        f'<div class="sidebar-box {active}" onclick="window.location.href=\'/?page={label}\'">{icon} {label}</div>',
        unsafe_allow_html=True
    )

# Menu tombol (ganti sesuai halaman)
render_button("Penjelasan", "📘")
render_button("Data", "💾")
render_button("Analisis & Klasterisasi", "📊")

# Tangani halaman via query param
query_params = st.query_params
if "page" in query_params:
    st.session_state.page = query_params["page"][0]

# ----------------- Header ----------------- #
st.markdown("<h1 style='text-align: center;'>Property Analysis</h1>", unsafe_allow_html=True)

# ----------------- Fitur 1: Penjelasan ----------------- #
if st.session_state.page == "Penjelasan":
    st.title("📘 Penjelasan")
    st.write("Ini adalah penjelasan aplikasi.")
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

    Silakan masuk ke halaman **Data** untuk mengunggah file Excel, lalu lanjut ke **Analisis & Klasterisasi**.
    """)

# ----------------- Fitur 2: Upload Data ----------------- #
elif st.session_state.page == "Data":
    st.title("💾 Data")
    st.write("Halaman untuk lihat data.")
    st.header("📂 Unggah Dataset")
    uploaded_angsuran = st.file_uploader("Upload file angsuran.xlsx", type="xlsx", key="angsuran")
    uploaded_master = st.file_uploader("Upload file data utama.xlsx", type="xlsx", key="data_master")

    if uploaded_angsuran and uploaded_master:
        st.success("✅ Kedua file berhasil diunggah. Silakan lanjut ke menu 'Analisis & Klasterisasi'.")

# ----------------- Fitur 3: Analisis & Klasterisasi ----------------- #
elif st.session_state.page == "Analisis & Klasterisasi":
    st.title("📊 Analisis & Klasterisasi")
    st.write("Analisis dan visualisasi klaster pelanggan.")

    uploaded_angsuran = st.file_uploader("Upload file angsuran.xlsx", type="xlsx", key="angsuran2")
    uploaded_master = st.file_uploader("Upload file data utama.xlsx", type="xlsx", key="data_master2")

    if uploaded_angsuran and uploaded_master:
        df1 = pd.read_excel(uploaded_angsuran)
        df2 = pd.read_excel(uploaded_master)
        df3 = pd.read_excel(uploaded_angsuran)

        # Preprocessing
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

        # Normalisasi dan Clustering
        fitur = dataset[['Jumlah Transaksi', 'Total Pembayaran', 'Harga', 'Selisih', 'Status Pembayaran', 'Jumlah Terlambat']]
        scaler = RobustScaler()
        dataset_nrmlzd = pd.DataFrame(scaler.fit_transform(fitur), columns=fitur.columns)

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
        st.subheader("📈 Evaluasi Klaster")
        st.write(f"**Silhouette Score**: {silhouette_score(fitur_np, dataset['Klaster']):.3f}")
        st.write(f"**Davies-Bouldin Index**: {davies_bouldin_score(fitur_np, dataset['Klaster']):.3f}")
        st.write(f"**Calinski-Harabasz Score**: {calinski_harabasz_score(fitur_np, dataset['Klaster']):.3f}")

        # Visualisasi t-SNE
        perplexity = st.slider("Perplexity t-SNE", 5, 50, 30)
        max_iter = st.slider("Max Iterasi t-SNE", 250, 1000, 300)

        st.subheader("🧬 Visualisasi t-SNE 2D")
        tsne_2d = TSNE(n_components=2, perplexity=perplexity, n_iter=max_iter, random_state=42)
        tsne_result = tsne_2d.fit_transform(fitur_np)
        dataset['TSNE-1'], dataset['TSNE-2'] = tsne_result[:, 0], tsne_result[:, 1]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=dataset, x='TSNE-1', y='TSNE-2', hue='Klaster', palette='tab10', s=60)
        plt.title("Visualisasi Klaster dengan t-SNE 2D")
        st.pyplot(fig)

        # Visualisasi t-SNE 3D
        st.subheader("Visualisasi t-SNE 3D Interaktif")
        tsne_3d = TSNE(n_components=3, perplexity=30, n_iter=300, random_state=42)
        tsne_3d_result = tsne_3d.fit_transform(fitur_np)
        tsne_3d_df = pd.DataFrame(tsne_3d_result, columns=['TSNE1_3D', 'TSNE2_3D', 'TSNE3_3D'])
        tsne_3d_df['Klaster'] = dataset['Klaster']

        fig_3d = px.scatter_3d(tsne_3d_df, x='TSNE1_3D', y='TSNE2_3D', z='TSNE3_3D', color='Klaster',
                               color_discrete_sequence=px.colors.qualitative.T10,
                               title='Visualisasi 3D Klaster dengan t-SNE')
        st.plotly_chart(fig_3d)

        # Tabel & Export
        st.subheader("📊 Dataset Final")
        st.dataframe(dataset)

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            dataset.to_excel(writer, index=False, sheet_name='Hasil_Cluster')
            writer.save()
        st.download_button("📥 Download Hasil Excel", output.getvalue(), file_name="hasil_klaster.xlsx")

    else:
        st.warning("⚠️ Silakan upload kedua file terlebih dahulu.")
