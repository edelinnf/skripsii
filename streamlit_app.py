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
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(to bottom left, #b8f0d4, #8cc6ff);
        padding: 2rem 1rem;
    }

    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] p {
        color: #002B5B;
    }

    .sidebar-title {
        font-size: 24px;
        font-weight: bold;
        color: #002B5B;
        margin-bottom: 10px;
        text-align: center;
    }

    .stRadio > div {
        background-color: #a0d2ff;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
        box-shadow: 0 0 4px rgba(0,0,0,0.1);
        font-weight: 500;
    }

    .stRadio > div:hover {
        background-color: #80bfff;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# CSS untuk style menu kotak (untuk tombol Streamlit)
st.sidebar.markdown(
    """
    <style>
    /* Style untuk tombol di sidebar */
    .stButton > button {
        background-color: #a3d8f4 !important;
        color: #002b5c !important;
        border: 2px solid transparent !important;
        border-radius: 12px !important;
        padding: 10px 16px !important;
        font-weight: 400 !important;
        width: 100% !important;
        text-align: left !important;
        margin-bottom: 10px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background-color: #74c0fc !important;
        border: 2px solid #1c7ed6 !important;
        color: #002b5c !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    }
    
    .stButton > button:active,
    .stButton > button:focus {
        background-color: #1c7ed6 !important;
        color: white !important;
        border: 2px solid #1c7ed6 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    /* Khusus untuk sidebar */
    [data-testid="stSidebar"] .stButton > button {
        background-color: #a3d8f4 !important;
        color: #002b5c !important;
        border: 2px solid transparent !important;
        border-radius: 12px !important;
        padding: 10px 16px !important;
        font-weight: 600 !important;
        width: 100% !important;
        text-align: center !important;
        margin-bottom: 10px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #74c0fc !important;
        border: 2px solid #1c7ed6 !important;
        color: #002b5c !important;
        transform: translateY(-2px) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Inisialisasi halaman aktif
if "halaman" not in st.session_state:
    st.session_state.halaman = "Penjelasan"

# ----------------- Sidebar: Branding dan Navigasi ----------------- #
with st.sidebar:
    st.markdown("# 🧠 Propalyze")
    st.markdown("### Menu")

    # Membuat tombol navigasi yang benar-benar berfungsi
    if st.button("📘 Penjelasan", key="btn_penjelasan", use_container_width=True):
        st.session_state.halaman = "Penjelasan"
    
    if st.button("📁 Data", key="btn_data", use_container_width=True):
        st.session_state.halaman = "Data"
    
    if st.button("📊 Analisis & Klasterisasi", key="btn_analisis", use_container_width=True):
        st.session_state.halaman = "Analisis & Klasterisasi"

# ----------------- Header ----------------- #
st.markdown("<h1 style='text-align: center;'>Property Analysis</h1>", unsafe_allow_html=True)

# ----------------- Fitur 1: Penjelasan ----------------- #
if st.session_state.halaman == "Penjelasan":
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
elif st.session_state.halaman == "Data":
    st.write("Halaman untuk lihat data.")
    st.header("📂 Unggah Dataset")
    uploaded_angsuran = st.file_uploader("Upload file angsuran.xlsx", type="xlsx", key="angsuran")
    uploaded_master = st.file_uploader("Upload file data utama.xlsx", type="xlsx", key="data_master")

    if uploaded_angsuran and uploaded_master:
        st.success("✅ Kedua file berhasil diunggah. Silakan lanjut ke menu 'Analisis & Klasterisasi'.")
        
        # Simpan file yang diupload ke session state
        st.session_state.uploaded_angsuran = uploaded_angsuran
        st.session_state.uploaded_master = uploaded_master
        
        # Preview data
        try:
            df1 = pd.read_excel(uploaded_angsuran)
            df2 = pd.read_excel(uploaded_master)
            
            st.subheader("📊 Preview Data Angsuran")
            st.dataframe(df1.head())
            
            st.subheader("📊 Preview Data Master")
            st.dataframe(df2.head())
            
        except Exception as e:
            st.error(f"Error membaca file: {str(e)}")

# ----------------- Fitur 3: Analisis & Klasterisasi ----------------- #
elif st.session_state.halaman == "Analisis & Klasterisasi":
    st.write("Analisis dan visualisasi klaster pelanggan.")

    # Cek apakah file sudah diupload sebelumnya
    if 'uploaded_angsuran' in st.session_state and 'uploaded_master' in st.session_state:
        uploaded_angsuran = st.session_state.uploaded_angsuran
        uploaded_master = st.session_state.uploaded_master
        st.success("✅ Data sudah tersedia dari halaman sebelumnya.")
    else:
        st.warning("⚠️ Data belum diupload. Silakan upload file terlebih dahulu.")
        uploaded_angsuran = st.file_uploader("Upload file angsuran.xlsx", type="xlsx", key="angsuran2")
        uploaded_master = st.file_uploader("Upload file data utama.xlsx", type="xlsx", key="data_master2")

    if uploaded_angsuran and uploaded_master:
        try:
            # Membaca data
            df1 = pd.read_excel(uploaded_angsuran)
            df2 = pd.read_excel(uploaded_master)
            df3 = pd.read_excel(uploaded_angsuran)

            # Preprocessing
            with st.spinner("Memproses data..."):
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

            st.success("✅ Data berhasil diproses!")
            
            # Normalisasi dan Clustering
            with st.spinner("Melakukan klasterisasi..."):
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
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Silhouette Score", f"{silhouette_score(fitur_np, dataset['Klaster']):.3f}")
            with col2:
                st.metric("Davies-Bouldin Index", f"{davies_bouldin_score(fitur_np, dataset['Klaster']):.3f}")
            with col3:
                st.metric("Calinski-Harabasz Score", f"{calinski_harabasz_score(fitur_np, dataset['Klaster']):.0f}")

            # Visualisasi t-SNE
            st.subheader("🔧 Pengaturan Visualisasi")
            col1, col2 = st.columns(2)
            with col1:
                perplexity = st.slider("Perplexity t-SNE", 5, 50, 30)
            with col2:
                max_iter = st.slider("Max Iterasi t-SNE", 250, 1000, 300)

            # Visualisasi t-SNE 2D
            st.subheader("🧬 Visualisasi t-SNE 2D")
            with st.spinner("Membuat visualisasi 2D..."):
                tsne_2d = TSNE(n_components=2, perplexity=perplexity, n_iter=max_iter, random_state=42)
                tsne_result = tsne_2d.fit_transform(fitur_np)
                dataset['TSNE-1'], dataset['TSNE-2'] = tsne_result[:, 0], tsne_result[:, 1]

                fig, ax = plt.subplots(figsize=(12, 8))
                sns.scatterplot(data=dataset, x='TSNE-1', y='TSNE-2', hue='Klaster', palette='tab10', s=60, alpha=0.7)
                plt.title("Visualisasi Klaster dengan t-SNE 2D", fontsize=16)
                plt.xlabel("t-SNE Dimensi 1", fontsize=12)
                plt.ylabel("t-SNE Dimensi 2", fontsize=12)
                plt.legend(title="Klaster", bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                st.pyplot(fig)

            # Visualisasi t-SNE 3D
            st.subheader("🌐 Visualisasi t-SNE 3D Interaktif")
            with st.spinner("Membuat visualisasi 3D..."):
                tsne_2d = TSNE(n_components=2, perplexity=perplexity, n_iter=max_iter, random_state=42)
                tsne_3d_result = tsne_3d.fit_transform(fitur_np)
                tsne_3d_df = pd.DataFrame(tsne_3d_result, columns=['TSNE1_3D', 'TSNE2_3D', 'TSNE3_3D'])
                tsne_3d_df['Klaster'] = dataset['Klaster'].astype(str)

                fig_3d = px.scatter_3d(tsne_3d_df, x='TSNE1_3D', y='TSNE2_3D', z='TSNE3_3D', 
                                       color='Klaster',
                                       color_discrete_sequence=px.colors.qualitative.T10,
                                       title='Visualisasi 3D Klaster dengan t-SNE',
                                       labels={'TSNE1_3D': 't-SNE Dimensi 1', 
                                               'TSNE2_3D': 't-SNE Dimensi 2', 
                                               'TSNE3_3D': 't-SNE Dimensi 3'})
                fig_3d.update_layout(height=600)
                st.plotly_chart(fig_3d, use_container_width=True)

            # Analisis Klaster
            st.subheader("📊 Analisis Klaster")
            cluster_summary = dataset.groupby('Klaster').agg({
                'Jumlah Transaksi': ['mean', 'count'],
                'Total Pembayaran': 'mean',
                'Harga': 'mean',
                'Selisih': 'mean',
                'Status Pembayaran': 'mean',
                'Jumlah Terlambat': 'mean'
            }).round(2)
            
            st.dataframe(cluster_summary)

            # Tabel Dataset Final
            st.subheader("📋 Dataset Final")
            st.dataframe(dataset, use_container_width=True)

            # Export ke Excel
            st.subheader("📥 Download Hasil")
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                dataset.to_excel(writer, index=False, sheet_name='Hasil_Cluster')
                cluster_summary.to_excel(writer, sheet_name='Ringkasan_Cluster')
            
            processed_data = output.getvalue()
            st.download_button(
                label="📥 Download Hasil Excel",
                data=processed_data,
                file_name="hasil_klaster.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan: {str(e)}")
            st.info("Pastikan format file Excel sesuai dengan yang diharapkan.")

    else:
        st.warning("⚠️ Silakan upload kedua file terlebih dahulu di halaman **Data** atau di atas.")

# Status bar di bagian bawah
st.markdown("---")
st.markdown(f"**Status Halaman Aktif:** {st.session_state.halaman}")
