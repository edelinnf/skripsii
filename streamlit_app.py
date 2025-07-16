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

# ----------------- Fitur 2: Upload & Proses Data ----------------- #
elif st.session_state.halaman == "Data":
    st.title("📁 Data")
    st.write("Unggah file dan lihat proses pembentukan dataset hingga final.")
    uploaded_angsuran = st.file_uploader("Upload file angsuran.xlsx", type="xlsx", key="angsuran")
    uploaded_master = st.file_uploader("Upload file data utama.xlsx", type="xlsx", key="data_master")

    if uploaded_angsuran and uploaded_master:
        st.success("✅ Kedua file berhasil diunggah.")

        try:
            # Baca file Excel
            df1 = pd.read_excel(uploaded_angsuran)
            df2 = pd.read_excel(uploaded_master)
            df3 = pd.read_excel(uploaded_angsuran)

            st.subheader("🔧 Proses Preprocessing")
            with st.spinner("Memproses data..."):
                # --- Langkah 1: Ringkasan Transaksi ---
                df1 = df1[['Nomor Unit', 'Nominal']].dropna()
                agg_df = df1.groupby('Nomor Unit').agg({'Nominal': ['count', 'sum']}).reset_index()
                agg_df.columns = ['Nomor Unit', 'Jumlah Transaksi', 'Total Pembayaran']
                st.markdown("✅ **Ringkasan Transaksi**")
                st.dataframe(agg_df.head())

                # --- Langkah 2: Bersihkan Data Master ---
                df2 = df2[['F', 'Unnamed: 3']].dropna()
                df2 = df2.iloc[2:].rename(columns={'F': 'Nomor Unit', 'Unnamed: 3': 'Harga'})
                df2['Harga'] = df2['Harga'].astype(float)
                st.markdown("✅ **Harga Properti**")
                st.dataframe(df2.head())

                # --- Langkah 3: Gabungkan Harga + Transaksi ---
                dataset = pd.merge(agg_df, df2, on='Nomor Unit', how='inner')
                dataset['Selisih'] = dataset['Harga'] - dataset['Total Pembayaran']
                dataset['Status Pembayaran'] = dataset['Selisih'].apply(lambda x: 1 if x <= 0 else 0)
                st.markdown("✅ **Gabungan Transaksi & Harga**")
                st.dataframe(dataset.head())

                # --- Langkah 4: Hitung Keterlambatan ---
                df3 = df3[['Nomor Unit', 'Tanggal Diterima', 'Tanggal Pembayaran']].dropna()
                df3['Tanggal Diterima'] = pd.to_datetime(df3['Tanggal Diterima'], dayfirst=True)
                df3['Tanggal Pembayaran'] = pd.to_datetime(df3['Tanggal Pembayaran'], dayfirst=True)
                df3['Selisih Hari'] = (df3['Tanggal Diterima'] - df3['Tanggal Pembayaran']).dt.days
                df4 = df3[df3['Selisih Hari'] > 0]
                df5 = df4.groupby('Nomor Unit').size().reset_index(name='Jumlah Terlambat')
                df6 = df3[['Nomor Unit']].drop_duplicates()
                data_terlambat = pd.merge(df6, df5, on='Nomor Unit', how='left')
                st.markdown("✅ **Data Keterlambatan**")
                st.dataframe(data_terlambat.head())

                # --- Langkah 5: Gabungkan Keterlambatan ke Dataset ---
                dataset = pd.merge(dataset, data_terlambat, on='Nomor Unit', how='left')
                dataset['Jumlah Terlambat'] = dataset['Jumlah Terlambat'].fillna(0).astype(int)

            st.success("✅ Data berhasil diproses!")

            # --- Tampilkan Dataset Final ---
            st.subheader("📌 Dataset Final")
            st.dataframe(dataset)

            # Simpan ke session_state
            st.session_state.dataset_final = dataset

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memproses file: {str(e)}")

    else:
        st.info("ℹ️ Silakan unggah kedua file untuk melihat proses dan dataset final.")

# ----------------- Fitur 3: Analisis & Klasterisasi ----------------- #
elif st.session_state.halaman == "Analisis & Klasterisasi":
    st.write("Unggah dataset (hasil dari fitur 'Data'), lakukan klasterisasi X-Means dan interpretasi hasil.")

    # Ambil dataset final dari session_state atau upload manual
    if 'dataset_final' in st.session_state:
        dataset = st.session_state.dataset_final
        st.success("✅ Dataset final tersedia dari halaman sebelumnya.")
    else:
        uploaded_dataset = st.file_uploader("📂 Upload dataset final (.xlsx)", type="xlsx")
        if uploaded_dataset:
            dataset = pd.read_excel(uploaded_dataset)
            st.success("✅ Dataset berhasil diunggah.")
        else:
            st.warning("⚠️ Dataset belum tersedia. Unggah dari halaman 'Data' atau upload file di atas.")
            st.stop()

    # ---------- Preprocessing ---------- #
    # --- PILIH FITUR UNTUK KLASTERISASI ---
    st.subheader("🧮 Pilih Fitur untuk Klasterisasi")
    all_features = ['Jumlah Transaksi', 'Total Pembayaran', 'Harga', 'Selisih', 'Status Pembayaran', 'Jumlah Terlambat']
    selected_features = st.multiselect("Pilih fitur yang akan digunakan untuk klasterisasi:", options=all_features, default=all_features[:4])

    # Validasi pemilihan fitur
    if len(selected_features) < 2:
        st.warning("⚠️ Pilih minimal 2 fitur untuk dapat melakukan klasterisasi.")
        st.stop()

    # --- NORMALISASI ---
    st.markdown("✅ Fitur yang dipilih akan dinormalisasi sebelum dilakukan klasterisasi.")
    fitur = dataset[selected_features]
    scaler = RobustScaler()
    dataset_nrmlzd = pd.DataFrame(scaler.fit_transform(fitur), columns=fitur.columns)

    # --- VISUALISASI DISTRIBUSI ---
    st.subheader("📈 Visualisasi Distribusi Fitur Terpilih")
    num_plots = len(selected_features)
    n_cols = 3
    n_rows = (num_plots // n_cols) + int(num_plots % n_cols != 0)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, f in enumerate(selected_features):
        sns.histplot(dataset_nrmlzd[f], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribusi {f}')

    # Kosongkan sisa subplot jika ada
    for j in range(len(selected_features), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    st.pyplot(fig)

    # --- SIMPAN UNTUK MODELING ---
    fitur_klaster = dataset_nrmlzd
    fitur_np = fitur_klaster.values


    # ---------- X-Means Clustering ----------
    st.subheader("📌 Klasterisasi X-Means")
    st.markdown("""
    X-Means mencari jumlah klaster optimal secara otomatis berdasarkan k-min dan k-max.
    """)
    kmin = st.slider("K-Min", 2, 5, 2)
    kmax = st.slider("K-Max", 3, 10, 5)

    with st.spinner("⏳ Menjalankan model X-Means..."):
        initial_centers = random_center_initializer(fitur_np, kmin).initialize()
        xmeans_model = xmeans(fitur_np, initial_centers, kmin=kmin, kmax=kmax)
        xmeans_model.process()
        clusters = xmeans_model.get_clusters()

        dataset['Klaster'] = -1
        for cluster_idx, instance_indices in enumerate(clusters):
            for instance_idx in instance_indices:
                if instance_idx < len(dataset):
                    dataset.loc[instance_idx, 'Klaster'] = cluster_idx

    st.success(f"✅ Klasterisasi selesai! Jumlah klaster ditemukan: **{len(clusters)}**")
    # Tampilkan tabel berisi fitur yang digunakan + Klaster
    st.subheader("📋 Hasil Klasterisasi")
    st.dataframe(dataset[['Nomor Unit'] + selected_features + ['Klaster']], use_container_width=True)


    # ---------- Evaluasi ----------
    st.subheader("📈 Evaluasi Klaster")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Silhouette Score", f"{silhouette_score(fitur_np, dataset['Klaster']):.3f}")
    with col2:
        st.metric("Davies-Bouldin Index", f"{davies_bouldin_score(fitur_np, dataset['Klaster']):.3f}")
    with col3:
        st.metric("Calinski-Harabasz Score", f"{calinski_harabasz_score(fitur_np, dataset['Klaster']):.0f}")

    # ---------- Visualisasi t-SNE ----------#
    st.subheader("📊 Visualisasi Status Pembayaran per Klaster")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Klaster', hue='Status Pembayaran', data=dataset, ax=ax)
    ax.set_title('Jumlah Pelanggan Berdasarkan Klaster dan Status Pembayaran')
    ax.set_xlabel('Klaster')
    ax.set_ylabel('Jumlah')
    ax.legend(title='Status Pembayaran', labels=['Belum Lunas', 'Lunas'])
    st.pyplot(fig)

    st.subheader("🧬 Visualisasi t-SNE 2D")
    perplexity = st.slider("Perplexity", 5, 50, 30)
    tsne_2d = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_result = tsne_2d.fit_transform(fitur_np)
    dataset['TSNE-1'], dataset['TSNE-2'] = tsne_result[:, 0], tsne_result[:, 1]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=dataset, x='TSNE-1', y='TSNE-2', hue='Klaster', palette='tab10', s=60)
    plt.title("Visualisasi Klaster dengan t-SNE 2D")
    st.pyplot(fig)

    st.subheader("🌐 Visualisasi t-SNE 3D Interaktif")

    try:
        tsne_3d_model = TSNE(n_components=3, perplexity=30, random_state=42)
        tsne_3d_result = tsne_3d_model.fit_transform(fitur_np)
        tsne_3d_df = pd.DataFrame(tsne_3d_result, columns=['TSNE1_3D', 'TSNE2_3D', 'TSNE3_3D'])
        tsne_3d_df['Klaster'] = dataset['Klaster'].astype(str)

        fig_3d = px.scatter_3d(tsne_3d_df,
                               x='TSNE1_3D', y='TSNE2_3D', z='TSNE3_3D',
                               color='Klaster',
                               color_discrete_sequence=px.colors.qualitative.T10,
                               title='Visualisasi 3D Klaster dengan t-SNE',
                               labels={'TSNE1_3D': 't-SNE Dimensi 1',
                                       'TSNE2_3D': 't-SNE Dimensi 2',
                                       'TSNE3_3D': 't-SNE Dimensi 3'})
        fig_3d.update_layout(height=600)
        st.plotly_chart(fig_3d, use_container_width=True)

    except Exception as e:
        st.error(f"Gagal menampilkan t-SNE 3D: {str(e)}")

    # ---------- Interpretasi ----------
    st.subheader("📌 Interpretasi & Analisis Klaster")
    cluster_summary = dataset.groupby('Klaster').agg({
        'Jumlah Transaksi': ['mean', 'count'],
        'Total Pembayaran': 'mean',
        'Harga': 'mean',
        'Selisih': 'mean',
        'Status Pembayaran': 'mean',
        'Jumlah Terlambat': 'mean'
    }).round(2)
    st.dataframe(cluster_summary)

    st.markdown("""
    **Interpretasi Awal:**
    - Klaster dengan nilai *Status Pembayaran* rata-rata mendekati 1 menunjukkan kelompok yang lunas.
    - Klaster dengan rata-rata *Selisih* tinggi kemungkinan besar adalah pelanggan dengan tunggakan.
    - *Jumlah Terlambat* bisa mengindikasikan kedisiplinan dalam pembayaran.
    """)

    # Simpan hasil ke session dan download
    st.session_state.dataset_klaster = dataset
    st.download_button("📥 Download Dataset Hasil", data=dataset.to_csv(index=False).encode(),
                       file_name="hasil_klaster.csv", mime="text/csv")

# Status bar di bagian bawah
st.markdown("---")
st.markdown(f"**Status Halaman Aktif:** {st.session_state.halaman}")
