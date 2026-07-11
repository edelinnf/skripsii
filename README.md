# Customer Segmentation Using X-Means Clustering

Implementasi penelitian skripsi mengenai **segmentasi pelanggan berdasarkan pola pembayaran** menggunakan algoritma **X-Means Clustering**. Aplikasi dikembangkan menggunakan **Python** dan **Streamlit** untuk mempermudah visualisasi hasil analisis dan mendukung pengambilan keputusan berbasis data.

---

## 📖 About the Project

Penelitian ini bertujuan untuk mengelompokkan pelanggan berdasarkan pola pembayaran historis sehingga perusahaan dapat memahami karakteristik setiap segmen pelanggan. Hasil segmentasi diharapkan dapat membantu perusahaan dalam menyusun strategi pelayanan, komunikasi, dan mitigasi risiko pembayaran. :contentReference[oaicite:1]{index=1}

Berbeda dengan K-Means yang mengharuskan jumlah cluster ditentukan di awal, **X-Means** mampu menentukan jumlah cluster yang lebih optimal secara otomatis menggunakan **Bayesian Information Criterion (BIC)**. :contentReference[oaicite:2]{index=2}

---

## ✨ Features

- Data preprocessing
- Data scaling menggunakan **RobustScaler**
- Customer segmentation menggunakan **X-Means Clustering**
- Evaluasi cluster menggunakan:
  - Silhouette Score
  - Davies-Bouldin Index (DBI)
  - Calinski-Harabasz Index (CHI)
- Visualisasi hasil clustering
- Dashboard interaktif menggunakan **Streamlit**

---

## 🛠 Technologies

- Python 3
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Plotly

---

## 📂 Repository Structure

```
.
├── streamlit_app.py          # Aplikasi Streamlit
├── requirements.txt          # Daftar dependencies
├── Laporan Skripsi.pdf       # Dokumen skripsi
├── README.md
└── .devcontainer/
```

---

## 🚀 Installation

Clone repository

```bash
git clone https://github.com/username/repository-name.git
cd repository-name
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Streamlit

```bash
streamlit run streamlit_app.py
```

---

## 📊 Methodology

Tahapan penelitian meliputi:

1. Data Collection
2. Data Preprocessing
3. Feature Scaling menggunakan RobustScaler
4. Customer Segmentation menggunakan X-Means
5. Cluster Evaluation
6. Data Visualization
7. Business Interpretation

---

## 📚 Research

Topik penelitian:

> **Segmentasi Pelanggan Berdasarkan Pola Pembayaran Menggunakan Algoritma X-Means Clustering**

Penelitian dilakukan menggunakan data historis pembayaran pelanggan pada perusahaan properti (PT XYZ) untuk mengidentifikasi karakteristik pelanggan berdasarkan perilaku pembayarannya. :contentReference[oaicite:3]{index=3}

---

## 📄 Documentation

Laporan penelitian dapat dilihat pada file:

```
Laporan Skripsi.pdf
```

---

## 👤 Author

**Edelin Fortuna**

Bachelor of Data Science

Universitas Pembangunan Nasional "Veteran" Jawa Timur

---
