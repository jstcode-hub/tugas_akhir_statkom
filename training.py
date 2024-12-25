import pandas as pd
import streamlit as st
from naive_bayes import classify_class

# Tentukan lokasi data
path = './dataset.xlsx'
sheet = 'DATASET TRAINING UNTUK PROGRAM'

# Muat data
data = pd.read_excel(path, sheet_name=sheet)

# # Tambahkan CSS untuk memperindah tampilan
# st.markdown("""
#     <style>
#         body {
#             font-family: 'Arial', sans-serif;
#             background-color: #f4f4f9;
#             color: #333;
#         }
#         .sidebar .sidebar-content {
#             background-color: #2e3d49;
#             color: white;
#         }
#         .stButton>button {
#             background-color: #4CAF50;
#             color: white;
#             border: none;
#             padding: 10px 24px;
#             cursor: pointer;
#             font-size: 16px;
#         }
#         .stButton>button:hover {
#             background-color: #45a049;
#         }
#         .stTitle, .stSubheader {
#             font-size: 24px;
#             color: #2e3d49;
#         }
#         .stDataFrame {
#             background-color: #fff;
#             border-radius: 8px;
#             box-shadow: 0 4px 6px rgba(0,0,0,0.1);
#         }
#         .stExpanderHeader {
#             font-weight: bold;
#             color: #4CAF50;
#         }
#     </style>
# """, unsafe_allow_html=True)

# Tampilkan data dengan kontainer
st.title('Data Training')
st.dataframe(data, use_container_width=True)

# Praproses data
st.title('Praproses Data')

# Penjelasan tentang apa yang dilakukan dalam praproses data
with st.expander("Penjelasan Praproses Data"):
    st.write(
        'Praproses data adalah proses mengubah data mentah menjadi format yang terorganisir dengan baik. '
        'Tujuannya adalah untuk membuat data siap untuk analisis dan model machine learning. '
        'Pada bagian ini, kita akan melakukan praproses data melalui langkah-langkah berikut:'
    )

# Langkah 1: Periksa nilai yang hilang
st.subheader('Langkah 1: Periksa Nilai yang Hilang')
st.write(
    'Pada langkah pertama, kita akan memeriksa apakah ada nilai yang hilang (missing values) dalam dataset. '
    'Nilai yang hilang bisa mempengaruhi kualitas model machine learning yang akan dibuat. Oleh karena itu, kita perlu mengetahui seberapa banyak data yang hilang di setiap kolom.'
)
missing_values = data.isnull().sum()
st.write('Jumlah nilai yang hilang di setiap kolom:')
st.write(missing_values)

# Langkah 2: Hapus nilai yang hilang
st.subheader('Langkah 2: Hapus Nilai yang Hilang')
st.write(
    'Setelah memeriksa nilai yang hilang, langkah selanjutnya adalah menghapus baris yang mengandung nilai hilang. '
    'Hal ini penting untuk memastikan data yang digunakan tidak memiliki kekurangan informasi, yang dapat mempengaruhi akurasi model.'
)
data_clean = data.dropna()
st.write('Data setelah menghapus nilai yang hilang:')
st.dataframe(data_clean, use_container_width=True)

# Langkah 3: Hapus kolom yang tidak diperlukan untuk training
st.subheader('Langkah 3: Hapus Kolom yang Tidak Diperlukan untuk Training')
st.write(
    'Pada langkah ketiga, kita akan menghapus kolom yang tidak relevan untuk model machine learning, seperti kolom "NO". '
    'Kolom ini hanya berfungsi sebagai identifikasi atau urutan, yang tidak memberikan informasi terkait dengan hasil prediksi.'
)
columns_to_drop = ['NO']
data_final = data_clean.drop(columns=columns_to_drop)
st.write('Data setelah menghapus kolom yang tidak diperlukan untuk training:')
st.dataframe(data_final, use_container_width=True)

# Langkah 4: Klasifikasi kelas berdasarkan nilai di setiap kolom
st.subheader('Langkah 4: Klasifikasi Kelas Berdasarkan Nilai di Setiap Kolom')

# Penjelasan mengenai ketentuan klasifikasi nilai
with st.expander("Ketentuan Klasifikasi Nilai"):
    st.write(
        """
        **Ketentuan Klasifikasi Nilai:**
        - **SB (Sangat Baik):** Nilai antara 80 dan 100
        - **B (Baik):** Nilai antara 70 dan 79
        - **C (Cukup):** Nilai antara 60 dan 69
        - **K (Kurang):** Nilai antara 55 dan 59
        - **SK (Sangat Kurang):** Nilai antara 50 dan 54
        """
    )

# Penjelasan proses klasifikasi berdasarkan nilai
st.write(
    "Pada langkah ini, kita akan mengklasifikasikan nilai setiap kolom menjadi kategori kelas tertentu. "
    "Klasifikasi ini penting agar kita dapat mengetahui kualitas atau hasil yang diperoleh berdasarkan nilai yang diberikan. "
    "Nilai-nilai tersebut akan dikategorikan ke dalam beberapa kelas, seperti 'SB' untuk nilai antara 80 dan 100, 'B' untuk nilai antara 70 dan 79, "
    "'C' untuk nilai antara 60 dan 69, dan seterusnya. Dengan menggunakan aturan klasifikasi ini, kita akan memberi label kelas yang sesuai untuk setiap nilai."
)

# Terapkan fungsi klasifikasi ke setiap kolom data
for column in data_final.columns:
    if column not in ['Nama Siswa', 'Hasil']:  # Abaikan kolom 'Nama Siswa' dan 'Hasil' karena mereka tidak memerlukan klasifikasi
        data_final[column] = data_final[column].apply(classify_class)

# Tampilkan data setelah klasifikasi kelas
st.write('Data setelah klasifikasi kelas berdasarkan nilai di setiap kolom:')
st.dataframe(data_final, use_container_width=True)

# Implementasi algoritma Naive Bayes
st.title('Implementasi Algoritma Naive Bayes')
st.write(
    'Naive Bayes adalah algoritma klasifikasi yang didasarkan pada Teorema Bayes. '
    'Algoritma ini disebut "naive" karena mengasumsikan bahwa fitur-fitur bersifat independen satu sama lain. '
    'Pada bagian ini, kita akan mengimplementasikan algoritma Naive Bayes untuk memprediksi kelas siswa berdasarkan data yang diberikan.'
)

# Penjelasan rumus yang digunakan
st.subheader("Penjelasan Rumus-Rumus Naive Bayes")

# Rumus Probabilitas Prior
st.markdown("### Probabilitas Prior (P(C))")
st.latex(r"P(C_k) = \frac{\text{Jumlah Data Kelas } C_k}{\text{Total Jumlah Data}}")
st.write(
    "Probabilitas prior menghitung kemungkinan awal dari suatu kelas tanpa memperhatikan fitur yang diberikan."
)

# Rumus Probabilitas Kondisional
st.markdown("### Probabilitas Kondisional (P(X_i | C_k))")
st.latex(r"P(X_i | C_k) = \frac{\text{Jumlah kemunculan fitur } X_i \text{ dalam kelas } C_k}{\text{Jumlah total data dalam kelas } C_k}")
st.write(
    "Probabilitas kondisional menghitung kemungkinan suatu fitur muncul dalam kelas tertentu."
)

# Rumus Teorema Bayes
st.markdown("### Teorema Bayes (P(C_k | X))")
st.latex(r"P(C_k | X) = \frac{P(C_k) \prod_{i=1}^n P(X_i | C_k)}{P(X)}")
st.write(
    "Teorema Bayes digunakan untuk menghitung probabilitas suatu kelas \(C_k\) "
    "dengan mempertimbangkan semua fitur \(X = (X_1, X_2, ..., X_n)\). "
    "Karena \(P(X)\) adalah konstanta, dalam implementasi kita hanya fokus pada bagian pembilang."
)

# Rumus Prediksi Kelas
st.markdown("### Prediksi Kelas")
st.latex(r"\hat{C} = \arg \max_{C_k} P(C_k) \prod_{i=1}^n P(X_i | C_k)")
st.write(
    "Kelas yang diprediksi (\(\\hat{C}\)) adalah kelas dengan probabilitas tertinggi setelah perhitungan."
)

# Hitung probabilitas prior
prior_prob = data_final['Hasil'].value_counts(normalize=True)

# Hitung probabilitas kondisional
conditional_prob = {}
for column in data_final.columns:
    if column not in ['Nama Siswa', 'Hasil']:
        conditional_prob[column] = data_final.groupby(['Hasil', column]).size() / data_final.groupby('Hasil').size()

# Tampilkan hasil di Streamlit
st.markdown("### Probabilitas Prior untuk Setiap Kelas")
st.write("Probabilitas prior menggambarkan distribusi awal data untuk setiap kelas.")
st.dataframe(prior_prob)

# Tampilkan probabilitas kondisional dengan tab
st.markdown("### Probabilitas Kondisional untuk Setiap Fitur Berdasarkan Kelas")
st.write(
    "Probabilitas kondisional menghitung kemungkinan fitur muncul dalam kelas tertentu berdasarkan data yang tersedia."
)

# Buat tab untuk setiap fitur
tabs = st.tabs([f"Fitur: {feature}" for feature in conditional_prob.keys()])

# Iterasi melalui setiap fitur dan tampilkan di tab
for tab, (feature, prob) in zip(tabs, conditional_prob.items()):
    with tab:
        st.write(f"**Probabilitas Kondisional untuk Fitur: {feature}**")
        st.dataframe(prob)

# Simpan probabilitas sebagai variabel global untuk diakses oleh file lain
global_prior_prob = prior_prob
global_conditional_prob = conditional_prob
