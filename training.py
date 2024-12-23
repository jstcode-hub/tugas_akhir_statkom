import pandas as pd
import streamlit as st
from naive_bayes import classify_class

# Tentukan lokasi data
path = './dataset.xlsx'
sheet = 'DATASET TRAINING UNTUK PROGRAM'

# Muat data
data = pd.read_excel(path, sheet_name=sheet)

# Tampilkan data
st.title('Data Training')
st.dataframe(data, use_container_width=True)

# Praproses data
st.title('Praproses Data')
st.write(
    'Praproses data adalah proses mengubah data mentah menjadi format yang terorganisir dengan baik. '
    'Tujuannya adalah untuk membuat data siap untuk analisis dan model machine learning. '
    'Pada bagian ini, kita akan melakukan praproses data melalui langkah-langkah berikut:'
)

# Langkah 1: Periksa nilai yang hilang
st.subheader('Langkah 1: Periksa Nilai yang Hilang')
missing_values = data.isnull().sum()
st.write('Jumlah nilai yang hilang di setiap kolom:')
st.write(missing_values)

# Langkah 2: Hapus nilai yang hilang
st.subheader('Langkah 2: Hapus Nilai yang Hilang')
data_clean = data.dropna()
st.write('Data setelah menghapus nilai yang hilang:')
st.dataframe(data_clean, use_container_width=True)

# Langkah 3: Hapus kolom yang tidak diperlukan untuk training
st.subheader('Langkah 3: Hapus Kolom yang Tidak Diperlukan untuk Training')
columns_to_drop = ['NO']
data_final = data_clean.drop(columns=columns_to_drop)
st.write('Data setelah menghapus kolom yang tidak diperlukan untuk training:')
st.dataframe(data_final, use_container_width=True)

# Langkah 4: Klasifikasi kelas berdasarkan nilai di setiap kolom
st.subheader('Langkah 4: Klasifikasi Kelas Berdasarkan Nilai di Setiap Kolom')

# Penjelasan klasifikasi nilai
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

# Terapkan fungsi ke setiap kolom
for column in data_final.columns:
    if column not in ['Nama Siswa', 'Hasil']:
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

# Hitung probabilitas prior
prior_prob = data_final['Hasil'].value_counts(normalize=True)

# Hitung probabilitas kondisional
conditional_prob = {}
for column in data_final.columns:
    if column not in ['Nama Siswa', 'Hasil']:
        conditional_prob[column] = data_final.groupby(['Hasil', column]).size() / data_final.groupby('Hasil').size()

# Tampilkan hasil di Streamlit
st.write('Probabilitas prior untuk setiap kelas:')
st.write(prior_prob)

st.write('Probabilitas kondisional untuk setiap fitur berdasarkan kelas:')
st.write(conditional_prob)

# Simpan probabilitas sebagai variabel global untuk diakses oleh file lain
global_prior_prob = prior_prob
global_conditional_prob = conditional_prob