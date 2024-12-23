import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from training import global_prior_prob, global_conditional_prob
from naive_bayes import prediksi_naive_bayes, compare_results, classify_class

# **1. Muat Dataset**
st.title("Data Testing dan Analisis Naive Bayes")
st.subheader("1. Muat Dataset")

# Tentukan lokasi data
path = './dataset.xlsx'
sheet_testing = 'DATASET TESTING UNTUK PROGRAM'
sheet_manual = 'HASIL PERHITUNGAN MANUAL UNTUK '

# Baca data
data = pd.read_excel(path, sheet_name=sheet_testing)
data_manual = pd.read_excel(path, sheet_name=sheet_manual)

# Tampilkan data awal
st.write("Data Testing:")
st.dataframe(data, use_container_width=True)

# **2. Praproses Data**
st.subheader("2. Praproses Data")

# Langkah 1: Periksa nilai yang hilang
st.write("**Langkah 1: Periksa Nilai yang Hilang**")
missing_values = data.isnull().sum()
st.write("Jumlah nilai yang hilang di setiap kolom:")
st.write(missing_values)

# Langkah 2: Hapus kolom yang tidak diperlukan
st.write("**Langkah 2: Hapus Kolom yang Tidak Diperlukan**")
columns_to_drop = ['NO']
data_final = data.drop(columns=columns_to_drop, errors="ignore")
st.write("Data setelah penghapusan kolom:")
st.dataframe(data_final, use_container_width=True)

# Langkah 3: Klasifikasi nilai
st.write("**Langkah 3: Klasifikasi Nilai**")
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

# Terapkan klasifikasi
for column in data_final.columns:
    if column not in ['Nama Siswa', 'Hasil']:
        data_final[column] = data_final[column].apply(classify_class)

st.write("Data setelah klasifikasi nilai:")
st.dataframe(data_final, use_container_width=True)

# **3. Prediksi dengan Naive Bayes**
st.subheader("3. Prediksi Kelas Menggunakan Naive Bayes")

# Lakukan prediksi
predictions = prediksi_naive_bayes(data_final, global_prior_prob, global_conditional_prob)
data_final['Hasil'] = predictions

# Tampilkan hasil prediksi
st.write("Hasil prediksi kelas:")
st.dataframe(data_final, use_container_width=True)

# Statistik prediksi
st.write("Statistik hasil prediksi:")
predictions_count = data_final['Hasil'].value_counts()
st.write(predictions_count)

# **4. Visualisasi Hasil Prediksi**
st.subheader("4. Visualisasi Hasil Prediksi")

# Grafik distribusi hasil prediksi
fig, ax = plt.subplots()
predictions_count.plot(kind='bar', ax=ax, color=['blue', 'green', 'orange', 'red', 'purple'])
ax.set_title("Distribusi Hasil Prediksi")
ax.set_xlabel("Kelas")
ax.set_ylabel("Jumlah")
st.pyplot(fig)

# **5. Evaluasi Perbandingan dengan Data Manual**
st.subheader("5. Perbandingan Hasil Prediksi")

# Bandingkan hasil prediksi dengan data manual
result = compare_results(data_manual, data_final, 'Hasil')

# Tampilkan hasil perbandingan
if isinstance(result, dict):
    st.write(f"Jumlah data yang benar: **{result['total_correct']}**")
    st.write(f"Persentase data yang benar: **{result['correct_percentage']:.2f}%**")
    st.write(f"Jumlah data yang berbeda: **{result['total_differences']}**")
    st.write(f"Persentase data yang berbeda: **{result['difference_percentage']:.2f}%**")

    if not result['differences'].empty:
        st.write("**Data yang berbeda:**")
        st.dataframe(result['differences'][['Dataset Manual', 'Dataset Prediksi']])
    else:
        st.write("Semua data cocok dengan hasil manual.")

    if not result['correct'].empty:
        st.write("**Data yang benar:**")
        st.dataframe(result['correct'][['Dataset Manual', 'Dataset Prediksi']])
else:
    st.error(result)
