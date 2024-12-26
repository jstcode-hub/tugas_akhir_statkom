import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from training import global_prior_prob, global_conditional_prob
from naive_bayes import prediksi_naive_bayes, compare_results, classify_class
import numpy as np

# Tentukan lokasi data
path = './dataset.xlsx'
sheet_testing = 'DATASET TESTING UNTUK PROGRAM'
sheet_manual = 'HASIL PERHITUNGAN MANUAL UNTUK '

# Baca data
data = pd.read_excel(path, sheet_name=sheet_testing)
data_manual = pd.read_excel(path, sheet_name=sheet_manual)

# **1. Muat Dataset**
st.title("Data Testing dan Analisis Naive Bayes")
st.subheader("1. Muat Dataset")

# Tampilkan data awal dalam container
st.write("Data Testing:")
st.dataframe(data, use_container_width=True)

# **2. Praproses Data**
st.subheader("2. Praproses Data")

# Langkah 1: Periksa nilai yang hilang
with st.expander("Langkah 1: Periksa Nilai yang Hilang"):
    missing_values = data.isnull().sum()
    st.write("Jumlah nilai yang hilang di setiap kolom:")
    st.write(missing_values)

# **Langkah 2: Hapus kolom yang tidak diperlukan**
with st.expander("Langkah 2: Hapus Kolom yang Tidak Diperlukan"):
    columns_to_drop = ['NO']  # Kolom yang ingin dihapus
    data_final = data.drop(columns=columns_to_drop, errors="ignore")  # Hapus kolom tersebut
    st.write("Data setelah penghapusan kolom yang tidak diperlukan:")
    st.dataframe(data_final, use_container_width=True)

# **Langkah 3: Klasifikasi nilai**
with st.expander("Langkah 3: Klasifikasi Nilai"):
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

# Penjelasan Proses:
st.write("""
    Pada bagian ini, kami menggunakan algoritma Naive Bayes untuk memprediksi kelas dari data testing 
    yang telah diproses sebelumnya. Proses prediksi dilakukan dengan menggunakan informasi probabilitas 
    yang dihitung selama tahap pelatihan, seperti prior probabilities dan conditional probabilities.

    Berikut adalah langkah-langkah yang dilakukan:
    1. **Menghitung Probabilitas Posterior**: Algoritma Naive Bayes menghitung probabilitas posterior 
       untuk setiap kelas berdasarkan fitur yang ada di data testing.
    2. **Memilih Kelas dengan Probabilitas Tertinggi**: Kelas dengan probabilitas tertinggi dipilih sebagai 
       hasil prediksi untuk setiap data.
    3. **Memasukkan Hasil Prediksi**: Hasil prediksi disimpan dalam kolom 'Hasil' pada dataset.
""")

# Lakukan prediksi
predictions = prediksi_naive_bayes(data_final, global_prior_prob, global_conditional_prob)

# Memasukkan hasil prediksi ke dalam kolom 'Hasil' di data_final
data_final['Hasil'] = predictions

# Tampilkan hasil prediksi dalam bentuk tabel
st.write("Hasil prediksi kelas untuk data testing:")
st.dataframe(data_final, use_container_width=True)

# Statistik prediksi
st.write("""
    Berikut adalah statistik mengenai hasil prediksi yang dilakukan oleh model Naive Bayes:
    - **Jumlah dan persentase kelas yang diprediksi**: Statistik ini akan memberikan gambaran tentang distribusi 
      kelas yang diprediksi oleh model untuk dataset testing.
""")

# Hitung dan tampilkan jumlah per kelas pada kolom 'Hasil'
predictions_count = data_final['Hasil'].value_counts()
st.write("Statistik jumlah kelas yang diprediksi:")

# Grafik distribusi hasil prediksi
fig, ax = plt.subplots(figsize=(8, 6))
predictions_count.plot(kind='bar', ax=ax, color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6'])
ax.set_title("Distribusi Hasil Prediksi", fontsize=16)
ax.set_xlabel("Kelas", fontsize=12)
ax.set_ylabel("Jumlah", fontsize=12)
ax.tick_params(axis='x', rotation=45)
st.pyplot(fig)

# **5. Evaluasi Perbandingan dengan Data Manual**
st.subheader("5. Perbandingan Hasil Prediksi dengan Confusion Matrix")

# Bandingkan hasil prediksi dengan data manual
result = compare_results(data_manual, data_final, 'Hasil')

# Jika perbandingan berhasil, hitung confusion matrix
if isinstance(result, dict):
    # Ambil hasil prediksi dan data manual
    y_true = data_manual['Hasil']  # Data manual
    y_pred = data_final['Hasil']   # Hasil prediksi

    # Dapatkan semua label unik yang ada dalam y_true dan y_pred
    labels = sorted(set(y_true).union(set(y_pred)))

    # Hitung confusion matrix secara manual
    cm = [[0 for _ in range(len(labels))] for _ in range(len(labels))]
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    for true_label, pred_label in zip(y_true, y_pred):
        true_idx = label_to_index[true_label]
        pred_idx = label_to_index[pred_label]
        cm[true_idx][pred_idx] += 1

    # Visualisasi Confusion Matrix secara manual
    cm = np.array(cm)  # Ubah ke numpy array untuk mempermudah pengolahan
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cm, cmap='Blues')  # Ganti ini jika ingin warna lain

    # Tambahkan anotasi nilai pada setiap sel
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')

    # Atur label sumbu X dan Y
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel('Prediksi')
    ax.set_ylabel('Data Manual')
    ax.set_title('Confusion Matrix - Hasil Prediksi vs Data Manual')

    st.write("""
    **Penjelasan tentang Confusion Matrix:**
    - **Diagonal utama (dari kiri atas ke kanan bawah):** menunjukkan jumlah data yang diprediksi dengan benar untuk setiap kelas.
    - **Sel di luar diagonal utama:** menunjukkan jumlah kesalahan prediksi. Misalnya, sel di kolom "B" dan baris "SB" menunjukkan berapa kali prediksi "B" diklasifikasikan sebagai "SB".
    """)
    st.pyplot(fig)

    # Tampilkan statistik tambahan
    st.write(f"Jumlah data yang benar: **{result['total_correct']}**")
    st.write(f"Persentase data yang benar: **{result['correct_percentage']:.2f}%**")
    st.write(f"Jumlah data yang berbeda: **{result['total_differences']}**")
    st.write(f"Persentase data yang berbeda: **{result['difference_percentage']:.2f}%**")

    # Tampilan kiri dan kanan
    col1, col2 = st.columns(2)

    # Data yang berbeda di kolom kiri
    with col1:
        if not result['differences'].empty:
            st.write("**Data yang berbeda:**")
            st.dataframe(result['differences'][['Dataset Manual', 'Dataset Prediksi']])
        else:
            st.write("Semua data cocok dengan hasil manual.")

    # Data yang benar di kolom kanan
    with col2:
        if not result['correct'].empty:
            st.write("**Data yang benar:**")
            st.dataframe(result['correct'][['Dataset Manual', 'Dataset Prediksi']])
        else:
            st.write("Tidak ada data yang benar.")

else:
    st.error(result)
