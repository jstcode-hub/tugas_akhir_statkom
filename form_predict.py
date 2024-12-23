import pandas as pd
import streamlit as st
from naive_bayes import prediksi_naive_bayes
from training import global_prior_prob, global_conditional_prob, classify_class

# Halaman Form untuk Prediksi Nilai dari User Input
st.title("Prediksi Kelas Berdasarkan Nilai Mata Pelajaran")

# Buat form input nilai
with st.form("form_prediksi_mata_pelajaran"):
    st.subheader("Masukkan Nilai Anda")
    
    # Input nilai untuk masing-masing mata pelajaran
    nilai_matematika = st.number_input(
        "Nilai Matematika:", min_value=0.0, max_value=100.0, step=0.1
    )
    nilai_ipa = st.number_input(
        "Nilai IPA:", min_value=0.0, max_value=100.0, step=0.1
    )
    nilai_b_inggris = st.number_input(
        "Nilai Bahasa Inggris:", min_value=0.0, max_value=100.0, step=0.1
    )
    nilai_b_indonesia = st.number_input(
        "Nilai Bahasa Indonesia:", min_value=0.0, max_value=100.0, step=0.1
    )
    
    # Tombol submit form
    submitted = st.form_submit_button("Prediksi Kelas")

# Lakukan prediksi setelah form disubmit
if submitted:
    # Buat DataFrame dari input pengguna
    user_data = pd.DataFrame([{
        "Matematika": nilai_matematika,
        "IPA": nilai_ipa,
        "B. Inggris": nilai_b_inggris,
        "B.Indonesia": nilai_b_indonesia
    }])
    
    # Terapkan klasifikasi kelas berdasarkan nilai
    for column in user_data.columns:
        user_data[column] = user_data[column].apply(classify_class)
    
    # Prediksi kelas menggunakan Naive Bayes
    user_prediction = prediksi_naive_bayes(user_data, global_prior_prob, global_conditional_prob)
    
    # Tampilkan hasil prediksi
    st.write("### Hasil Prediksi")
    st.write("Kelas yang diprediksi:", user_prediction[0])
    
    # Tampilkan data yang dimasukkan pengguna
    st.write("Data yang Anda masukkan (setelah klasifikasi):")
    st.dataframe(user_data)
