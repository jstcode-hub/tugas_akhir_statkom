import pandas as pd
import streamlit as st
from naive_bayes import prediksi_naive_bayes
from training import global_prior_prob, global_conditional_prob, classify_class

# Halaman Form untuk Prediksi Nilai dari User Input
st.title("Prediksi Kelas Berdasarkan Nilai Mata Pelajaran")

# Menambahkan deskripsi untuk memberikan informasi kepada pengguna
st.write(
    "Selamat datang di sistem prediksi kelas berdasarkan nilai mata pelajaran. "
    "Masukkan nilai Anda untuk beberapa mata pelajaran, dan sistem akan memprediksi kelas yang sesuai "
    "menggunakan algoritma Naive Bayes. Pastikan nilai yang dimasukkan berada dalam rentang 0-100."
)

# Membuat form input nilai dengan tata letak yang lebih baik
with st.form("form_prediksi_mata_pelajaran"):
    st.subheader("Masukkan Nilai Anda")

    # Menambahkan penjelasan singkat di bawah setiap input untuk memberikan instruksi
    nilai_matematika = st.number_input(
        "Nilai Matematika:", min_value=0.0, max_value=100.0, step=1.0, help="Masukkan nilai Matematika Anda (0-100)."
    )
    nilai_ipa = st.number_input(
        "Nilai IPA:", min_value=0.0, max_value=100.0, step=1.0, help="Masukkan nilai IPA Anda (0-100)."
    )
    nilai_b_inggris = st.number_input(
        "Nilai Bahasa Inggris:", min_value=0.0, max_value=100.0, step=1.0, help="Masukkan nilai Bahasa Inggris Anda (0-100)."
    )
    nilai_b_indonesia = st.number_input(
        "Nilai Bahasa Indonesia:", min_value=0.0, max_value=100.0, step=1.0, help="Masukkan nilai Bahasa Indonesia Anda (0-100)."
    )
    
    # Tombol submit form dengan penataan tombol yang lebih jelas
    submitted = st.form_submit_button("Prediksi Kelas")

# Lakukan prediksi setelah form disubmit
if submitted:
    # Menampilkan progress bar sementara menunggu prediksi
    with st.spinner('Melakukan prediksi, harap tunggu...'):
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

        # Hitung probabilitas untuk setiap kelas dan tampilkan probabilitas setiap mata pelajaran
        class_probabilities = {}
        feature_probabilities = {}  # Menyimpan probabilitas untuk setiap mata pelajaran per kelas
        
        for class_name in global_prior_prob.index:
            prob = global_prior_prob[class_name]  # Probabilitas prior untuk kelas
            feature_probabilities[class_name] = {}  # Inisialisasi dictionary untuk probabilitas setiap fitur per kelas
            for column in user_data.columns:
                if column not in ['Nama Siswa', 'Hasil']:
                    feature_probabilities[class_name][column] = global_conditional_prob.get(column, {}).get((class_name, user_data[column].iloc[0]), 0)
                    prob *= feature_probabilities[class_name][column]  # Mengalikan dengan probabilitas kondisional fitur

            class_probabilities[class_name] = prob

    # Tampilkan hasil prediksi
    st.write("### Hasil Prediksi")
    st.markdown(f"**Kelas yang diprediksi**: {user_prediction[0]}", unsafe_allow_html=True)

    # Tampilkan data yang dimasukkan pengguna
    st.write("### Data yang Anda Masukkan (Setelah Klasifikasi):")
    st.dataframe(user_data)

    # Tampilkan nilai global_prior_prob
    st.write("### Probabilitas Prior (global_prior_prob):")
    st.write(global_prior_prob)

    # Tampilkan nilai global_conditional_prob
    st.write("### Probabilitas Kondisional (global_conditional_prob):")
    st.write(global_conditional_prob)

    # Menampilkan probabilitas untuk setiap mata pelajaran per kelas
    st.write("### Probabilitas Setiap Mata Pelajaran per Kelas:")
    for class_name, prob_dict in feature_probabilities.items():
        st.write(f"**Kelas: {class_name}**")
        prob_df = pd.DataFrame(list(prob_dict.items()), columns=["Mata Pelajaran", "Probabilitas"])
        prob_df["Probabilitas"] = prob_df["Probabilitas"].round(4)  # Membulatkan nilai untuk tampilan yang lebih rapi
        st.dataframe(prob_df)
    
    # Tampilkan probabilitas akhir untuk setiap kelas
    st.write("### Probabilitas Kelas Berdasarkan Prediksi:")
    prob_df = pd.DataFrame(list(class_probabilities.items()), columns=["Kelas", "Probabilitas"])
    prob_df["Probabilitas"] = prob_df["Probabilitas"].round(4)  # Membulatkan nilai untuk tampilan yang lebih rapi
    st.dataframe(prob_df)

    # Menambahkan penjelasan terkait hasil prediksi
    st.write(
        "Berdasarkan nilai yang Anda masukkan, sistem memprediksi kelas Anda. "
        "Probabilitas untuk setiap mata pelajaran dan untuk setiap kelas juga ditampilkan di atas. "
        "Jika Anda ingin mencoba dengan nilai yang berbeda, Anda bisa mengisi form lagi."
    )
